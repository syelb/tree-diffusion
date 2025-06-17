from absl import app
from absl import flags
from absl import logging

from td.environments import Environment, environments
from td.learning.tokenizer import Tokenizer
from td.learning.gpt import TreeDiffusion, TransformerConfig
from td.samplers import ConstrainedRandomSampler
from td.learning.constrained_decoding import ar_decoder

from typing import List, Tuple, Dict
from td.samplers.mutator import Mutation
from td.learning.constrained_decoding import (
    DecoderState,
)
import tqdm
import pickle
import numpy as np
from dataclasses import dataclass
import torch
import os
import uuid

from typing import TypeVar

T = TypeVar("T")


flags.DEFINE_string("checkpoint_name", None, "Path to the checkpoint to evaluate")
flags.DEFINE_string("ar_checkpoint_name", None, "Path to the AR checkpoint.")
flags.DEFINE_string("problem_filename", None, "Number of problems to evaluate")
flags.DEFINE_integer("max_steps", 100, "Maximum number of steps to take")
flags.DEFINE_integer("max_depth", 30, "Maximum number of depth to take")
flags.DEFINE_integer("evaluation_batch_size", 16, "Batch size for evaluation")
flags.DEFINE_integer("num_replicas", 32, "Batch size for evaluation")
flags.DEFINE_float("temperature", 0.7, "Temperature for sampling")
flags.DEFINE_string("evaluation_dir", "evals", "Evaluations directory")
flags.DEFINE_bool("wandb", True, "Log to wandb")
flags.DEFINE_string("device", "cuda", "Device to use")
# added 0n 01/04/25
flags.DEFINE_string("resume_from", None, "Path to resume training from")

FLAGS = flags.FLAGS


def generate_uuid():
    return str(uuid.uuid4())


def ar_init(checkpoint_name, target_images):
    with open(checkpoint_name, "rb") as f:
        state = pickle.load(f)

    config = state["config"]

    env_name = config["env"]
    image_model = config["image_model"]
    d_model = config["d_model"]
    n_layers = config["n_layers"]
    num_heads = config["num_heads"]
    max_sequence_length = config["max_sequence_length"]

    env: Environment = environments[env_name]()
    tokenizer = Tokenizer(
        env.grammar,
        max_token_length=max_sequence_length,
        max_sequence_length=max_sequence_length,
    )

    model = TreeDiffusion(
        TransformerConfig(
            vocab_size=tokenizer.vocabulary_size,
            max_seq_len=tokenizer.max_sequence_length,
            n_layer=n_layers,
            n_head=num_heads,
            n_embd=d_model,
        ),
        input_channels=env.compiled_shape[-1],
        image_model_name=image_model,
    )
    model.load_state_dict(state["model"])
    model.cuda()

    rv = ar_decoder(
        model,
        env,
        tokenizer,
        config["num_image_tokens"],
        target_images,
        temperature=0.1,
    )

    del model
    torch.cuda.empty_cache()
    return rv


@dataclass
class MutationWithLogprobs:
    mutation: Mutation
    token_logprobs: np.ndarray

    @property
    def position_logprob(self):
        return self.token_logprobs[0]

    def __eq__(self, other):
        return self.mutation == other.mutation

    def __hash__(self):
        return hash(self.mutation)


def sample_model_top_k(
    model: TreeDiffusion,
    env: Environment,
    tokenizer: Tokenizer,
    current_expressions,
    target_images,
    current_images=None,
    temperature=1.0,
    k=3,
    replace_invalid_with_none=False,
) -> List[List[MutationWithLogprobs]]:
    with torch.inference_mode():
        device = next(model.parameters()).device
        current_images = (
            torch.tensor(np.array([env.compile(x) for x in current_expressions]))
            .float()
            .permute(0, 3, 1, 2)
            .to(device)
            if current_images is None
            else current_images
        )
        image_embeddings = model.image_embeddings(target_images, current_images)
        image_embeddings = image_embeddings.repeat_interleave(k, dim=0)

        context_tokens = [tokenizer._tokenize_one(x) for x in current_expressions]
        start_decoding_positions = [len(x) for x in context_tokens]
        # Pad to max length.
        max_length = tokenizer.max_token_length
        current_tokens = torch.tensor(
            [
                x
                + [tokenizer.sos_token]
                + [tokenizer.pad_token] * (max_length - len(x) - 1)
                for x in context_tokens
            ]
        ).to(device)
        start_decoding_positions = torch.tensor(start_decoding_positions)

        current_tokens = current_tokens.repeat_interleave(k, dim=0)
        start_decoding_positions = start_decoding_positions.repeat_interleave(k, dim=0)
        logprobs = torch.zeros_like(current_tokens, dtype=torch.float32, device="cpu")

        decode_states = [
            DecoderState(env.grammar, tokenizer, x)
            for x in current_expressions
            for _ in range(k)
        ]
        end_indexes = [-1] * len(decode_states)

        current_position = 0
        k_cache = None
        v_cache = None

        while (
            not all(x._decode_state == DecoderState.States.END for x in decode_states)
            and current_position < max_length - 1
        ):
            logits, k_cache, v_cache = model.transformer(
                current_tokens[:, [current_position]],
                extra_emb=image_embeddings,
                k_cache=k_cache,
                v_cache=v_cache,
                start_idx=current_position,
            )
            logits = logits.cpu()

            logits_for_positions = logits[:, 0, :]
            logits_for_positions = logits_for_positions / temperature
            decode_mask = torch.tensor(np.stack([x.mask for x in decode_states])).bool()
            logits_for_positions = torch.where(
                decode_mask, logits_for_positions, -torch.tensor(float("inf"))
            )
            probs = torch.nn.functional.softmax(logits_for_positions, dim=-1)
            sampled_tokens = torch.multinomial(probs, 1).squeeze(-1)
            logprobs[:, [current_position]] = torch.log(
                probs[torch.arange(len(probs)), sampled_tokens]
            ).unsqueeze(-1)

            # Update decode states.
            for i, token in enumerate(np.array(sampled_tokens)):
                if current_position < start_decoding_positions[i]:
                    continue

                if decode_states[i]._decode_state == DecoderState.States.END:
                    if end_indexes[i] == -1:
                        end_indexes[i] = current_position

                    continue

                decode_states[i].feed_token(token)
                current_tokens[i, current_position + 1] = sampled_tokens[i].item()

                if decode_states[i]._decode_state == DecoderState.States.END:
                    if end_indexes[i] == -1:
                        end_indexes[i] = current_position + 1

            # Update current tokens and positions.
            current_position += 1

        ungrouped = [
            MutationWithLogprobs(
                x.get_mutation(),
                logprobs[
                    i,
                    start_decoding_positions[i] : end_indexes[i],
                ].numpy(),
            )
            for i, x in enumerate(decode_states)
        ]

        grouped = [set() for _ in range(len(current_expressions))]

        for i, x in enumerate(ungrouped):
            if end_indexes[i] == -1:
                continue
            grouped[i // k].add(x)

        for i, x in enumerate(grouped):
            grouped[i] = list(x)
            if replace_invalid_with_none:
                grouped[i] += [None] * (k - len(x))

        return grouped


@dataclass
class BeamSearchResult:
    parent: Dict[str, str]
    actions: Dict[Tuple[str, str], Mutation]
    levels: List[List[str]]
    finished: bool = False


def batched_beam_search(
    model: TreeDiffusion,
    target_image,
    starting_expressions,
    env,
    tokenizer,
    beam_size=16,
    k=3,
    max_depth=30,
    temperature=1.0,
    verbose=False,
    target_idx=None  # Keep track of the target image index

):
    device = next(model.parameters()).device
    starting_expressions = list(set(starting_expressions))

    batched_target_image = (
        torch.tensor(target_image).float().permute(2, 0, 1).to(device)
    )
    repeat_len = max(len(starting_expressions), beam_size) * k
    batched_target_image = batched_target_image.unsqueeze(0).repeat(repeat_len, 1, 1, 1)

    visited = set(starting_expressions)
    queue = [(e, 0) for e in starting_expressions]

    depth = 0
    best_value = float("-inf")
    best_one = None
    expansions = 0

    with tqdm.tqdm(total=max_depth, disable=not verbose) as pbar:
        while depth < max_depth:
            compiled_expressions = [env.compile(e) for e, _ in queue]
            if len(compiled_expressions) == 0:
                return False, expansions
            if any(
                env.goal_reached(current_image, target_image)
                for current_image in compiled_expressions
            ):
                return True, expansions, best_one

            current_best, current_best_value = None, float("-inf")
            for i, (e, v) in enumerate(queue):
                val = env._goal_checker.goal_reached_value(
                    compiled_expressions[i], target_image
                )
                if val > current_best_value:
                    current_best = e
                    current_best_value = val

            if current_best_value > best_value:
                best_value = current_best_value
                best_one = current_best

            pbar.set_postfix(
                {
                    "Best Value": best_value,
                    "Queue Length": len(queue),
                    "Depth": depth,
                }
            )

            queue = queue[:beam_size]
            current_expressions = [x[0] for x in queue]
            model_result = sample_model_top_k(
                model,
                env,
                tokenizer,
                current_expressions[: len(current_expressions)],
                batched_target_image[: len(current_expressions)],
                k=k,
                temperature=temperature,
            )

            new_queue = []

            for expression_i in range(len(current_expressions)):
                value_so_far = queue[expression_i][1]

                for mutation_i in range(len(model_result[expression_i])):
                    mutation = model_result[expression_i][mutation_i]
                    if mutation is None:
                        continue

                    new_expression = mutation.mutation.apply(
                        current_expressions[expression_i]
                    )
                    if new_expression in visited:
                        continue

                    visited.add(new_expression)
                    new_queue.append(
                        (new_expression, value_so_far + mutation.position_logprob)
                    )

            # queue_expressions = [x[0] for x in new_queue]
            # queue_compiled = np.array([env.compile(x) for x in queue_expressions])
            # queue_compiled_torch = (
            #     torch.tensor(queue_compiled).float().permute(0, 3, 1, 2).to(device)
            # )
            # values = compute_value(
            #     batched_target_image[: len(queue_compiled_torch)],
            #     queue_compiled_torch,
            # ).squeeze(-1)
            # new_queue = [(x[0], values[i]) for i, x in enumerate(new_queue)]

            queue = sorted(new_queue, key=lambda x: -x[1])
            expansions += len(queue)
            depth += 1
            pbar.update(1)

    return False, expansions, best_one

# commented out on 01/04/25 and other part (up to state = added)
# def load_model(checkpoint_name, device):
#     with open(checkpoint_name, "rb") as f:
#         state = pickle.load(f)

import glob
from os.path import join, getmtime

def load_model(checkpoint_name, device):
    # If resume_from is provided, find the latest checkpoint in that directory
    if FLAGS.resume_from is not None:
        checkpoint_dir = FLAGS.resume_from
        # Find all checkpoint files in the directory (with .pt extension)
        checkpoint_files = glob.glob(join(checkpoint_dir, "*.pt"))
        
        if not checkpoint_files:
            raise ValueError(f"No checkpoint files (.pt) found in directory: {checkpoint_dir}")
            
        # Sort by modification time to get the latest checkpoint
        latest_checkpoint = max(checkpoint_files, key=getmtime)
        checkpoint_path = latest_checkpoint
        logging.info(f"Resuming from latest checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = checkpoint_name
        
    with open(checkpoint_path, "rb") as f:
        state = pickle.load(f)

    # Rest of the function remains the same...


    config = state["config"]

    env_name = config["env"]
    image_model = config["image_model"]
    d_model = config["d_model"]
    n_layers = config["n_layers"]
    num_heads = config["num_heads"]
    max_sequence_length = config["max_sequence_length"]
    target_observation = config["target_observation"]

    for key, value in config.items():
        logging.info(f"{key}: {value}")

    env: Environment = environments[env_name]()
    sampler = ConstrainedRandomSampler(env.grammar)
    tokenizer = Tokenizer(
        env.grammar,
        max_token_length=max_sequence_length,
        max_sequence_length=max_sequence_length,
    )

    model = TreeDiffusion(
        TransformerConfig(
            vocab_size=tokenizer.vocabulary_size,
            max_seq_len=tokenizer.max_sequence_length,
            n_layer=n_layers,
            n_head=num_heads,
            n_embd=d_model,
        ),
        input_channels=env.compiled_shape[-1],
        image_model_name=image_model,
    )

    model.load_state_dict(state["model"])
    model.to(device)

    return model, env, tokenizer, sampler, target_observation, config


# def main(argv):
#     logging.info(f"Evaluating {FLAGS.checkpoint_name}")

#     if not os.path.exists(FLAGS.evaluation_dir):
#         os.makedirs(FLAGS.evaluation_dir)

#     local_run_id = generate_uuid()
#     logging.info(f"Local run id: {local_run_id}")

#     save_filename = os.path.join(FLAGS.evaluation_dir, f"{local_run_id}.pkl")

#     td_model, env, tokenizer, sampler, target_observation, _ = load_model(
#         FLAGS.checkpoint_name, FLAGS.device
#     )

#     with open(FLAGS.problem_filename, "rb") as f:
#         target_expressions = pickle.load(f)

#     solved_so_far = 0
#     seen_so_far = 0

#     num_expansions_required = np.zeros(len(target_expressions)) + np.inf

#     for i in tqdm.trange(len(target_expressions)):
#         target_image = env.compile(target_expressions[i])

#         with open(f'saved_images/target_image_{i}.pkl', 'wb') as f:
#             pickle.dump(target_image, f)
        
#         target_image_torch = (
#             torch.tensor(target_image[None]).float().permute(0, 3, 1, 2).to("cuda")
#         )
#         replicated_target_image = target_image_torch.repeat(64, 1, 1, 1)

#         initial_expressions = ar_init(
#             FLAGS.ar_checkpoint_name,
#             replicated_target_image,
#         )

#         # Are any of the initial expressions already solved?
#         solved = False
#         for e_i, e in enumerate(initial_expressions):
#             if env.goal_reached(env.compile(e), target_image):
#                 solved = True
#                 num_expansions = e_i + 1

#         if not solved:
#             solved, num_expansions, best_expression = batched_beam_search(
#                 td_model,
#                 target_image,
#                 initial_expressions,
#                 env,
#                 tokenizer,
#                 beam_size=64,
#                 verbose=True,
#                 max_depth=FLAGS.max_depth,
#                 k=3,
#             )
#             num_expansions += len(initial_expressions)

#         if solved:
#             solved_so_far += 1
#             num_expansions_required[i] = num_expansions

#         seen_so_far += 1

#         logging.info(
#             f"Solved {solved_so_far}/{seen_so_far} ({solved_so_far/seen_so_far:.2f})"
#         )

#         with open(save_filename, "wb") as f:
#             pickle.dump(
#                 {
#                     "num_expansions_required": num_expansions_required,
#                     "solved_so_far": solved_so_far,
#                     "seen_so_far": seen_so_far,
#                 },
#                 f,
#             )
#         if solved: 
#             with open(f'saved_images/final_solved_expression_{i}.pkl', 'wb') as f:
#                 pickle.dump(best_expression, f)  # Save the best expression
            
#             final_compiled_image = env.compile(best_expression)
            
#             with open(f'saved_images/final_solved_image_{i}.pkl', 'wb') as f:
#                 pickle.dump(final_compiled_image, f)
#         else:
#             with open(f'saved_images/final_unsolved_expression_{i}.pkl', 'wb') as f:
#                 pickle.dump(best_expression, f)  # Save the best expression
            
#             final_compiled_image = env.compile(best_expression)
            
#             with open(f'saved_images/final_unsolved_image_{i}.pkl', 'wb') as f:
#                 pickle.dump(final_compiled_image, f)

def main(argv):
    logging.info(f"Evaluating {FLAGS.checkpoint_name}")

    if not os.path.exists(FLAGS.evaluation_dir):
        os.makedirs(FLAGS.evaluation_dir)

    local_run_id = generate_uuid()
    logging.info(f"Local run id: {local_run_id}")

    save_filename = os.path.join(FLAGS.evaluation_dir, f"{local_run_id}.pkl")

    td_model, env, tokenizer, sampler, target_observation, _ = load_model(
        FLAGS.checkpoint_name, FLAGS.device
    )

    with open(FLAGS.problem_filename, "rb") as f:
        graph_silhouette_data = pickle.load(f)

    # We'll now process all graphs instead of just graph_1
    for graph_key in sorted(graph_silhouette_data.keys()):  # Sort to process in order (graph_1, graph_2, etc.)
        logging.info(f"Processing {graph_key}")
        targets = list(graph_silhouette_data[graph_key].keys())  # Get all expressions for this graph

        solved_so_far = 0
        seen_so_far = 0

        # Process each expression in the current graph
        for i, exp_key in enumerate(targets):
            target_expression = graph_silhouette_data[graph_key][exp_key]["target_expression"]
            target_image = env.compile(target_expression)
            
            target_image_torch = (
                torch.tensor(target_image[None]).float().permute(0, 3, 1, 2).to("cuda")
            )
            replicated_target_image = target_image_torch.repeat(64, 1, 1, 1)

            initial_expressions = ar_init(
                FLAGS.ar_checkpoint_name,
                replicated_target_image,
            )

            # Check for initial solutions
            solved = False
            num_expansions = 0  # Initialize to 0
            for e_i, e in enumerate(initial_expressions):
                if env.goal_reached(env.compile(e), target_image):
                    solved = True
                    num_expansions = e_i + 1
                    best_expression = e
                    break

            # If not solved initially, try beam search
            if not solved:
                solved, num_expansions, best_expression = batched_beam_search(
                    td_model,
                    target_image,
                    initial_expressions,
                    env,
                    tokenizer,
                    beam_size=64,
                    verbose=True,
                    max_depth=FLAGS.max_depth,
                    k=3,
                )
                num_expansions += len(initial_expressions)

            # Store all results in our data structure
            graph_silhouette_data[graph_key][exp_key].update({
                "solution_expression": best_expression,
                "solution_image": env.compile(best_expression),
                "solved_status": solved,  # Store whether this expression was solved
                "num_expansions": num_expansions  # Store the number of expansions used
            })

            if solved:
                solved_so_far += 1

            seen_so_far += 1

            logging.info(
                f"{graph_key} - Solved {solved_so_far}/{seen_so_far} ({solved_so_far/seen_so_far:.2f})"
            )

            # Save intermediate results after each expression
            with open(os.path.join(FLAGS.evaluation_dir, 'graph_silhouette_data.pkl'), 'wb') as f:
                pickle.dump(graph_silhouette_data, f)

    # Final save of all results
    with open(os.path.join(FLAGS.evaluation_dir, 'graph_silhouette_data.pkl'), 'wb') as f:
        pickle.dump(graph_silhouette_data, f)

    logging.info("Evaluation complete")
    
# commented out 01/19/2025
    # # Save all target images, best expressions, compiled images, and solved status
    # with open(os.path.join(FLAGS.evaluation_dir, 'all_target_images.pkl'), 'wb') as f:
    #     pickle.dump(all_target_images, f)

    # with open(os.path.join(FLAGS.evaluation_dir, 'all_best_expressions.pkl'), 'wb') as f:
    #     pickle.dump(all_best_expressions, f)

    # with open(os.path.join(FLAGS.evaluation_dir, 'all_compiled_best_expressions.pkl'), 'wb') as f:
    #     pickle.dump(all_compiled_best_expressions, f)

    # with open(os.path.join(FLAGS.evaluation_dir, 'solved_status.pkl'), 'wb') as f:
    #     pickle.dump(solved_status, f)
    
    # with open(os.path.join(FLAGS.evaluation_dir, 'all_num_expansions.pkl'), 'wb') as f:
    #     pickle.dump(all_num_expansions, f)

    # with open(os.path.join(FLAGS.evaluation_dir, 'graph_silhouette_data.pkl'), 'wb') as f:
    #     pickle.dump(graph_silhouette_data, f)


if __name__ == "__main__":
    app.run(main)



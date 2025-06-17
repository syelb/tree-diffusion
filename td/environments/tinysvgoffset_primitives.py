from typing import Tuple

import iceberg as ice
from lark import Transformer, Tree
from lark.visitors import v_args

from td.environments.environment import Environment
from td.environments.goal_checker import GaussianImageGoalChecker
from td.grammar import Compiler, Grammar

_grammar_spec = r"""
// s: arrange | move | pad | compose | rect | ellipse

s: arrange | rect | upper_left_l | lower_right_l

direction: "v" -> v | "h" -> h
color: "black" -> black
//color: "red" -> red | "green" -> green | "blue" -> blue | "yellow" -> yellow | "purple" -> purple | "orange" -> orange | "black" -> black | "white" -> white | "none" -> none
// number: "0" -> zero | "1" -> one | "2" -> two | "3" -> three | "4" -> four | "5" -> five | "6" -> six |  // "7" -> seven | "8" -> eight | "9" -> nine 

number: "0" -> zero
      | "1" -> one
      | "2" -> two
      | "3" -> three
      | "4" -> four
      | "5" -> five
      | "6" -> six
      | "7" -> seven
      | "8" -> eight
      | "9" -> nine
      | "10" -> ten
      | "11" -> eleven
      | "12" -> twelve
      | "13" -> thirteen
      | "14" -> fourteen
      | "15" -> fifteen
      | "16" -> sixteen
      | "17" -> seventeen
      | "18" -> eighteen
      | "19" -> nineteen
      | "20" -> twenty
      | "21" -> twenty_one
      | "22" -> twenty_two
      | "23" -> twenty_three
      | "24" -> twenty_four
      | "25" -> twenty_five
      | "26" -> twenty_six
      | "27" -> twenty_seven
      | "28" -> twenty_eight
      | "29" -> twenty_nine
      | "30" -> thirty
      | "31" -> thirty_one
      | "32" -> thirty_two
      | "33" -> thirty_three
      | "34" -> thirty_four
      | "35" -> thirty_five
      | "36" -> thirty_six
      | "37" -> thirty_seven
      | "38" -> thirty_eight
      | "39" -> thirty_nine
      | "40" -> forty
      | "41" -> forty_one
      | "42" -> forty_two
      | "43" -> forty_three
      | "44" -> forty_four
      | "45" -> forty_five
      | "46" -> forty_six
      | "47" -> forty_seven
      | "48" -> forty_eight
      | "49" -> forty_nine
      | "50" -> fifty
      | "51" -> fifty_one
      | "52" -> fifty_two
      | "53" -> fifty_three
      | "54" -> fifty_four
      | "55" -> fifty_five
      | "56" -> fifty_six
      | "57" -> fifty_seven
      | "58" -> fifty_eight
      | "59" -> fifty_nine
      | "60" -> sixty
      | "61" -> sixty_one
      | "62" -> sixty_two
      | "63" -> sixty_three
      | "64" -> sixty_four
      | "65" -> sixty_five
      | "66" -> sixty_six
      | "67" -> sixty_seven
      | "68" -> sixty_eight
      | "69" -> sixty_nine
      | "70" -> seventy
      | "71" -> seventy_one
      | "72" -> seventy_two
      | "73" -> seventy_three
      | "74" -> seventy_four
      | "75" -> seventy_five
      | "76" -> seventy_six
      | "77" -> seventy_seven
      | "78" -> seventy_eight
      | "79" -> seventy_nine
      | "80" -> eighty
      | "81" -> eighty_one
      | "82" -> eighty_two
      | "83" -> eighty_three
      | "84" -> eighty_four
      | "85" -> eighty_five
      | "86" -> eighty_six
      | "87" -> eighty_seven
      | "88" -> eighty_eight
      | "89" -> eighty_nine
      | "90" -> ninety


sign: "+" -> plus | "-" -> minus
boolean: "true" -> true | "false" -> false

// Rectangle w h fillcolor strokecolor strokewidth ox oy
rect: "(" "Rectangle" " " number " " number " " color " " sign number " " sign number ")"

// UpperLeftL w h fillcolor strokecolor strokewidth ox oy
upper_left_l: "(" "UpperLeftL" " " number " " number " " color " " sign number " " sign number ")"

// LowerRightL w h fillcolor strokecolor strokewidth ox oy
lower_right_l: "(" "LowerRightL" " " number " " number " " color " " sign number " " sign number ")"

// Arrange direction left right gap
arrange: "(" "Arrange" " " direction " " s " " s " " number ")"

// Move x y negx negy
// move: "(" "Move" " " s " " number " " number " " boolean " " boolean ")"

// Pad t r b l
// pad: "(" "Pad" " " s " " number " " number " " number " " number ")"

// Compose without arranging
// compose: "(" "Compose" (" " s)+ ")"

%ignore /[\t\n\f\r]+/ 
"""

_CANVAS_WIDTH = 224
_CANVAS_HEIGHT = 224

_ice_renderer = ice.Renderer(gpu=False)
_ice_canvas = ice.Blank(
ice.Bounds(size=(_CANVAS_WIDTH, _CANVAS_HEIGHT)), ice.Colors.WHITE
)


class _Move(ice.Drawable):
    child: ice.Drawable
    x: float
    y: float

    def setup(self):
        self._child_bounds = self.child.bounds
        self._moved = self.child.move(self.x, self.y)

    @property
    def bounds(self):
        return self._child_bounds

    def draw(self, canvas):
        self._moved.draw(canvas)

    @property
    def children(self):
        return [self._moved]


class TSVGOToIceberg(Transformer):
    def __init__(
        self,
        visit_tokens: bool = True,
        stroke_width_divisor: float = 2.0,
        size_multiplier: float = 1.0,
    ) -> None:
        super().__init__(visit_tokens)
        self._stroke_width_divisor = stroke_width_divisor
        self._size_multiplier = size_multiplier

    @v_args(meta=True)
    def rect(self, meta, children):
        w, h, fill_color, signx, ox, signy, oy = children  # Updated parameters
    
        w = w * self._size_multiplier
        h = h * self._size_multiplier
    
        ox = ox * signx * self._size_multiplier
        oy = oy * signy * self._size_multiplier
    
        # Adjust for center positioning
        ox -= w / 2
        oy -= h / 2
    
        rv = ice.Rectangle(
            ice.Bounds(size=(w, h)),
            fill_color=fill_color,
            border_color=None,
            border_thickness=0,
            anti_alias=False,
            dont_modify_bounds=True,
        )
        rv._lark_meta = meta
    
        rv = _Move(child=rv, x=ox, y=oy)
    
        return rv
    
    @v_args(meta=True)
    def upper_left_l(self, meta, children):
        w, h, fill_color, signx, ox, signy, oy = children  # Updated parameters
        
        w = w * self._size_multiplier
        h = h * self._size_multiplier
        
        ox = ox * signx * self._size_multiplier
        oy = oy * signy * self._size_multiplier
        
        # Create the L-shape using two rectangles with no borders
        horizontal_rect = ice.Rectangle(
            ice.Bounds(size=(w, h/2)),
            fill_color=fill_color,
            border_color=None,
            border_thickness=0,
            anti_alias=False,
            dont_modify_bounds=True,
        )
        
        vertical_rect = ice.Rectangle(
            ice.Bounds(size=(w/2, h/2)),
            fill_color=fill_color,
            border_color=None,
            border_thickness=0,
            anti_alias=False,
            dont_modify_bounds=True,
        )
        
        vertical_rect = _Move(child=vertical_rect, x=0, y=h/2)
        
        l_shape = ice.Compose([horizontal_rect, vertical_rect])
        l_shape._lark_meta = meta
        
        # Apply offset (adjust for center positioning)
        ox -= w/2
        oy -= h/2
        
        rv = _Move(child=l_shape, x=ox, y=oy)
        
        return rv
    
    @v_args(meta=True)
    def lower_right_l(self, meta, children):
        w, h, fill_color, signx, ox, signy, oy = children  # Updated parameters
        
        w = w * self._size_multiplier
        h = h * self._size_multiplier
        
        ox = ox * signx * self._size_multiplier
        oy = oy * signy * self._size_multiplier
        
        horizontal_rect = ice.Rectangle(
            ice.Bounds(size=(w, h/2)),
            fill_color=fill_color,
            border_color=None,
            border_thickness=0,
            anti_alias=False,
            dont_modify_bounds=True,
        )
        
        vertical_rect = ice.Rectangle(
            ice.Bounds(size=(w/2, h/2)),
            fill_color=fill_color,
            border_color=None,
            border_thickness=0,
            anti_alias=False,
            dont_modify_bounds=True,
        )
        
        horizontal_rect = _Move(child=horizontal_rect, x=0, y=h/2)
        vertical_rect = _Move(child=vertical_rect, x=w/2, y=0)
        
        l_shape = ice.Compose([horizontal_rect, vertical_rect])
        l_shape._lark_meta = meta
        
        ox -= w/2
        oy -= h/2
        
        rv = _Move(child=l_shape, x=ox, y=oy)
        
        return rv

    def arrange(self, children):
        direction, left, right, gap = children

        return ice.Arrange(
            [left, right],
            arrange_direction=ice.Arrange.Direction.HORIZONTAL
            if direction == "h"
            else ice.Arrange.Direction.VERTICAL,
            gap=gap,
        )

    def move(self, children):
        drawable, x, y, negx, negy = children

        x = x * self._size_multiplier
        y = y * self._size_multiplier

        x = x if not negx else -x
        y = y if not negy else -y
        return _Move(child=drawable, x=x, y=y)
    
    # edited 10.16.24 - permits combining shapes for one compilation 
    def compose(self, children):
        return ice.Compose(children)  

    def s(self, children):
        # If multiple shapes are parsed, compose them automatically
        if len(children) > 1:
            return ice.Compose(children)  # Combine all shapes
        else:
            return children[0]  # If only one shape, return it directly
    #     return children[0]

    def v(self, _):
        return "v"

    def h(self, _):
        return "h"

    def red(self, _):
        return ice.Colors.RED

    def green(self, _):
        return ice.Colors.GREEN

    def blue(self, _):
        return ice.Colors.BLUE

    def yellow(self, _):
        return ice.Colors.YELLOW

    def purple(self, _):
        return ice.Color.from_hex("#800080")

    def orange(self, _):
        return ice.Color.from_hex("#FFA500")

    def black(self, _):
        return ice.Colors.BLACK

    def white(self, _):
        return ice.Colors.WHITE

    def none(self, _):
        return None

    def zero(self, _):
        return 0
    def one(self, _):
        return 1
    def two(self, _):
        return 2
    def three(self, _):
        return 3
    def four(self, _):
        return 4
    def five(self, _):
        return 5
    def six(self, _):
        return 6
    def seven(self, _):
        return 7
    def eight(self, _):
        return 8
    def nine(self, _):
        return 9
    def ten(self, _):
        return 10
    def eleven(self, _):
        return 11
    def twelve(self, _):
        return 12
    def thirteen(self, _):
        return 13
    def fourteen(self, _):
        return 14
    def fifteen(self, _):
        return 15
    def sixteen(self, _):
        return 16
    def seventeen(self, _):
        return 17
    def eighteen(self, _):
        return 18
    def nineteen(self, _):
        return 19
    def twenty(self, _):
        return 20
    def twenty_one(self, _):
        return 21
    def twenty_two(self, _):
        return 22
    def twenty_three(self, _):
        return 23
    def twenty_four(self, _):
        return 24
    def twenty_five(self, _):
        return 25
    def twenty_six(self, _):
        return 26
    def twenty_seven(self, _):
        return 27
    def twenty_eight(self, _):
        return 28
    def twenty_nine(self, _):
        return 29
    def thirty(self, _):
        return 30
    def thirty_one(self, _):
        return 31
    def thirty_two(self, _):
        return 32
    def thirty_three(self, _):
        return 33
    def thirty_four(self, _):
        return 34
    def thirty_five(self, _):
        return 35
    def thirty_six(self, _):
        return 36
    def thirty_seven(self, _):
        return 37
    def thirty_eight(self, _):
        return 38
    def thirty_nine(self, _):
        return 39
    def forty(self, _):
        return 40
    def forty_one(self, _):
        return 41
    def forty_two(self, _):
        return 42
    def forty_three(self, _):
        return 43
    def forty_four(self, _):
        return 44
    def forty_five(self, _):
        return 45
    def forty_six(self, _):
        return 46
    def forty_seven(self, _):
        return 47
    def forty_eight(self, _):
        return 48
    def forty_nine(self, _):
        return 49
    def fifty(self, _):
        return 50
    def fifty_one(self, _):
        return 51
    def fifty_two(self, _):
        return 52
    def fifty_three(self, _):
        return 53
    def fifty_four(self, _):
        return 54
    def fifty_five(self, _):
        return 55
    def fifty_six(self, _):
        return 56
    def fifty_seven(self, _):
        return 57
    def fifty_eight(self, _):
        return 58
    def fifty_nine(self, _):
        return 59
    def sixty(self, _):
        return 60
    def sixty_one(self, _):
        return 61
    def sixty_two(self, _):
        return 62
    def sixty_three(self, _):
        return 63
    def sixty_four(self, _):
        return 64
    def sixty_five(self, _):
        return 65
    def sixty_six(self, _):
        return 66
    def sixty_seven(self, _):
        return 67
    def sixty_eight(self, _):
        return 68
    def sixty_nine(self, _):
        return 69
    def seventy(self, _):
        return 70
    def seventy_one(self, _):
        return 71
    def seventy_two(self, _):
        return 72
    def seventy_three(self, _):
        return 73
    def seventy_four(self, _):
        return 74
    def seventy_five(self, _):
        return 75
    def seventy_six(self, _):
        return 76
    def seventy_seven(self, _):
        return 77
    def seventy_eight(self, _):
        return 78
    def seventy_nine(self, _):
        return 79
    def eighty(self, _):
        return 80
    def eighty_one(self, _):
        return 81
    def eighty_two(self, _):
        return 82
    def eighty_three(self, _):
        return 83
    def eighty_four(self, _):
        return 84
    def eighty_five(self, _):
        return 85
    def eighty_six(self, _):
        return 86
    def eighty_seven(self, _):
        return 87
    def eighty_eight(self, _):
        return 88
    def eighty_nine(self, _):
        return 89
    def ninety(self, _):
        return 90

    def true(self, _):
        return True

    def false(self, _):
        return False

    def plus(self, _):
        return 1

    def minus(self, _):
        return -1


class TSVGOCompiler(Compiler):
    def __init__(self) -> None:
        super().__init__()
        self._expression_to_iceberg = TSVGOToIceberg()

    def compile(self, expression: Tree):
        drawable = self._expression_to_iceberg.transform(expression)
        scene = ice.Anchor((_ice_canvas, _ice_canvas.add_centered(drawable)))
        # return scene
        _ice_renderer.render(scene)
        rv = _ice_renderer.get_rendered_image()[:, :, :3] / 255.0

        return rv


class TinySVGOffset_primitives(Environment):
    def __init__(self) -> None:
        super().__init__()

        self._grammar = Grammar(
            _grammar_spec,
            start="s",
            primitives=["rect", "upper_left_l", "lower_right_l"],
        )

        self._compiler = TSVGOCompiler()
        self._goal_checker = GaussianImageGoalChecker(self.compiled_shape, sigma=0.1)

    @property
    def grammar(self) -> Grammar:
        return self._grammar

    @property
    def compiler(self) -> Compiler:
        return self._compiler

    @property
    def compiled_shape(self) -> Tuple[int, ...]:
        return _CANVAS_WIDTH, _CANVAS_HEIGHT, 3

    @classmethod
    def name(self) -> str:
        return "tinysvgoffset_primitives"

    def goal_reached(self, compiledA, compiledB) -> bool:
        return self._goal_checker.goal_reached(compiledA, compiledB)

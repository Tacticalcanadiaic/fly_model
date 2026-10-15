"""
cube.py - a Rubik's cube simulator.

How it works: imagine the cube sitting at the center of a 3D grid.
  x points right, y points up, z points toward you (out of the front face).
Every sticker has
  - a position: which little cube ("cubie") it is stuck to, e.g. (1, 1, 1)
    is the top-right-front corner. Each number is -1, 0 or 1.
  - a normal: the direction the sticker faces, e.g. (0, 1, 0) means "up".
To turn a face, we take every sticker in that layer and rotate both its
position and its normal by 90 degrees. Nothing else is needed.

Run this file directly to check the simulator works:  python cube.py
"""
import random
from itertools import product

# The direction each face points, as (x, y, z)
FACES = {
    "U": (0, 1, 0),   # up
    "D": (0, -1, 0),  # down
    "R": (1, 0, 0),   # right
    "L": (-1, 0, 0),  # left
    "F": (0, 0, 1),   # front
    "B": (0, 0, -1),  # back
}

# Starting color of each face (standard scheme). Letters:
# W white, Y yellow, R red, O orange, G green, b blue (lowercase so it
# isn't confused with the B face)
START_COLORS = {"U": "W", "D": "Y", "R": "R", "L": "O", "F": "G", "B": "b"}

# The 12 moves: "R" turns the right face clockwise (as seen looking at it),
# "R'" (say "R prime") turns it counterclockwise.
MOVES = [face + turn for face in "UDRLFB" for turn in ("", "'")]


def dot(a, b):
    """How far vector a reaches in direction b."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    """A vector at right angles to both a and b."""
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def rotate_clockwise(v, axis):
    """Rotate vector v by 90 degrees clockwise, as seen looking down `axis`
    from outside the cube. (Rodrigues' rotation formula with -90 degrees.)"""
    c = cross(axis, v)
    d = dot(axis, v)
    return (-c[0] + axis[0] * d,
            -c[1] + axis[1] * d,
            -c[2] + axis[2] * d)


def inverse(move):
    """The move that undoes `move`:  R <-> R'."""
    return move[0] if move.endswith("'") else move + "'"


class Cube:
    def __init__(self):
        """Make a solved cube."""
        # stickers maps (position, normal) -> color
        self.stickers = {}
        for pos in product((-1, 0, 1), repeat=3):
            for face, normal in FACES.items():
                if dot(pos, normal) == 1:   # this cubie touches this face
                    self.stickers[(pos, normal)] = START_COLORS[face]

    def move(self, name):
        """Apply one move, like "R" or "U'"."""
        axis = FACES[name[0]]
        quarter_turns = 3 if name.endswith("'") else 1   # 3 clockwise = 1 counterclockwise
        for _ in range(quarter_turns):
            turned = {}
            for (pos, normal), color in self.stickers.items():
                if dot(pos, axis) == 1:     # sticker is in the turning layer
                    turned[(rotate_clockwise(pos, axis),
                            rotate_clockwise(normal, axis))] = color
                else:
                    turned[(pos, normal)] = color
            self.stickers = turned

    def apply(self, moves):
        """Apply a list of moves, or a string like "R U R' U'"."""
        if isinstance(moves, str):
            moves = moves.split()
        for m in moves:
            self.move(m)

    def scramble(self, n, rng=random):
        """Apply n random moves and return them. Never picks the move that
        would immediately undo the previous one."""
        moves = []
        for _ in range(n):
            choices = [m for m in MOVES if not moves or m != inverse(moves[-1])]
            moves.append(rng.choice(choices))
        self.apply(moves)
        return moves

    def is_solved(self):
        """True if every face is a single color."""
        for normal in FACES.values():
            colors = {c for (pos, n), c in self.stickers.items() if n == normal}
            if len(colors) != 1:
                return False
        return True

    def state(self):
        """All 54 sticker colors in a fixed order. Later, this list is what
        we'll turn into signals for the fly's eyes."""
        return [self.stickers[key] for key in sorted(self.stickers)]

    def face_grid(self, face):
        """The 3x3 colors of one face, as you'd see it looking straight at it."""
        # For each face: which direction is "right" and which is "down"
        # on the page, when that face is drawn in the unfolded net.
        layout = {
            "U": ((1, 0, 0), (0, 0, 1)),
            "D": ((1, 0, 0), (0, 0, -1)),
            "F": ((1, 0, 0), (0, -1, 0)),
            "B": ((-1, 0, 0), (0, -1, 0)),
            "R": ((0, 0, -1), (0, -1, 0)),
            "L": ((0, 0, 1), (0, -1, 0)),
        }
        normal = FACES[face]
        right, down = layout[face]
        grid = []
        for row in (-1, 0, 1):
            line = []
            for col in (-1, 0, 1):
                pos = tuple(normal[i] + col * right[i] + row * down[i] for i in range(3))
                line.append(self.stickers[(pos, normal)])
            grid.append(line)
        return grid

    def __str__(self):
        """Draw the cube unfolded, like a cardboard box laid flat:
              U
            L F R B
              D
        """
        g = {f: self.face_grid(f) for f in FACES}
        pad = " " * 8
        lines = []
        for r in range(3):
            lines.append(pad + " ".join(g["U"][r]))
        for r in range(3):
            lines.append("  ".join(" ".join(g[f][r]) for f in "LFRB"))
        for r in range(3):
            lines.append(pad + " ".join(g["D"][r]))
        return "\n".join(lines)


# ---------- Self-test: runs only when you do `python cube.py` ----------
if __name__ == "__main__":
    def check(description, result):
        print(("PASS  " if result else "FAIL  ") + description)
        return result

    ok = True
    c = Cube()
    ok &= check("new cube has 54 stickers", len(c.stickers) == 54)
    ok &= check("new cube is solved", c.is_solved())

    c.move("R")
    ok &= check("after R, cube is NOT solved", not c.is_solved())
    up_greens = sum(color == "G" for row in c.face_grid("U") for color in row)
    ok &= check("R moves front (green) stickers onto the top", up_greens == 3)
    c.move("R'")
    ok &= check("R then R' gives a solved cube", c.is_solved())

    c = Cube()
    c.apply("R R R R")
    ok &= check("R four times gives a solved cube", c.is_solved())

    c = Cube()
    c.apply("R U R' U'" + " R U R' U'" * 5)
    ok &= check("(R U R' U') six times gives a solved cube", c.is_solved())

    c = Cube()
    c.apply("R U R' U'")
    ok &= check("(R U R' U') once does NOT solve it", not c.is_solved())

    rng = random.Random(0)
    for trial in range(100):
        c = Cube()
        moves = c.scramble(20, rng)
        counts = {col: c.state().count(col) for col in START_COLORS.values()}
        if any(v != 9 for v in counts.values()):
            break
        c.apply([inverse(m) for m in reversed(moves)])
        if not c.is_solved():
            break
    else:
        trial = None
    ok &= check("100 random scrambles: 9 of each color, and undoing solves", trial is None)

    print()
    c = Cube()
    print("After the move R:")
    c.move("R")
    print(c)
    print()
    print("ALL TESTS PASSED" if ok else "SOME TESTS FAILED")

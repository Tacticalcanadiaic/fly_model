"""
beam.py - solve cubes with beam search, guided by the fly brain.

solve.py lets the fly pick one move at a time and never looks back. That is
"greedy" search. Here the fly's move scores guide a beam search instead:

  1. Start with the scrambled cube.
  2. Try all 12 moves on every cube we're keeping.
  3. Score each new cube by how confident the fly was in the moves that led
     there, and keep only the best WIDTH of them. That's the "beam".
  4. Repeat until some cube is solved, or we run out of moves.

Width 1 is exactly what solve.py does. Wider beams can recover from a bad
move, because the second- and third-best options are still around.

All of this happens in the simulator. The search checks for "solved" itself,
the same way DeepCubeA does. The fly's only job is to say which moves look
promising.

Three guides are compared on the same scrambles:
  - fly brain:       readout of the real wiring
  - shuffled brain:  same neurons and connections, rewired at random
  - random:          no brain, each move gets a random score
A wide enough beam can stumble on short solutions with no help at all. So the
question is whether the fly solves more than random guidance at the same width.

Run (from ~/fly_model, next to cube.py, encode.py, experiment.py, solve.py):
  uv run python beam.py
"""
import math
import random
import time

import numpy as np
from flybrain import FlyBrain

from cube import Cube, MOVES
from encode import CubeEncoder
from experiment import states_up_to, good_moves
from solve import FlySolver, DEPTH, TRIALS

# ---------- Settings ----------
WIDTHS = (1, 10, 100)      # beam widths to compare (1 = greedy, like solve.py)
LENGTHS = (1, 2, 3, 4, 5)  # scramble lengths to test; training covers up to DEPTH
SCRAMBLES = 20             # test scrambles per length (the same ones for every guide)
EXTRA_MOVES = 3            # search this many moves past the scramble length
TEMPERATURE = 0.1          # low = trust the fly's top pick more; high = all moves alike


# ---------- Small helpers ----------
def key(cube):
    """The cube as one string, so two cubes can be compared quickly."""
    return "".join(cube.state())


def copy_cube(cube):
    """An independent copy, so trying a move doesn't change the original."""
    new = Cube.__new__(Cube)          # an empty Cube, without setting it up
    new.stickers = dict(cube.stickers)
    return new


def inverse(m):
    """The move that undoes m: R <-> R'."""
    return m[:-1] if m.endswith("'") else m + "'"


def to_probs(scores, temperature):
    """Softmax: turn any 12 numbers into positive confidences adding up to 1."""
    s = np.asarray(scores, dtype=np.float64) / temperature
    s = s - s.max()                   # keeps exp() from overflowing; same answer
    e = np.exp(s)
    return e / e.sum()


# ---------- Guides: who says which moves look promising ----------
class FlyGuide:
    """Asks a trained FlySolver how good each move looks from a cube.

    The brain is frozen, so we remember each cube's answer and reuse it if
    the search meets that cube again. That saves a lot of time; it also means
    each cube gets one noisy look for the whole run."""

    def __init__(self, solver, seed):
        self.solver = solver
        self.seed = seed
        self.memory = {}

    @property
    def looks(self):
        return len(self.memory)

    def probs(self, cube):
        k = key(cube)
        if k not in self.memory:
            X = self.solver.look(cube, seed=self.seed + len(self.memory))
            scores = self.solver.scores(X[None])[0]    # [None] = a batch of one
            self.memory[k] = to_probs(scores, TEMPERATURE)
        return self.memory[k]


class RandomGuide:
    """No brain: every move gets a random score. Same memory trick, so a
    cube always gets the same random opinion."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.memory = {}
        self.looks = 0

    def probs(self, cube):
        k = key(cube)
        if k not in self.memory:
            self.memory[k] = to_probs(self.rng.normal(0, 1, len(MOVES)), 1.0)
        return self.memory[k]


# ---------- The search ----------
def beam_search(start, guide, width, max_depth):
    """Returns (solution, cubes_asked). solution is a list of moves, or None
    if nothing was found. cubes_asked counts how many cubes the guide had to
    judge, which is the search's cost."""
    if start.is_solved():
        return [], 0

    beam = [(0.0, start, [])]          # each entry: (score, cube, moves so far)
    seen = {key(start)}                # cubes already reached, so we skip repeats
    asked = 0

    for depth in range(max_depth):
        candidates = []
        for score, cube, path in beam:
            p = guide.probs(cube)
            asked += 1
            for i, move in enumerate(MOVES):
                if path and move == inverse(path[-1]):
                    continue           # would just undo the last move
                new = copy_cube(cube)
                new.move(move)
                if new.is_solved():
                    return path + [move], asked
                k = key(new)
                if k in seen:
                    continue           # reached this cube another way already
                seen.add(k)
                # Adding logs = multiplying confidences along the path.
                # The tiny 1e-12 avoids log(0) when a confidence rounds to 0.
                candidates.append((score + math.log(p[i] + 1e-12), new, path + [move]))
        if not candidates:
            break
        candidates.sort(key=lambda c: c[0], reverse=True)   # best scores first
        beam = candidates[:width]                           # keep the top `width`
    return None, asked


# ---------- Testing ----------
def run_tests(guide, scrambles, name):
    """Every width on every scramble. Returns {(width, length): stats}."""
    results = {}
    for width in WIDTHS:
        start = time.time()
        for length, move_lists in scrambles.items():
            wins = solution_moves = asked_total = 0
            for moves in move_lists:
                cube = Cube()
                cube.apply(moves)
                solution, asked = beam_search(cube, guide, width, length + EXTRA_MOVES)
                asked_total += asked
                if solution is not None:
                    check = Cube()                     # replay it to be sure
                    check.apply(moves + solution)
                    assert check.is_solved(), "search returned a wrong solution"
                    wins += 1
                    solution_moves += len(solution)
            results[width, length] = (wins, solution_moves, asked_total)
        print(f"  {name}, width {width}: {time.time() - start:.0f} s "
              f"({guide.looks:,} fly looks so far)")
    return results


def train_solver(brain, encoder, states, samples, label):
    solver = FlySolver(brain, encoder)
    print(f"\n{label}: pilot run")
    solver.choose_population(states)
    print(f"{label}: training on {len(samples):,} trials")
    solver.train(samples)
    return solver


def print_tables(all_results):
    cols = "".join(f"{str(n) + ('*' if n > DEPTH else ''):>8}" for n in LENGTHS)
    rows = [(name, w) for name in all_results for w in WIDTHS]

    def table(title, cell):
        print(f"\n{title}")
        print(f"{'scramble length':26}{cols}")
        for name, w in rows:
            label = f"{name}, width {w}"
            line = "".join(f"{cell(all_results[name][w, n]):>8}" for n in LENGTHS)
            print(f"{label:26}{line}")

    table(f"Solved, out of {SCRAMBLES} (* = longer than the fly trained on)",
          lambda r: r[0])
    table("Average solution length in moves (- = none solved)",
          lambda r: f"{r[1] / r[0]:.1f}" if r[0] else "-")
    table("Average cubes the guide judged per search (the cost)",
          lambda r: f"{r[2] / SCRAMBLES:.0f}")


if __name__ == "__main__":
    t0 = time.time()

    # The same scrambles for every guide, so the comparison is fair
    rng = random.Random(42)
    scrambles = {}
    for n in LENGTHS:
        scrambles[n] = []
        for _ in range(SCRAMBLES):
            scrambles[n].append(Cube().scramble(n, rng))

    # Training data, exactly as in solve.py
    table = states_up_to(DEPTH)
    states = [v[1] for v in table.values() if v[0] >= 1]
    print(f"{len(states):,} cube states within {DEPTH} moves of solved")
    labeled = [(moves, good_moves(moves, table)) for moves in states]
    samples = [s for s in labeled for _ in range(TRIALS)]

    all_results = {}

    brain = FlyBrain(device="cpu")
    encoder = CubeEncoder(brain)
    solver = train_solver(brain, encoder, states, samples, "Fly brain")
    print("\nFly brain guiding the search")
    fly_guide = FlyGuide(solver, seed=3_000_000)
    all_results["fly brain"] = run_tests(fly_guide, scrambles, "fly brain")

    # One example, before the brain gets rewired below
    example = scrambles[max(LENGTHS)][0]
    cube = Cube()
    cube.apply(example)
    solution, asked = beam_search(cube, fly_guide, max(WIDTHS), len(example) + EXTRA_MOVES)
    print(f"\nExample: scramble {' '.join(example)}")
    if solution is None:
        print(f"  not solved at width {max(WIDTHS)} ({asked} cubes judged)")
    else:
        print(f"  fly-guided solution: {' '.join(solution)} ({asked} cubes judged)")

    # Rewire at random: every connection keeps its source and strength,
    # gets a random target. The readout must be retrained for the new brain.
    brain.indices = np.random.default_rng(7).permutation(brain.indices)
    solver = train_solver(brain, encoder, states, samples, "Shuffled brain")
    print("\nShuffled brain guiding the search")
    all_results["shuffled brain"] = run_tests(FlyGuide(solver, seed=4_000_000),
                                              scrambles, "shuffled brain")

    print("\nRandom guide (no brain)")
    all_results["random"] = run_tests(RandomGuide(seed=1), scrambles, "random")

    print_tables(all_results)
    print(f"\nDone in {time.time() - t0:.0f} s")

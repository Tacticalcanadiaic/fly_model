"""
see_colors.py - can the fly tell what colors are on a scrambled cube?

We show the fly a scrambled cube, record what its brain does, and train a
readout to say the color of every one of the 54 stickers. Then we hand it a
scramble it has never seen and draw the cube the way the fly "saw" it.

As before, the readout never sees the neurons we stimulate directly. It has
to recover the colors from what the wiring did with the signal.

Two brains, same test:
  - fly brain:       the real wiring
  - shuffled brain:  same neurons and connections, rewired at random

Run:
  uv run python see_colors.py
  uv run python see_colors.py "Scramble (F R2 U')"
  uv run python see_colors.py --plain      (no colored squares)
"""
import random
import re
import sys
import time

import numpy as np
from flybrain import FlyBrain, fit_ridge

from cube import Cube
from encode import COLORS, N_STICKERS, CubeEncoder

# ---------- Settings ----------
STEPS = 10              # 20 ms steps per look: 0.2 s of fly time
TRAIN_CUBES = 2000      # random scrambles the readout learns from
TEST_CUBES = 200        # random scrambles it never sees in training
YOUR_LOOKS = 5          # noisy looks at your scramble
SCRAMBLE_LENGTH = 20
COMPONENTS = 400        # the readout compresses activity to this many numbers
LAMBDAS = (0.1, 1.0, 10.0, 100.0)   # simplicity settings to try
MIN_ACTIVITY = 0.02     # ignore neurons that fire in under 2% of looks

DEFAULT_SCRAMBLE = "Scramble (R U2 F' L D2 B R' U L2 F2 D' R2 B2 U' L F' D2 R U' B2)"


# ---------- Scrambles ----------
def parse_scramble(text):
    """Turn "Scramble (F R2 U')" (or just "F R2 U'") into moves cube.py
    understands. R2 means R twice, so it becomes ["R", "R"]."""
    inside = re.search(r"\((.*)\)", text)
    tokens = (inside.group(1) if inside else text).split()
    moves = []
    for t in tokens:
        if not re.fullmatch(r"[UDRLFB]['2]?", t):
            raise ValueError(f"don't understand the move {t!r}")
        moves += [t[0], t[0]] if t.endswith("2") else [t]
    return moves


def random_scramble(rng, n=SCRAMBLE_LENGTH):
    """A scramble written like "Scramble (F R2 U')". Never turns the same
    face twice in a row, so no moves cancel out."""
    moves = []
    for _ in range(n):
        face = rng.choice([f for f in "UDRLFB" if not moves or f != moves[-1][0]])
        moves.append(face + rng.choice(["", "'", "2"]))
    return "Scramble (" + " ".join(moves) + ")"


def cube_from(scramble):
    c = Cube()
    c.apply(parse_scramble(scramble))
    return c


def cube_with_colors(colors):
    """Build a cube whose 54 stickers are `colors`, listed in state() order.
    Used to draw what the fly read."""
    c = Cube()
    c.stickers = dict(zip(sorted(c.stickers), colors))
    return c


# ---------- Running the brain ----------
def record(brain, encoder, cubes, label):
    """Show the brain each cube once, each with its own noise.
    Returns spike counts: one row per look, one column per neuron."""
    X = np.zeros((len(cubes), brain.n), np.uint8)
    start = time.time()
    for i, c in enumerate(cubes):
        inject = encoder.inject(c)
        brain.reset(1000 + i)
        for _ in range(STEPS):
            X[i, brain.step(inject=inject)] += 1
        if (i + 1) % 250 == 0 or i + 1 == len(cubes):
            print(f"  {label}: {i + 1}/{len(cubes)} looks ({time.time() - start:.0f} s)")
    return X


def readable_neurons(X_train, encoder):
    """Neurons the readout may watch: not stimulated by us, and active enough."""
    keep = np.count_nonzero(X_train, axis=0) >= MIN_ACTIVITY * len(X_train)
    keep[np.concatenate(encoder.groups)] = False
    return keep


# ---------- The readout ----------
def answers(cubes):
    """What the readout should say: for each sticker, a 1 for its color
    and 0 for the other five. 54 x 6 = 324 numbers per cube."""
    Y = np.zeros((len(cubes), N_STICKERS * len(COLORS)), np.float32)
    for i, c in enumerate(cubes):
        for s, color in enumerate(c.state()):
            Y[i, s * len(COLORS) + COLORS.index(color)] = 1.0
    return Y


class ColorReader:
    """Compress activity to its main patterns (PCA), then one weighted sum
    per (sticker, color). Each sticker's guess is its highest-scoring color."""

    def __init__(self, X, Y, lam):
        self.mu = X.mean(0)
        _, S, Vt = np.linalg.svd(X - self.mu, full_matrices=False)
        keep = min(COMPONENTS, int((S > S[0] * 1e-6).sum()))
        self.P = Vt[:keep].T
        Z = (X - self.mu) @ self.P
        self.sd = Z.std(0) + 1e-6
        self.W, self.b = fit_ridge(Z / self.sd, Y, lam)

    def scores(self, X):
        Z = ((X - self.mu) @ self.P) / self.sd
        return (Z @ self.W.T + self.b).reshape(len(X), N_STICKERS, len(COLORS))

    def read(self, X):
        """The color index the readout picks for every sticker of every look."""
        return self.scores(X).argmax(2)


def color_numbers(cubes):
    return np.array([[COLORS.index(col) for col in c.state()] for c in cubes])


def train_reader(X, cubes, movable):
    """Pick the simplicity setting on a held-back slice of the training
    cubes, then refit on all of them."""
    n_val = len(cubes) // 6
    truth = color_numbers(cubes[-n_val:])[:, movable]
    best, best_acc = None, -1.0
    for lam in LAMBDAS:
        r = ColorReader(X[:-n_val], answers(cubes[:-n_val]), lam)
        acc = (r.read(X[-n_val:])[:, movable] == truth).mean()
        print(f"    simplicity {lam:g}: {acc:.1%} of held-back stickers right")
        if acc > best_acc:
            best, best_acc = lam, acc
    print(f"    using {best:g}")
    return ColorReader(X, answers(cubes), best)


# ---------- Drawing ----------
ANSI = {"W": "107;30", "Y": "103;30", "R": "41;97", "O": "48;5;208;30",
        "G": "42;30", "b": "44;97"}


def draw(cube, wrong=None, use_color=True):
    """The unfolded cube (U on top, then L F R B, then D). Stickers in
    `wrong` (a cube of True/False marks) get a * after their letter."""
    def cell(face, r, k):
        letter = cube.face_grid(face)[r][k]
        mark = "*" if wrong is not None and wrong.face_grid(face)[r][k] else " "
        text = f"{letter}{mark}"
        return f"\033[{ANSI[letter]}m {text}\033[0m" if use_color else f" {text}"

    def row(face, r):
        return "".join(cell(face, r, k) for k in range(3))

    pad = " " * 10
    lines = [pad + row("U", r) for r in range(3)]
    lines += [" ".join(row(f, r) for f in "LFRB") for r in range(3)]
    lines += [pad + row("D", r) for r in range(3)]
    return "\n".join(lines)


# ---------- Main ----------
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--plain"]
    use_color = "--plain" not in sys.argv and sys.stdout.isatty()
    your_scramble = args[0] if args else DEFAULT_SCRAMBLE
    your_cube = cube_from(your_scramble)          # fails early on a typo

    t0 = time.time()
    rng = random.Random(0)
    train = [cube_from(random_scramble(rng)) for _ in range(TRAIN_CUBES)]
    test = [cube_from(random_scramble(rng)) for _ in range(TEST_CUBES)]
    looks = train + test + [your_cube] * YOUR_LOOKS
    n_tr, n_te = len(train), len(test)

    # Centers never move with face turns, so they're free points. Score only
    # the 48 stickers that actually change.
    movable = color_numbers(train).std(0) > 0
    truth_test = color_numbers(test)[:, movable]
    truth_yours = color_numbers([your_cube])[0]

    brain = FlyBrain(device="cpu")
    encoder = CubeEncoder(brain)
    results = {}

    for name in ("fly brain", "shuffled brain"):
        print(f"\n{name.capitalize()}")
        if name == "shuffled brain":
            # Every connection keeps its source and strength, gets a random target
            brain.indices = np.random.default_rng(7).permutation(brain.indices)
        X = record(brain, encoder, looks, name.split()[0])
        keep = readable_neurons(X[:n_tr], encoder)
        X = X[:, keep].astype(np.float32)
        print(f"  readout watches {keep.sum():,} neurons")
        reader = train_reader(X[:n_tr], train, movable)

        guess_test = reader.read(X[n_tr:n_tr + n_te])[:, movable]
        # Your cube: add up the scores from all looks, then pick (like squinting
        # at it several times and going with the overall impression)
        your_scores = reader.scores(X[n_tr + n_te:])
        guess_yours = your_scores.sum(0).argmax(1)
        single = (your_scores.argmax(2) == truth_yours)[:, movable].mean()
        results[name] = {
            "test": (guess_test == truth_test).mean(),
            "whole": (guess_test == truth_test).all(1).mean(),
            "single": single,
            "yours": (guess_yours == truth_yours)[movable].mean(),
            "guess": guess_yours,
        }
        del X

    print("\nResults (share of the 48 moving stickers read correctly)")
    print(f"{'':18}{'new cubes':>11}{'whole cube':>12}{'yours, 1 look':>15}"
          f"{f'yours, {YOUR_LOOKS} looks':>16}")
    for name, r in results.items():
        print(f"{name:18}{r['test']:>11.1%}{r['whole']:>12.1%}"
              f"{r['single']:>15.1%}{r['yours']:>16.1%}")
    print(f"{'chance':18}{1 / len(COLORS):>11.1%}")

    fly_guess = results["fly brain"]["guess"]
    read_cube = cube_with_colors([COLORS[i] for i in fly_guess])
    wrong = cube_with_colors(list(fly_guess != truth_yours))

    print(f"\n{your_scramble}")
    print("\nThe actual cube:")
    print(draw(your_cube, use_color=use_color))
    n_wrong = int((fly_guess != truth_yours).sum())
    print(f"\nWhat the fly brain read ({n_wrong} wrong, marked *):")
    print(draw(read_cube, wrong, use_color=use_color))
    print(f"\nDone in {time.time() - t0:.0f} s")

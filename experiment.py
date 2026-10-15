"""
experiment.py - can a readout of the fly brain pick the move that solves a cube?

Two tests:
  1. Recognize:  cube is 1 move from solved (12 states). Train and test on the
                 same states, but test trials have fresh noise.
  2. Generalize: cube is 2 moves from solved (114 states). A quarter of the
                 states are hidden during training; we test only on those.
                 A guess counts as right if it brings the cube closer to solved.

Three contestants, all with the same simple readout:
  - no brain:        the readout sees the 324 sticker facts directly
  - fly brain:       the readout sees spike counts from the real wiring
  - shuffled brain:  same neurons and connections, rewired at random

The readout never sees the neurons we stimulate, only what the wiring
did with the signal.

Run:  python experiment.py
"""
import time

import numpy as np
from flybrain import FlyBrain, fit_ridge

from cube import Cube, MOVES
from encode import CubeEncoder

# ---------- Settings ----------
STEPS = 10                # 20 ms steps per trial: 0.2 s of fly time
TRIALS_RECOGNIZE = 20     # noisy trials per state, test 1
TRIALS_GENERALIZE = 10    # noisy trials per state, test 2
TEST_FRACTION = 0.25      # share of trials (test 1) or states (test 2) held out
COMPONENTS = 60           # the readout compresses activity to this many numbers
LAMBDA = 1.0              # how strongly the readout is kept simple
MIN_ACTIVITY = 0.02       # ignore neurons that fire in under 2% of trials


# ---------- Cube bookkeeping ----------
def states_up_to(depth):
    """Every cube state reachable in `depth` moves or fewer, found by
    breadth-first search. Returns {state: (distance, moves_to_reach_it)}."""
    table = {tuple(Cube().state()): (0, [])}
    frontier = [[]]
    for d in range(1, depth + 1):
        next_frontier = []
        for moves in frontier:
            for m in MOVES:
                c = Cube()
                c.apply(moves + [m])
                key = tuple(c.state())
                if key not in table:          # first time we reach it = shortest path
                    table[key] = (d, moves + [m])
                    next_frontier.append(moves + [m])
        frontier = next_frontier
    return table


def good_moves(moves, table):
    """The moves that bring this cube one step closer to solved."""
    c = Cube()
    c.apply(moves)
    distance = table[tuple(c.state())][0]
    good = []
    for m in MOVES:
        after = Cube()
        after.apply(moves + [m])
        entry = table.get(tuple(after.state()))
        if entry is not None and entry[0] < distance:
            good.append(m)
    return good


def make_samples(seed=0):
    """The list of trials to run: which cube, which answers are right,
    which test it belongs to, and whether it's for training or testing."""
    table = states_up_to(2)
    rng = np.random.default_rng(seed)
    samples = []

    depth1 = sorted(v[1] for v in table.values() if v[0] == 1)
    n_train = round(TRIALS_RECOGNIZE * (1 - TEST_FRACTION))
    for moves in depth1:
        good = good_moves(moves, table)
        for t in range(TRIALS_RECOGNIZE):
            samples.append({"moves": moves, "good": good, "task": "recognize",
                            "split": "train" if t < n_train else "test"})

    depth2 = sorted(v[1] for v in table.values() if v[0] == 2)
    order = rng.permutation(len(depth2))
    n_test = round(len(depth2) * TEST_FRACTION)
    hidden = set(order[:n_test].tolist())
    for i, moves in enumerate(depth2):
        good = good_moves(moves, table)
        for t in range(TRIALS_GENERALIZE):
            samples.append({"moves": moves, "good": good, "task": "generalize",
                            "split": "test" if i in hidden else "train"})

    for i, s in enumerate(samples):
        s["seed"] = 1000 + i          # each trial gets its own noise
    print(f"{len(depth1)} one-move states, {len(depth2)} two-move states "
          f"({n_test} hidden for testing), {len(samples)} trials in total")
    return samples


# ---------- Running the brain ----------
def record(brain, encoder, samples, label):
    """Show the brain every trial's cube. Returns spike counts,
    one row per trial and one column per neuron."""
    X = np.zeros((len(samples), brain.n), np.uint8)
    start = time.time()
    for i, s in enumerate(samples):
        c = Cube()
        c.apply(s["moves"])
        inject = encoder.inject(c)
        brain.reset(s["seed"])
        for _ in range(STEPS):
            X[i, brain.step(inject=inject)] += 1
        if (i + 1) % 200 == 0 or i + 1 == len(samples):
            print(f"  {label}: {i + 1}/{len(samples)} trials ({time.time() - start:.0f} s)")
    return X


def keep_readable(X, encoder):
    """Drop the neurons we stimulated directly, and neurons that hardly fire."""
    readable = np.count_nonzero(X, axis=0) >= MIN_ACTIVITY * len(X)
    readable[np.concatenate(encoder.groups)] = False
    print(f"  readout watches {readable.sum():,} neurons; "
          f"on average {np.count_nonzero(X, axis=1).mean():,.0f} fire per trial")
    return X[:, readable].astype(np.float32)


def sticker_facts(samples, encoder):
    """The 'no brain' input: 324 yes/no sticker facts per trial."""
    X = np.zeros((len(samples), len(encoder.groups)), np.float32)
    for i, s in enumerate(samples):
        c = Cube()
        c.apply(s["moves"])
        X[i, encoder.feature_numbers(c)] = 1.0
    return X


# ---------- The readout ----------
def score(X, samples, task):
    """Train a readout on this task's training trials, test it on the
    test trials. Returns (accuracy, chance level)."""
    train = [i for i, s in enumerate(samples) if s["task"] == task and s["split"] == "train"]
    test = [i for i, s in enumerate(samples) if s["task"] == task and s["split"] == "test"]

    # Target: share the credit equally among all the right answers
    Y = np.zeros((len(train), len(MOVES)), np.float32)
    for row, i in enumerate(train):
        for m in samples[i]["good"]:
            Y[row, MOVES.index(m)] = 1.0 / len(samples[i]["good"])

    # Compress to the main patterns of variation (PCA), then fit a linear map
    Xtr, Xte = X[train], X[test]
    mu = Xtr.mean(0)
    _, S, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
    keep = min(COMPONENTS, int((S > S[0] * 1e-6).sum()))
    P = Vt[:keep].T
    Ztr = (Xtr - mu) @ P
    sd = Ztr.std(0)
    Ztr, Zte = Ztr / sd, ((Xte - mu) @ P) / sd
    W, b = fit_ridge(Ztr, Y, LAMBDA)
    guesses = (Zte @ W.T + b).argmax(1)

    correct = [MOVES[g] in samples[i]["good"] for g, i in zip(guesses, test)]
    chance = np.mean([len(samples[i]["good"]) / len(MOVES) for i in test])
    return float(np.mean(correct)), float(chance)


# ---------- Main ----------
if __name__ == "__main__":
    t0 = time.time()
    samples = make_samples()
    brain = FlyBrain(device="cpu")
    encoder = CubeEncoder(brain)
    results = {}

    print("\nNo brain")
    results["no brain (stickers)"] = sticker_facts(samples, encoder)

    print("\nFly brain")
    X = record(brain, encoder, samples, "fly")
    results["fly brain"] = keep_readable(X, encoder)
    del X                                   # free the memory

    print("\nShuffled brain")
    # Every connection keeps its source neuron and strength but gets a random
    # target. Each neuron keeps its number of inputs and outputs.
    brain.indices = np.random.default_rng(7).permutation(brain.indices)
    X = record(brain, encoder, samples, "shuffled")
    results["shuffled brain"] = keep_readable(X, encoder)
    del X

    print("\nResults (share of test cubes where the guessed move was right)")
    print(f"{'':24}{'recognize':>12}{'generalize':>12}")
    chances = {}
    for name, feats in results.items():
        row = []
        for task in ("recognize", "generalize"):
            acc, chances[task] = score(feats, samples, task)
            row.append(acc)
        print(f"{name:24}{row[0]:>12.1%}{row[1]:>12.1%}")
    print(f"{'chance (random guess)':24}{chances['recognize']:>12.1%}{chances['generalize']:>12.1%}")
    print(f"\nDone in {time.time() - t0:.0f} s")

"""
baseline.py - the fair comparison for rl.py.

Everything matches rl.py: the same 1,314 descending neurons, the same
0.4 s looks, the same cube states (up to 3 moves from solved), the same
kind of readout (one weighted sum per move), and the same test.
The only difference: here the readout is TOLD the right moves.

It is trained twice:
  - on 2 noisy looks per state (exactly the data rl.py learned from)
  - on 5 noisy looks per state (does seeing more noise help on fresh looks?)

Run:  python baseline.py
"""
import random
import time

import numpy as np
from flybrain import FlyBrain, fit_ridge

from cube import Cube, MOVES
from encode import CubeEncoder
from rl import (MAX_DEPTH, TABLE_DEPTH, TEST_SCRAMBLES, FlyEyes, copy_cube,
                distance_table, key, play)

LOOKS = 5          # looks recorded per state
LAMBDAS = (0.01, 0.1, 1.0, 10.0)   # regularization strengths to try


class TaughtUnits:
    """Same shape as rl.py's MoveUnits: one weighted sum per move over
    normalized descending-neuron activity. Fitted from the right answers
    in one go (ridge regression) instead of learned from reward."""

    def __init__(self, X, Y, lam):
        self.mean = X.mean(0)
        self.sd = X.std(0) + 1e-3
        self.W, self.b = fit_ridge(self.normalize(X), Y, lam)

    def normalize(self, counts, update=False):
        return (counts.astype(np.float64) - self.mean) / self.sd

    def choose(self, z, explore=False):
        scores = z @ self.W.T + self.b
        return int(np.argmax(scores)), None

    def accuracy(self, X, Y):
        """Share of looks where the top-scoring move is a good one."""
        guesses = (self.normalize(X) @ self.W.T + self.b).argmax(1)
        return float(np.mean(Y[np.arange(len(Y)), guesses] > 0))


def fit_fairly(X, Y, looks):
    """Try each regularization strength, judged on the last recorded look
    (fresh noise, like the test). Then refit on `looks` looks with the best.
    The judging look is never trained on while choosing."""
    n = X.shape[2]
    tune_looks = min(looks, LOOKS - 1)          # keep the last look aside
    tune_X = X[:tune_looks].reshape(-1, n)
    tune_Y = np.tile(Y, (tune_looks, 1))
    scores = {lam: TaughtUnits(tune_X, tune_Y, lam).accuracy(X[LOOKS - 1], Y)
              for lam in LAMBDAS}
    best = max(scores, key=scores.get)
    print("  held-out accuracy by regularization: " +
          ", ".join(f"{lam:g}: {acc:.1%}" for lam, acc in scores.items()) + f"  -> using {best:g}")
    return TaughtUnits(X[:looks].reshape(-1, n), np.tile(Y, (looks, 1)), best)


if __name__ == "__main__":
    t0 = time.time()
    print("Building the referee's distance table...")
    dist, puzzles = distance_table(TABLE_DEPTH, keep_up_to=MAX_DEPTH)

    brain = FlyBrain(device="cpu")
    encoder = CubeEncoder(brain)
    eyes = FlyEyes(brain, encoder, random.Random(0))

    # Every state up to MAX_DEPTH, with the moves that bring it closer to solved
    states = [c for d in range(1, MAX_DEPTH + 1) for c in puzzles[d]]
    targets = []
    for c in states:
        d = dist[key(c)]
        good = []
        for i, m in enumerate(MOVES):
            after = copy_cube(c)
            after.move(m)
            if dist.get(key(after), 99) < d:
                good.append(i)
        y = np.zeros(len(MOVES))
        y[good] = 1.0 / len(good)
        targets.append(y)

    print(f"\nRecording {LOOKS} looks of each of {len(states):,} states "
          f"through {len(eyes.dn):,} descending neurons")
    X = np.zeros((LOOKS, len(states), len(eyes.dn)), np.float32)
    for i, c in enumerate(states):
        for j in range(LOOKS):
            X[j, i] = eyes.record(c)
        if (i + 1) % 200 == 0 or i + 1 == len(states):
            print(f"  {i + 1:,}/{len(states):,} states ({time.time() - t0:.0f} s)")
    Y = np.array(targets)

    for looks in (2, LOOKS):
        print(f"\nTold the answers, {looks} looks per state")
        units = fit_fairly(X, Y, looks)
        print("  Testing with fresh noise on every look:")
        test_rng = random.Random(42)          # the exact same scrambles as rl.py's test
        for length in range(1, MAX_DEPTH + 2):
            wins = 0
            for _ in range(TEST_SCRAMBLES):
                c = Cube()
                c.scramble(length, test_rng)
                if key(c) not in dist:
                    continue
                ok, _ = play(c, length + 3, eyes, units, dist,
                             explore=False, learn=False, fresh=True)
                wins += ok
            note = "  <- beyond training" if length > MAX_DEPTH else ""
            print(f"  {length}-move scrambles: solved {wins}/{TEST_SCRAMBLES}{note}")
    print(f"\nDone in {time.time() - t0:.0f} s")

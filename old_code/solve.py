"""
solve.py - let the fly brain solve scrambled cubes, one move at a time.

1. Training: show the fly every cube state within DEPTH moves of solved,
   several times each, and fit a readout from its brain activity to the
   move(s) that bring that cube closer to solved.
2. Solving: show the fly a scrambled cube, let the readout pick a move from
   the brain's activity, apply it, show the fly the new cube, repeat.
   Every move comes from the fly. Nothing else picks or corrects moves.

The readout only learns states it was shown, so expect success up to DEPTH
and mostly failure beyond it. We test one depth further on purpose.

Run:  python solve.py
"""
import random
import time

import numpy as np
from flybrain import FlyBrain, fit_ridge

from cube import Cube, MOVES
from encode import CubeEncoder
from experiment import states_up_to, good_moves

# ---------- Settings ----------
DEPTH = 3            # train on every state up to this many moves from solved
TRIALS = 3           # noisy trials per state during training
STEPS = 10           # 20 ms steps per look at the cube: 0.2 s of fly time
COMPONENTS = 300     # readout compresses activity to this many numbers
LAMBDA = 0.01        # small = readout may fit the training states closely
SCRAMBLES = 50       # test scrambles per length
EXTRA_MOVES = 3      # moves allowed beyond the scramble length before giving up


class FlySolver:
    def __init__(self, brain, encoder):
        self.brain, self.encoder = brain, encoder
        self.population = None   # which neurons the readout watches
        self.slot = None

    def look(self, cube, seed):
        """Show the fly this cube for STEPS steps; return spike counts of
        the watched neurons (or of every neuron, before we've chosen)."""
        self.brain.reset(seed)
        inject = self.encoder.inject(cube)
        width = self.brain.n if self.population is None else len(self.population)
        counts = np.zeros(width, np.float32)
        for _ in range(STEPS):
            fired = self.brain.step(inject=inject)
            if self.population is None:
                counts[fired] += 1
            else:
                s = self.slot[fired]
                counts[s[s >= 0]] += 1
        return counts

    def choose_population(self, move_lists, seed=0):
        """Pilot run: watch neurons that fire reasonably often, excluding
        the ones we stimulate directly (so the readout can't just read our
        own input back)."""
        rng = random.Random(seed)
        pilot = rng.sample(move_lists, min(300, len(move_lists)))
        active = np.zeros(self.brain.n, np.int32)
        for i, moves in enumerate(pilot):
            c = Cube()
            c.apply(moves)
            active += self.look(c, seed=500_000 + i) > 0
        keep = active >= 0.02 * len(pilot)
        keep[np.concatenate(self.encoder.groups)] = False
        self.population = np.flatnonzero(keep)
        self.slot = np.full(self.brain.n, -1, np.int64)
        self.slot[self.population] = np.arange(len(self.population))
        print(f"  readout watches {len(self.population):,} neurons")

    def train(self, samples):
        """samples: list of (moves, good_moves). Records the brain and fits
        the readout."""
        X = np.zeros((len(samples), len(self.population)), np.float32)
        Y = np.zeros((len(samples), len(MOVES)), np.float32)
        start = time.time()
        for i, (moves, good) in enumerate(samples):
            c = Cube()
            c.apply(moves)
            X[i] = self.look(c, seed=1000 + i)
            for m in good:
                Y[i, MOVES.index(m)] = 1.0 / len(good)
            if (i + 1) % 500 == 0 or i + 1 == len(samples):
                print(f"  {i + 1}/{len(samples)} trials ({time.time() - start:.0f} s)")

        # PCA: find the main patterns of variation across trials.
        # (Done through the trials-by-trials matrix, which is much smaller
        # than neurons-by-neurons here.)
        self.mu = X.mean(0)
        Xc = X - self.mu
        vals, vecs = np.linalg.eigh((Xc @ Xc.T).astype(np.float64))
        order = np.argsort(vals)[::-1][:COMPONENTS]
        vals, vecs = vals[order], vecs[:, order]
        ok = vals > vals[0] * 1e-9
        vals, vecs = vals[ok], vecs[:, ok]
        self.P = ((Xc.T @ vecs) / np.sqrt(vals)).astype(np.float32)
        Z = Xc @ self.P
        self.sd = Z.std(0) + 1e-6
        self.W, self.b = fit_ridge(Z / self.sd, Y, LAMBDA)

        guesses = self.scores(X).argmax(1)
        right = np.mean([MOVES[g] in good for g, (_, good) in zip(guesses, samples)])
        print(f"  readout picks a good move on {right:.1%} of training trials")

    def scores(self, X):
        Z = ((X - self.mu) @ self.P) / self.sd
        return Z @ self.W.T + self.b

    def choose(self, cube, seed):
        """The fly looks at the cube; the readout names a move."""
        return MOVES[int(self.scores(self.look(cube, seed)[None]).argmax())]

    def solve(self, cube, max_moves, seed):
        """Let the fly play until solved or out of moves. Returns its moves
        and whether it succeeded."""
        played = []
        for t in range(max_moves):
            if cube.is_solved():
                return played, True
            m = self.choose(cube, seed=seed + t)
            cube.move(m)
            played.append(m)
        return played, cube.is_solved()


if __name__ == "__main__":
    t0 = time.time()
    table = states_up_to(DEPTH)
    states = [v[1] for v in table.values() if v[0] >= 1]
    print(f"{len(states):,} cube states within {DEPTH} moves of solved")
    print("Finding the moves that help for each state...")
    labeled = [(moves, good_moves(moves, table)) for moves in states]
    samples = [s for s in labeled for _ in range(TRIALS)]

    brain = FlyBrain(device="cpu")
    encoder = CubeEncoder(brain)
    solver = FlySolver(brain, encoder)

    print("\nPilot run")
    solver.choose_population(states)
    print(f"\nTraining on {len(samples):,} trials")
    solver.train(samples)

    print("\nSolving fresh scrambles (new noise every look)")
    rng = random.Random(42)
    seed = 2_000_000
    for length in range(1, DEPTH + 2):
        wins, total_moves = 0, 0
        for _ in range(SCRAMBLES):
            c = Cube()
            c.scramble(length, rng)
            played, ok = solver.solve(c, length + EXTRA_MOVES, seed)
            seed += 100
            wins += ok
            total_moves += len(played) if ok else 0
        note = "  <- beyond training" if length > DEPTH else ""
        avg = f", {total_moves / wins:.1f} moves on average" if wins else ""
        print(f"  {length}-move scrambles: solved {wins}/{SCRAMBLES}{avg}{note}")

    print("\nOne solve, step by step:")
    c = Cube()
    scramble = c.scramble(DEPTH, rng)
    print(f"Scramble: {' '.join(scramble)}")
    print(c)
    played, ok = solver.solve(c, DEPTH + EXTRA_MOVES, seed)
    print(f"\nFly played: {' '.join(played)}  ->  {'SOLVED' if ok else 'not solved'}")
    print(c)
    print(f"\nDone in {time.time() - t0:.0f} s")

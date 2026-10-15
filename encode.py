"""
encode.py - turn a Rubik's cube into signals for the fly brain.

The idea: the cube has 54 stickers, each one of 6 colors, so there are
54 x 6 = 324 possible "sticker S is color C" facts. We give each fact its
own small group of the fly's visual projection neurons (the cells that
carry visual information from the eyes into the brain). To show the fly
a cube, we stimulate the 54 groups whose facts are true right now.

This is a made-up code, not real vision: a real fly can't read a cube's
colors this way. Everything downstream of these neurons is the real wiring.

Run this file directly to test it:  python encode.py
"""
import time

import numpy as np

from cube import Cube, START_COLORS

COLORS = sorted(START_COLORS.values())   # the 6 colors in a fixed order
N_STICKERS = 54
N_FEATURES = N_STICKERS * len(COLORS)    # 324


class CubeEncoder:
    def __init__(self, brain, strength=0.8, seed=0):
        """brain: a FlyBrain. strength: voltage added to each stimulated
        neuron per step. seed: which random assignment of neurons to use."""
        self.strength = strength
        neurons = brain.cells(["visual_projection"])
        if len(neurons) < N_FEATURES * 3:
            raise ValueError(f"only {len(neurons)} visual projection neurons; too few for {N_FEATURES} groups")

        # Shuffle the neurons, then deal them into 324 groups like cards.
        # A fixed seed means the same assignment every time you run it.
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(neurons)
        self.groups = np.array_split(shuffled, N_FEATURES)
        self.n_neurons = len(neurons)

    def feature_numbers(self, cube):
        """Which of the 324 facts are true: one number per sticker."""
        return [s * len(COLORS) + COLORS.index(color)
                for s, color in enumerate(cube.state())]

    def inject(self, cube):
        """The (neurons, voltage) pairs to pass to brain.step(inject=...)."""
        return [(self.groups[f], self.strength) for f in self.feature_numbers(cube)]


def show(brain, encoder, cube, steps=10, seed=1):
    """Show the fly one cube for `steps` steps (20 ms each), starting from a
    silent brain. Returns the set of neurons that fired at least once."""
    brain.reset(seed)
    inject = encoder.inject(cube)
    fired_ever = set()
    for _ in range(steps):
        fired = brain.step(inject=inject)
        fired_ever.update(fired.tolist())
    return fired_ever


def similarity(a, b):
    """Jaccard similarity: shared neurons / all neurons. 1 = identical, 0 = no overlap."""
    return len(a & b) / max(1, len(a | b))


# ---------- Self-test: runs only when you do `python encode.py` ----------
if __name__ == "__main__":
    from flybrain import FlyBrain

    brain = FlyBrain(device="cpu")
    encoder = CubeEncoder(brain)
    sizes = [len(g) for g in encoder.groups]
    print(f"{encoder.n_neurons:,} visual projection neurons -> {N_FEATURES} groups "
          f"of {min(sizes)}-{max(sizes)} neurons")

    solved = Cube()
    turned = Cube()
    turned.move("R")
    print(f"solved and R-turned cubes differ in "
          f"{sum(a != b for a, b in zip(solved.state(), turned.state()))} of 54 stickers")

    print("\nWarming up (numba compiles on the first step, this can take a minute)...")
    show(brain, encoder, solved, steps=1)

    start = time.time()
    a1 = show(brain, encoder, solved, seed=1)
    seconds_per_step = (time.time() - start) / 10
    a2 = show(brain, encoder, solved, seed=2)   # same cube, different noise
    b1 = show(brain, encoder, turned, seed=1)   # different cube, same noise
    b2 = show(brain, encoder, turned, seed=2)
    brain.reset(1)
    quiet = set()
    for _ in range(10):                          # no stimulation at all
        quiet.update(brain.step().tolist())

    print(f"\nNeurons that fired in 0.2 s of fly time:")
    print(f"  no cube shown:         {len(quiet):,}")
    print(f"  solved cube:           {len(a1):,}")
    print(f"  R-turned cube:         {len(b1):,}")
    print(f"\nHow similar were the responses? (1.0 = identical)")
    print(f"  same cube, different noise:  {similarity(a1, a2):.3f}")
    print(f"  different cube, same noise:  {similarity(a1, b1):.3f}")
    print(f"  different cube, different noise:  {similarity(a1, b2):.3f}")
    print(f"\nSpeed: {seconds_per_step * 1000:.0f} ms per step on this computer")

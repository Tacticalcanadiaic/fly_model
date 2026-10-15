"""
levels.py - how much cube information survives at each level of the fly?

The signal path:  our input (visual projection) -> optic lobe / central brain
                  -> descending neurons -> nerve cord -> motor neurons

We record the whole brain once, then run the same two tests as
experiment.py with the readout restricted to one population at a time.

To keep it fair, every population is cut down to the same number of
randomly chosen neurons (the size of the smallest big group, the nerve-cord
motor neurons). Otherwise big populations would win just by being big.

Run:  python levels.py
"""
import time

import numpy as np
from flybrain import FlyBrain

import experiment
from encode import CubeEncoder
from experiment import make_samples, record, score, sticker_facts

experiment.STEPS = 25   # 0.5 s per trial: the signal has a long way to travel

# Ordered roughly along the signal path, from input to muscles
POPULATIONS = {
    "visual projection (input)": "visual_projection",
    "optic lobe": "ol_intrinsic",
    "central brain": "cb_intrinsic",
    "descending": "descending_neuron",
    "nerve cord": "vnc_intrinsic",
    "ascending": "ascending_neuron",
    "motor, brain": "cb_motor",
    "motor, nerve cord": "vnc_motor",
}

if __name__ == "__main__":
    t0 = time.time()
    samples = make_samples()
    brain = FlyBrain(device="cpu")
    encoder = CubeEncoder(brain)
    superclass = brain.superclass.astype(str)
    pops = {name: np.flatnonzero(superclass == sc) for name, sc in POPULATIONS.items()}
    size = len(pops["motor, nerve cord"])
    rng = np.random.default_rng(0)

    print(f"\nRecording the whole brain, {experiment.STEPS} steps per trial")
    X = record(brain, encoder, samples, "fly")

    rows = {"no brain (stickers)": (sticker_facts(samples, encoder), None)}
    for name, idx in pops.items():
        pick = idx if len(idx) <= size else np.sort(rng.choice(idx, size, replace=False))
        feats = X[:, pick].astype(np.float32)
        rows[f"{name} [{len(pick)}]"] = (feats, feats.mean())
    feats = X[:, pops["descending"]].astype(np.float32)
    rows[f"descending, all [{len(pops['descending'])}]"] = (feats, feats.mean())
    del X

    print(f"\nShare of test cubes where the guessed move was right")
    print(f"(each population cut to {size} random neurons; [n] = neurons used)\n")
    print(f"{'readout watches':34}{'spikes/neuron':>14}{'recognize':>11}{'generalize':>12}")
    chances = {}
    for name, (feats, activity) in rows.items():
        accs = []
        for task in ("recognize", "generalize"):
            acc, chances[task] = score(feats, samples, task)
            accs.append(acc)
        act = "" if activity is None else f"{activity:.2f}"
        print(f"{name:34}{act:>14}{accs[0]:>11.1%}{accs[1]:>12.1%}")
    print(f"{'chance (random guess)':34}{'':>14}{chances['recognize']:>11.1%}{chances['generalize']:>12.1%}")
    print(f"\nDone in {time.time() - t0:.0f} s")

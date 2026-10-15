"""
motors.py - what motor neurons does the fly have, and does a cube reach them?

Part 1 lists every category ("superclass") of neuron in flybrain's data,
then the cell types inside any motor category, so we can see whether
legs and wings can be told apart.

Part 2 shows the fly a cube for 0.5 s and counts spikes in each motor
category and in flybrain's prebuilt behavior groups, compared with no cube.

Run:  python motors.py
"""
from collections import Counter

import numpy as np
from flybrain import FlyBrain

from cube import Cube
from encode import CubeEncoder

STEPS = 25          # 25 x 20 ms = 0.5 s of fly time
SEEDS = range(5)    # repeat each condition 5 times and average


def spike_counts(brain, inject, seed):
    """Total spikes per neuron over STEPS steps, starting from a silent brain."""
    brain.reset(seed)
    counts = np.zeros(brain.n, np.int32)
    for _ in range(STEPS):
        counts[brain.step(inject=inject)] += 1
    return counts


if __name__ == "__main__":
    brain = FlyBrain(device="cpu")
    encoder = CubeEncoder(brain)
    superclass = brain.superclass.astype(str)
    cell_type = brain.cell_type.astype(str)

    # ---------- Part 1: what's in there ----------
    print("Neuron categories (superclass): count")
    for name, n in Counter(superclass).most_common():
        print(f"  {name:32} {n:>7,}")

    motor_classes = sorted({str(s) for s in superclass if "motor" in s.lower()})
    print(f"\nMotor categories found: {motor_classes or 'none'}")
    for mc in motor_classes:
        types = Counter(cell_type[superclass == mc])
        print(f"\n{mc}: {sum(types.values()):,} neurons, {len(types)} cell types. Most common:")
        for t, n in types.most_common(40):
            print(f"  {t:40} {n}")

    # ---------- Part 2: does a cube reach them? ----------
    populations = {mc: np.flatnonzero(superclass == mc) for mc in motor_classes}
    populations["descending_neuron"] = brain.cells(["descending_neuron"])
    for name, idx in brain.groups.items():
        populations[f"group: {name}"] = np.asarray(idx)

    turned = Cube()
    turned.move("R")
    conditions = {"no cube": [], "solved cube": encoder.inject(Cube()),
                  "R-turned cube": encoder.inject(turned)}

    print("\nWarming up (numba compiles on the first step)...")
    spike_counts(brain, [], 0)

    results = {}
    for cname, inject in conditions.items():
        runs = [spike_counts(brain, inject, seed) for seed in SEEDS]
        results[cname] = np.mean(runs, axis=0)

    print(f"\nAverage spikes per neuron in {STEPS * 20} ms ({len(SEEDS)} runs each)")
    print(f"{'population':34}{'neurons':>8}" + "".join(f"{c:>15}" for c in conditions))
    for pname, idx in populations.items():
        if len(idx) == 0:
            continue
        row = "".join(f"{results[c][idx].mean():>15.3f}" for c in conditions)
        print(f"{pname:34}{len(idx):>8,}{row}")

    print("\nMost active motor/descending neurons with a cube, compared with no cube:")
    watch = np.concatenate([populations[k] for k in motor_classes + ["descending_neuron"]])
    gain = results["solved cube"][watch] - results["no cube"][watch]
    for i in watch[np.argsort(gain)[::-1][:15]]:
        print(f"  {cell_type[i]:24} side {brain.side[i]}  {superclass[i]:22}"
              f"  no cube {results['no cube'][i]:.1f}  ->  cube {results['solved cube'][i]:.1f} spikes")

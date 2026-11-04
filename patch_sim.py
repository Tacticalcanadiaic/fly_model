"""
patch_sim.py - run the cut-down optic lobe patch as a spiking network in PyTorch.


This is our own simulator, built so we can train it later. It follows the same
idea as flybrain (leaky integrate-and-fire neurons, 20 ms steps), but it is not
a copy of flybrain's exact settings, which we don't know.

It checks three things:
  1. Which way the connection matrix points (rows = senders or receivers?).
  2. How strong one connection should be, using a quick sweep.
  3. That spikes flow from the input neurons into the rest of the patch.

Needs patch_brain.npz and patch_weights.npz from cut_columns.py.
"""

import numpy as np
import scipy.sparse as sp
import torch

# ---------------- Settings you can change ----------------
DT = 0.02            # seconds per step (same as flybrain)
TAU = 0.02           # how fast a neuron's charge leaks away, in seconds
THRESHOLD = 1.0      # charge needed to fire
TARGET_KICK = 0.15   # a typical single connection adds 15% of the threshold
STRENGTH = 0.8       # input added to each input neuron per step while stimulating
STIM_STEPS = 10      # stimulate for 10 steps (0.2 s)...
TOTAL_STEPS = 25     # ...then watch until 25 steps (0.5 s)
INPUT_TYPES = ["R7", "R8", "L1", "L2", "L3", "L4", "L5"]  # the eye's first stages

# Well-known connections in the fly's visual system: (sender, receivers)
KNOWN_PAIRS = [("L1", ["Mi1"]), ("L2", ["Tm1"]), ("L3", ["Tm9"]),
               ("Mi1", ["T4a", "T4b", "T4c", "T4d"]),
               ("Tm1", ["T5a", "T5b", "T5c", "T5d"])]

DECAY = float(np.exp(-DT / TAU))   # fraction of charge kept each step


# ---------------- Step 1: which way does the matrix point? ----------------
def rows_are_senders(W, types):
    """Compare known sender->receiver connections in both directions."""
    forward = backward = 0.0
    for pre, posts in KNOWN_PAIRS:
        a = np.where(types == pre)[0]
        b = np.where(np.isin(types, posts))[0]
        if len(a) and len(b):
            forward += abs(W[a][:, b]).sum()
            backward += abs(W[b][:, a]).sum()
    print(f"Known connections, reading rows as senders:   {forward:,.0f}")
    print(f"Known connections, reading rows as receivers: {backward:,.0f}")
    if forward > 3 * backward:
        return True
    if backward > 3 * forward:
        return False
    raise SystemExit("Can't tell which way the matrix points. Paste this output to Claude.")


def pick_inputs(brain, types):
    """Input neurons: the eye's first stages, or flybrain's own visual inputs."""
    rows = np.where(np.isin(types, INPUT_TYPES))[0]
    if len(rows):
        return rows
    if "visual" in brain.files:   # flybrain's visual input list, in full-brain rows
        rows = np.where(np.isin(brain["original_rows"], brain["visual"]))[0]
    if len(rows) == 0:
        raise SystemExit("No input neurons found in the patch. Paste this to Claude.")
    return rows


# ---------------- Step 2: the spiking network ----------------
class SpikeFn(torch.autograd.Function):
    """Fire if charge >= threshold. During training, pretend it's a smooth curve."""

    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v >= THRESHOLD).float()

    @staticmethod
    def backward(ctx, grad):
        (v,) = ctx.saved_tensors
        return grad / (1 + 10 * (v - THRESHOLD).abs()) ** 2   # the surrogate gradient


class Patch(torch.nn.Module):
    def __init__(self, W_in, scale):
        """W_in[receiver, sender] = connection strength."""
        super().__init__()
        coo = W_in.tocoo()
        self.n = W_in.shape[0]
        self.register_buffer("index", torch.tensor(np.vstack([coo.row, coo.col]),
                                                   dtype=torch.long))
        # The connection strengths. Being a Parameter means training can change them.
        self.weights = torch.nn.Parameter(torch.tensor(coo.data * scale,
                                                       dtype=torch.float32))

    def forward(self, drive):
        """drive: (steps, n, batch) input for every neuron at every step.
        Returns spikes with the same shape."""
        W = torch.sparse_coo_tensor(self.index, self.weights, (self.n, self.n), check_invariants=False)
        v = torch.zeros(drive.shape[1:], device=drive.device)
        s = torch.zeros_like(v)
        out = []
        for t in range(drive.shape[0]):
            v = DECAY * v + torch.sparse.mm(W, s) + drive[t]   # leak, then add input
            s = SpikeFn.apply(v)
            v = v * (1 - s)                                    # reset neurons that fired
            out.append(s)
        return torch.stack(out)


def make_drive(n, inputs, device):
    drive = torch.zeros(TOTAL_STEPS, n, 1, device=device)
    drive[:STIM_STEPS, inputs, 0] = STRENGTH
    return drive


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    brain = np.load("patch_brain.npz")
    W = sp.load_npz("patch_weights.npz").tocsr()
    types = brain["cell_type"]
    n = W.shape[0]
    print(f"Patch: {n:,} neurons, {W.nnz:,} connections, running on {device}\n")

    senders = rows_are_senders(W, types)
    W_in = W.T.tocsr() if senders else W        # we want rows = receivers
    print("Rows are", "senders" if senders else "receivers", "\n")

    inputs = pick_inputs(brain, types)
    others = np.setdiff1d(np.arange(n), inputs)
    print(f"Input neurons: {len(inputs):,}")
    kinds, counts = np.unique(types[inputs], return_counts=True)
    print("  " + ", ".join(f"{k} ({c})" for k, c in zip(kinds, counts)), "\n")

    base = TARGET_KICK / np.median(np.abs(W.data))
    drive = make_drive(n, inputs, device)

    # ---- Sweep the connection scale ----
    print("Scale  | others that fired | spikes per step after stimulus ends")
    with torch.no_grad():                       # no training yet, so skip gradients
        for mult in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            spikes = Patch(W_in, base * mult).to(device)(drive)[:, :, 0].cpu().numpy()
            fired = spikes[:, others].any(axis=0).mean()
            after = spikes[STIM_STEPS + 3:, others].sum(axis=1).mean()
            print(f"{mult:>4}x  | {fired:>17.1%} | {after:,.1f}")

        # ---- A closer look at 1x ----
        spikes = Patch(W_in, base * 0.4).to(device)(drive)[:, :, 0].cpu().numpy()

    print("\nNon-input spikes per step at 0.4x (stimulus on for the first "
          f"{STIM_STEPS} steps):")
    print(" ".join(str(int(x)) for x in spikes[:, others].sum(axis=1)))

    active = spikes[:, others].any(axis=0)
    kinds, total = np.unique(types[others], return_counts=True)
    _, on = np.unique(types[others][active], return_counts=True)
    on_by_kind = dict(zip(np.unique(types[others][active]), on))
    share = sorted(((on_by_kind.get(k, 0) / t, k, t) for k, t in zip(kinds, total)
                    if t >= 5), reverse=True)
    print("\nCell types that responded most (types with 5+ cells):")
    for frac, kind, t in share[:10]:
        print(f"  {kind:<12} {frac:>6.0%} of {t}")


if __name__ == "__main__":
    main()





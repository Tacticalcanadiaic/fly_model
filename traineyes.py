"""
train_eyes.py - teach the fly brain itself to read a cube's colors.

see_colors.py froze the brain and only trained the readout. Here the brain's
connections learn too. flybrain can't compute gradients, so this script copies
the fly's wiring into PyTorch and runs its own version of the spiking brain:

  1. Copy the wiring: every connection's source, target and strength.
  2. Calibrate: tune the noise and overall connection strength until the
     PyTorch brain fires about as much as flybrain does, quietly and while
     looking at a cube.
  3. Train with surrogate gradients. Spikes stay all-or-nothing going
     forward; going backward, each spike is treated as a smooth curve so
     the learning signal can reach every connection.

Each connection's strength is its fly value times a learned gain. Gains start
at 1, so training begins from the real fly. Excitatory connections stay
excitatory and inhibitory stay inhibitory.

Run (from ~/fly_model, after: uv pip install torch):
  uv run python train_eyes.py                 train every connection
  uv run python train_eyes.py --frozen        baseline: only the readout learns
  uv run python train_eyes.py --shuffled      control: randomly rewired brain
  uv run python train_eyes.py --show "Scramble (F R2 U')"
Flags combine. Press Ctrl+C any time; run the same command again to resume.
"""
import argparse
import math
import os
import random
import time

import numpy as np
import torch
from flybrain import FlyBrain

from cube import Cube
from encode import COLORS, N_STICKERS, CubeEncoder
from see_colors import (DEFAULT_SCRAMBLE, cube_from, cube_with_colors, draw,
                        random_scramble)

# ---------- Settings ----------
STEPS = 10               # 20 ms steps per look: 0.2 s of fly time
BATCH = 16               # cubes per training step
TOTAL_BATCHES = 3000     # stop after this many (resume to go further)
EVAL_EVERY = 50          # test and save a checkpoint this often
TEST_CUBES = 128         # fixed scrambles never trained on
READOUT_NEURONS = 4096   # random non-stimulated neurons the readout watches
LR_READOUT = 1e-3
LR_BRAIN = 1e-3
SHARPNESS = 10.0         # how pointy the surrogate spike curve is
ACTIVITY_LIMIT = 2.0     # penalize firing above 2x the calibrated level
CHECKPOINTS = "checkpoints"


# ---------- Copying the wiring out of flybrain ----------
def find(obj, names):
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return name, value
    return None, None


def show_attributes(brain):
    print("\nflybrain's attributes:")
    for k, v in sorted(vars(brain).items()):
        print(f"  {k:24} {type(v).__name__:14} {getattr(v, 'shape', '')}")


def load_wiring(brain):
    """Returns (target, source, weight) arrays, one entry per connection."""
    _, indptr = find(brain, ["indptr", "row_ptr", "offsets", "ptr"])
    indices = getattr(brain, "indices", None)
    _, weights = find(brain, ["weights", "data", "w", "values", "strengths"])
    ok = (indptr is not None and indices is not None and weights is not None
          and len(indptr) == brain.n + 1 and len(weights) == len(indices))
    if not ok:
        show_attributes(brain)
        raise SystemExit("\nI couldn't find the wiring under the names I guessed. "
                         "Paste the list above into the chat and we'll fix it.")
    indptr = np.asarray(indptr, np.int64)
    target = np.asarray(indices, np.int64)
    weight = np.asarray(weights, np.float32)
    source = np.repeat(np.arange(brain.n), np.diff(indptr))

    # PyTorch wants each (target, source) pair once, sorted. Merge duplicates
    # (the shuffled brain can create some) by adding their strengths.
    pair = target * brain.n + source
    unique, where = np.unique(pair, return_inverse=True)
    merged = np.bincount(where, weights=weight).astype(np.float32)
    print(f"  copied {len(weight):,} connections "
          f"({len(weight) - len(unique):,} duplicates merged)")
    return unique // brain.n, unique % brain.n, merged


def neuron_constants(brain):
    """Threshold and leak, if flybrain exposes them; sensible guesses if not.
    Calibration absorbs most of the error from a wrong guess."""
    name, thr = find(brain, ["threshold", "v_th", "v_thresh", "theta"])
    thr = float(np.mean(thr)) if thr is not None else 1.0
    print(f"  threshold: {thr:g}" + (f" (from brain.{name})" if name else " (guess)"))
    name, tau = find(brain, ["tau", "tau_m", "tau_mem"])
    decay = math.exp(-20.0 / float(np.mean(tau))) if tau is not None else math.exp(-1)
    print(f"  leak: keeps {decay:.0%} of voltage per step"
          + (f" (from brain.{name})" if name else " (guess)"))
    return thr, decay


# ---------- The trainable brain ----------
class Spike(torch.autograd.Function):
    """Forward: fire (1) if voltage is over threshold, else 0.
    Backward: pretend it was a smooth step, steepest at the threshold."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad / (1.0 + SHARPNESS * x.abs()) ** 2


class Synapses(torch.autograd.Function):
    """Input to every neuron from the neurons that just fired.

    PyTorch's own sparse backward builds a full neurons-by-neurons grid
    (166,700 squared = far too much memory), so this works out the gradient
    for each real connection directly, a chunk of connections at a time."""

    CHUNK = 2_000_000

    @staticmethod
    def forward(ctx, w, idx, s, n):
        A = torch.sparse_coo_tensor(idx, w, (n, n), is_coalesced=True)
        ctx.save_for_backward(w, idx, s)
        ctx.n = n
        return torch.sparse.mm(A, s)

    @staticmethod
    def backward(ctx, grad_out):
        w, idx, s = ctx.saved_tensors
        target, source = idx[0], idx[1]
        grad_w = torch.empty_like(w)
        for i in range(0, len(w), Synapses.CHUNK):
            t, src = target[i:i + Synapses.CHUNK], source[i:i + Synapses.CHUNK]
            grad_w[i:i + Synapses.CHUNK] = (grad_out[t] * s[src]).sum(1)
        # Gradient for the spikes: send grad_out back along each connection
        grad_s = _backward_spikes(w, target, source, grad_out, s)
        return grad_w, None, grad_s, None


def _backward_spikes(w, target, source, grad_out, s):
    grad_s = torch.zeros_like(s)
    for i in range(0, len(w), Synapses.CHUNK):
        sl = slice(i, i + Synapses.CHUNK)
        grad_s.index_add_(0, source[sl], w[sl, None] * grad_out[target[sl]])
    return grad_s


class TorchFly(torch.nn.Module):
    def __init__(self, n, target, source, weight, decay, readout_neurons, train_brain):
        super().__init__()
        self.n, self.decay = n, decay
        self.register_buffer("idx", torch.from_numpy(np.stack([target, source])))
        self.register_buffer("w0", torch.from_numpy(weight))
        self.register_buffer("readout_idx", torch.from_numpy(readout_neurons))
        self.log_gain = torch.nn.Parameter(torch.zeros(len(weight)),
                                           requires_grad=train_brain)
        self.readout = torch.nn.Linear(len(readout_neurons), N_STICKERS * len(COLORS))
        self.scale = 1.0         # set by calibration
        self.noise_sd = 0.1      # set by calibration

    def run(self, drive, generator):
        """drive: (neurons, cubes) voltage added every step.
        Returns readout spike counts (cubes, readout neurons) and the
        average share of neurons firing per step."""
        w = self.w0 * torch.exp(self.log_gain) * self.scale
        v = torch.zeros_like(drive)
        s = torch.zeros_like(drive)
        counts, rate = 0.0, 0.0
        for _ in range(STEPS):
            noise = torch.randn(drive.shape, generator=generator) * self.noise_sd
            v = self.decay * v * (1 - s.detach()) + Synapses.apply(w, self.idx, s, self.n) + drive + noise
            s = Spike.apply(v - 1.0)
            counts = counts + s[self.readout_idx]
            rate = rate + s.mean()
        return counts.T / STEPS, rate / STEPS

    def forward(self, drive, generator):
        counts, rate = self.run(drive, generator)
        logits = self.readout(counts).view(-1, N_STICKERS, len(COLORS))
        return logits, rate


def drive_for(cubes, encoder, n, strength):
    drive = torch.zeros(n, len(cubes))
    for b, c in enumerate(cubes):
        for neurons, _ in encoder.inject(c):
            drive[torch.from_numpy(np.asarray(neurons, np.int64)), b] = strength
    return drive


def color_targets(cubes):
    return torch.tensor([[COLORS.index(col) for col in c.state()] for c in cubes])


# ---------- Calibration ----------
def flybrain_activity(brain, encoder, cubes):
    """Average share of neurons firing per step in flybrain itself."""
    rates = []
    for i, c in enumerate(cubes):
        inject = encoder.inject(c) if c is not None else ()
        brain.reset(500 + i)
        for _ in range(STEPS):
            rates.append(len(brain.step(inject=inject)) / brain.n)
    return float(np.mean(rates))


def bisect(set_value, measure, target, low, high, label):
    """Find the value (searched on a log scale) whose activity matches target."""
    for _ in range(18):
        mid = math.sqrt(low * high)
        set_value(mid)
        if measure() < target:
            low = mid
        else:
            high = mid
    value = math.sqrt(low * high)
    set_value(value)
    print(f"  {label}: {value:.4g}  (activity {measure():.3%} vs flybrain {target:.3%})")
    return value


def calibrate(model, brain, encoder, strength):
    cubes = [cube_from(random_scramble(random.Random(i))) for i in range(4)]
    quiet_target = flybrain_activity(brain, encoder, [None] * 4)
    cube_target = flybrain_activity(brain, encoder, cubes)

    def measure(drive):
        with torch.no_grad():
            return float(model.run(drive, torch.Generator().manual_seed(0))[1])

    quiet = torch.zeros(model.n, 4)
    looking = drive_for(cubes, encoder, model.n, strength)

    def set_noise(x):
        model.noise_sd = x

    def set_scale(x):
        model.scale = x

    # Noise first, with the wiring nearly off, then the wiring's overall strength.
    model.scale = 1e-9
    bisect(set_noise, lambda: measure(quiet), max(quiet_target, 1e-5), 1e-3, 10.0, "noise level")
    bisect(set_scale, lambda: measure(looking), cube_target, 1e-6, 1e3, "wiring strength")
    return cube_target


# ---------- Testing ----------
def evaluate(model, test_cubes, encoder, strength, movable):
    model.eval()
    gen = torch.Generator().manual_seed(12345)    # same noise every test
    right = []
    with torch.no_grad():
        for i in range(0, len(test_cubes), BATCH):
            chunk = test_cubes[i:i + BATCH]
            logits, _ = model(drive_for(chunk, encoder, model.n, strength), gen)
            right.append((logits.argmax(2) == color_targets(chunk))[:, movable])
    model.train()
    return torch.cat(right).float().mean().item()


def show_reading(model, scramble, encoder, strength, use_color):
    cube = cube_from(scramble)
    with torch.no_grad():
        logits, _ = model(drive_for([cube] * 5, encoder, model.n, strength),
                          torch.Generator().manual_seed(7))
    guess = logits.sum(0).argmax(1).numpy()     # pool 5 noisy looks
    truth = np.array([COLORS.index(col) for col in cube.state()])
    print(f"\n{scramble}\n\nThe actual cube:")
    print(draw(cube, use_color=use_color))
    print(f"\nWhat the fly read ({int((guess != truth).sum())} wrong, marked *):")
    print(draw(cube_with_colors([COLORS[i] for i in guess]),
               cube_with_colors(list(guess != truth)), use_color=use_color))


# ---------- Main ----------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--frozen", action="store_true", help="only the readout learns")
    p.add_argument("--shuffled", action="store_true", help="randomly rewired control")
    p.add_argument("--batches", type=int, default=TOTAL_BATCHES)
    p.add_argument("--show", nargs="?", const=DEFAULT_SCRAMBLE,
                   help="after training, draw how the fly reads this scramble")
    p.add_argument("--plain", action="store_true", help="no colored squares")
    args = p.parse_args()
    use_color = not args.plain and os.isatty(1)

    name = f"{'shuffled' if args.shuffled else 'fly'}_{'frozen' if args.frozen else 'trained'}"
    os.makedirs(CHECKPOINTS, exist_ok=True)
    path = os.path.join(CHECKPOINTS, f"eyes_{name}.pt")
    print(f"Run: {name}  (checkpoint: {path})")
    torch.manual_seed(0)

    print("\nLoading flybrain")
    brain = FlyBrain(device="cpu")
    if args.shuffled:
        brain.indices = np.random.default_rng(7).permutation(brain.indices)
    encoder = CubeEncoder(brain)
    thr, decay = neuron_constants(brain)
    strength = encoder.strength / thr
    target, source, weight = load_wiring(brain)

    stimulated = np.concatenate(encoder.groups)
    candidates = np.setdiff1d(np.arange(brain.n), stimulated)
    readout_neurons = np.sort(np.random.default_rng(3).choice(
        candidates, min(READOUT_NEURONS, len(candidates)), replace=False))

    model = TorchFly(brain.n, target, source, weight, decay, readout_neurons,
                     train_brain=not args.frozen)
    del target, source, weight

    groups = [{"params": model.readout.parameters(), "lr": LR_READOUT}]
    if not args.frozen:
        groups.append({"params": [model.log_gain], "lr": LR_BRAIN})
    opt = torch.optim.Adam(groups)

    done = 0
    if os.path.exists(path):
        ck = torch.load(path)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        model.scale, model.noise_sd = ck["scale"], ck["noise_sd"]
        target_rate, done, history = ck["target_rate"], ck["done"], ck["history"]
        print(f"\nResuming after {done} batches")
    else:
        print("\nCalibrating against flybrain")
        target_rate = calibrate(model, brain, encoder, strength)
        history = []
    del brain                                     # free flybrain's memory

    test_rng = random.Random(999)
    test_cubes = [cube_from(random_scramble(test_rng)) for _ in range(TEST_CUBES)]
    movable = color_targets(test_cubes).float().std(0) > 0   # centers never move

    def save():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "scale": model.scale, "noise_sd": model.noise_sd,
                    "target_rate": target_rate, "done": done, "history": history},
                   path + ".tmp")
        os.replace(path + ".tmp", path)           # never leaves a half-written file

    if done == 0:
        acc = evaluate(model, test_cubes, encoder, strength, movable)
        history.append((0, acc))
        print(f"\nBefore training: {acc:.1%} of moving stickers right on test cubes "
              f"(chance {1 / len(COLORS):.1%})")

    print(f"\nTraining ({'readout only' if args.frozen else 'brain + readout'}). "
          f"Ctrl+C to stop; progress is saved every {EVAL_EVERY} batches.")
    loss_fn = torch.nn.CrossEntropyLoss()
    try:
        while done < args.batches:
            t = time.time()
            rng = random.Random(10_000 + done)        # different cubes every batch
            cubes = [cube_from(random_scramble(rng)) for _ in range(BATCH)]
            gen = torch.Generator().manual_seed(20_000 + done)
            logits, rate = model(drive_for(cubes, encoder, model.n, strength), gen)
            truth = color_targets(cubes)
            loss = loss_fn(logits.reshape(-1, len(COLORS)), truth.reshape(-1))
            # Keep the brain from learning to fire like crazy
            loss = loss + torch.relu(rate / target_rate - ACTIVITY_LIMIT) ** 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            done += 1

            if done % 10 == 0:
                train_acc = (logits.argmax(2) == truth)[:, movable].float().mean()
                print(f"  batch {done:5d}  loss {loss.item():.3f}  "
                      f"train {train_acc:.1%}  firing {rate.item() / target_rate:.2f}x  "
                      f"{time.time() - t:.1f} s/batch")
            if done % EVAL_EVERY == 0:
                acc = evaluate(model, test_cubes, encoder, strength, movable)
                history.append((done, acc))
                drift = model.log_gain.detach().abs().mean().exp().item()
                print(f"  -- test: {acc:.1%} of moving stickers right"
                      + ("" if args.frozen else f"  (connections changed by {drift:.2f}x on average)"))
                save()
    except KeyboardInterrupt:
        print("\nStopping.")
    save()
    print(f"\nSaved after {done} batches. Test accuracy over time:")
    print("  " + ", ".join(f"{b}: {a:.1%}" for b, a in history))

    if args.show:
        show_reading(model, args.show, encoder, strength, use_color)

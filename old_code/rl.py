"""
rl.py - the fly learns to solve cubes from reward and punishment alone.

  1. The fly looks at the cube; its descending neurons respond.
  2. Twelve "move units" each add up descending-neuron activity with their
     own connection strengths. The fly picks a move, mostly the best-scoring
     one but with some randomness, so it explores.
  3. The move is applied. Reward +1 if the cube got closer to solved,
     -1 if it got farther, +1 extra for finishing.
  4. Connections that led to a rewarded move get stronger; connections that
     led to a punished move get weaker. (This rule is called REINFORCE.)

Nobody ever tells the fly the right move. Only the move units learn; the
brain itself stays frozen in this stage.

Curriculum: start with cubes 1 move from solved; move up a level once the
fly succeeds PROMOTE_AT of the time; keep practicing the easier levels too.

Run:  python rl.py
"""
import random
import time
from collections import deque

import numpy as np
from flybrain import FlyBrain

from cube import Cube, MOVES
from encode import CubeEncoder

# ---------- Settings ----------
STEPS = 20              # 20 ms steps per look: 0.4 s of fly time
LOOKS_PER_STATE = 2     # noisy looks recorded per cube state, then reused
MAX_DEPTH = 3           # hardest curriculum level
TABLE_DEPTH = 5         # how far from solved the referee can judge
PROMOTE_AT = 0.9        # success rate needed to move up a level
WINDOW = 100            # ...measured over this many recent episodes
EPISODES = 20000        # most training episodes to run
LEARNING_RATE = 0.0005  # how big each connection change is
TEST_SCRAMBLES = 50     # test scrambles per length after training


# ---------- The referee: how far is each cube from solved? ----------
def key(cube):
    return "".join(cube.state())


def copy_cube(cube):
    new = Cube.__new__(Cube)          # an empty Cube, without solving it first
    new.stickers = dict(cube.stickers)
    return new


def distance_table(depth, keep_up_to):
    """Breadth-first search out from solved. Returns {state: distance},
    plus the cubes at each distance up to `keep_up_to` (to use as puzzles)."""
    solved = Cube()
    dist = {key(solved): 0}
    frontier, puzzles = [solved], {}
    for d in range(1, depth + 1):
        next_frontier = []
        for c in frontier:
            for m in MOVES:
                c2 = copy_cube(c)
                c2.move(m)
                k = key(c2)
                if k not in dist:
                    dist[k] = d
                    next_frontier.append(c2)
        frontier = next_frontier
        if d <= keep_up_to:
            puzzles[d] = frontier
        print(f"  {len(frontier):>7,} states at distance {d}")
    return dist, puzzles


# ---------- The fly's side ----------
class FlyEyes:
    """Shows the fly a cube; returns descending-neuron spike counts.
    The brain is frozen in this stage, so a state's response never changes
    except for noise. We record LOOKS_PER_STATE noisy looks per state and
    reuse them, which saves a lot of time."""

    def __init__(self, brain, encoder, rng):
        self.brain, self.encoder, self.rng = brain, encoder, rng
        self.dn = brain.cells(["descending_neuron"])
        self.slot = np.full(brain.n, -1, np.int64)
        self.slot[self.dn] = np.arange(len(self.dn))
        self.memory = {}
        self.live_looks = 0
        self.seed = 3_000_000

    def record(self, cube):
        self.brain.reset(self.seed)
        self.seed += 1
        self.live_looks += 1
        inject = self.encoder.inject(cube)
        counts = np.zeros(len(self.dn), np.uint8)
        for _ in range(STEPS):
            s = self.slot[self.brain.step(inject=inject)]
            counts[s[s >= 0]] += 1
        return counts

    def look(self, cube, fresh=False):
        if fresh:                          # testing: always a brand-new look
            return self.record(cube)
        seen = self.memory.setdefault(key(cube), [])
        if len(seen) < LOOKS_PER_STATE:
            seen.append(self.record(cube))
            return seen[-1]
        return seen[self.rng.randrange(len(seen))]


class MoveUnits:
    """Twelve move units, each a weighted sum of descending-neuron activity.
    Their connection strengths W are the only thing that learns."""

    def __init__(self, n_inputs, seed=0):
        self.W = np.zeros((len(MOVES), n_inputs))
        self.mean = np.zeros(n_inputs)     # running average activity per neuron
        self.var = np.ones(n_inputs)       # running spread per neuron
        self.baseline = 0.0                # running average reward
        self.rng = np.random.default_rng(seed)

    def normalize(self, counts, update):
        """Put every neuron on a common scale: 0 = its usual activity."""
        x = counts.astype(np.float64)
        if update:
            self.mean += 0.01 * (x - self.mean)
            self.var += 0.01 * ((x - self.mean) ** 2 - self.var)
        return (x - self.mean) / np.sqrt(self.var + 1e-3)

    def choose(self, z, explore):
        scores = self.W @ z
        p = np.exp(scores - scores.max())
        p /= p.sum()                       # turn scores into probabilities
        a = int(self.rng.choice(len(MOVES), p=p)) if explore else int(p.argmax())
        return a, p

    def learn(self, z, action, p, reward):
        """REINFORCE: nudge the chosen move up if reward beat expectations,
        down if it fell short, in proportion to each neuron's activity."""
        surprise = reward - self.baseline
        self.baseline += 0.01 * (reward - self.baseline)
        push = -p
        push[action] += 1.0
        self.W += LEARNING_RATE * surprise * np.outer(push, z)


def play(cube, max_moves, eyes, units, dist, explore=True, learn=True, fresh=False):
    """One episode. Returns (solved?, moves played)."""
    for t in range(max_moves):
        if cube.is_solved():
            return True, t
        before = dist[key(cube)]
        z = units.normalize(eyes.look(cube, fresh), update=learn)
        a, p = units.choose(z, explore)
        cube.move(MOVES[a])
        after = dist.get(key(cube))
        if after is None:                  # wandered too far for the referee
            if learn:
                units.learn(z, a, p, -1.0)
            return False, t + 1
        reward = (1.0 if after < before else -1.0) + (1.0 if after == 0 else 0.0)
        if learn:
            units.learn(z, a, p, reward)
    return cube.is_solved(), max_moves


# ---------- Main ----------
if __name__ == "__main__":
    t0 = time.time()
    print("Building the referee's distance table (takes a little while)...")
    dist, puzzles = distance_table(TABLE_DEPTH, keep_up_to=MAX_DEPTH)

    brain = FlyBrain(device="cpu")
    encoder = CubeEncoder(brain)
    rng = random.Random(0)
    eyes = FlyEyes(brain, encoder, rng)
    units = MoveUnits(len(eyes.dn))
    print(f"\nThe fly acts through {len(eyes.dn):,} descending neurons")

    print("\nTraining")
    level, recent = 1, deque(maxlen=WINDOW)
    for ep in range(1, EPISODES + 1):
        d = rng.randint(1, level)                       # practice all levels so far
        cube = copy_cube(rng.choice(puzzles[d]))
        solved, _ = play(cube, d + 2, eyes, units, dist)
        if d == level:
            recent.append(solved)
        rate = np.mean(recent) if recent else 0.0
        full = len(recent) == WINDOW
        if ep % 200 == 0:
            print(f"  episode {ep:>5}: level {level}, recent success {rate:.0%}, "
                  f"live looks {eyes.live_looks:,} ({time.time() - t0:.0f} s)")
        if full and rate >= PROMOTE_AT:
            if level == MAX_DEPTH:
                print(f"  episode {ep:>5}: passed level {level}. Training complete!")
                break
            level += 1
            recent.clear()
            print(f"  episode {ep:>5}: promoted to level {level}")
    else:
        print(f"  stopped after {EPISODES} episodes at level {level}")

    np.savez("rl_weights.npz", W=units.W, mean=units.mean, var=units.var)
    print("Saved the learned connections to rl_weights.npz")

    print("\nTesting: no exploration, fresh noise on every look")
    test_rng = random.Random(42)
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

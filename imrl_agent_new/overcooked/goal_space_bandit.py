from collections import defaultdict, Counter
import random

class GoalSpaceBandit:
    def __init__(self, spaces: dict, epsilon=0.2):
        self.spaces = spaces
        self.epsilon = epsilon
        self.avg_lp  = defaultdict(float)
        self.count   = Counter()

    def next_goal(self, context):
        # choose space
        global k
        if random.random() < self.epsilon or not self.avg_lp:
            k = random.choice(list(self.spaces))
        else:
            # exploit ∝ positive learning-progress
            weights = {k:max(0,p) for k,p in self.avg_lp.items()}
            total   = sum(weights.values()) or 1e-6
            r       = random.random() * total
            acc = 0
            for k,w in weights.items():
                acc += w
                if acc >= r: break
        # sample g inside space
        g = self.spaces[k].sample()
        return k, g

    def update(self, k, r_i):
        self.count[k] += 1
        n = self.count[k]
        self.avg_lp[k] += (r_i - self.avg_lp[k]) / n

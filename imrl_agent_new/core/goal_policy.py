import random
from collections import defaultdict, Counter
from typing import Dict, Tuple, Any

class GoalSpacePolicy:
    """
    ε-greedy non-stationary bandit over goal-spaces.
    80 % exploit   – pick space ∝ current averaged learning-progress
    20 % explore   – uniform random
    """
    def __init__(self, goal_spaces: Dict[str, Any], epsilon: float = 0.20):
        self.spaces: Dict[str, Any] = goal_spaces
        self.epsilon: float = epsilon

        # running stats
        self.avg_progress = defaultdict(float)     # r̄_k
        self.count        = Counter()              # how many updates per space

    # ------------------------------------------------------------------ public
    def next_goal(self, context: Any) -> Tuple[str, Any]:
        """
        Returns (space_id, g) where g is a concrete goal vector from that space.
        context is kept for future extensions but unused in default implementation.
        """
        # 1) choose goal-space k
        global k
        if random.random() < self.epsilon or not self.avg_progress:
            k = random.choice(list(self.spaces))          # pure exploration
        else:                                             # exploit LP
            weights = {k: max(0.0, lp) for k, lp in self.avg_progress.items()}
            total   = sum(weights.values()) or 1e-6       # avoid /0
            r = random.random() * total
            acc = 0.0
            for k, w in weights.items():
                acc += w
                if acc >= r:
                    break

        # 2) sample concrete goal inside chosen space
        g = self.spaces[k].sample()
        return k, g

    def update(self, k: str, r_i: float):
        """
        Call this **only for exploitation episodes** (the 20 % that used Π).
        Adds intrinsic reward r_i to the running mean for space k.
        """
        self.count[k] += 1
        n = self.count[k]
        self.avg_progress[k] += (r_i - self.avg_progress[k]) / n

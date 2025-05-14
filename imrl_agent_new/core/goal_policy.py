# goal_policy.py
import random
from collections import defaultdict
from typing import Dict, Tuple, Any


class GoalSpacePolicy:
    """
    ε-greedy, non-stationary bandit over goal-spaces.

    •  ε %  of the time  → uniform exploration
    • (1-ε) % of the time
        – if at least one space has positive avg-LP → sample ∝ LP (soft max)
        – else fallback to uniform   (no progress anywhere yet)

    Learning-progress is tracked with an exponential moving average so
    old data is forgotten and the agent can return to a space if it
    starts improving again.
    """

    def __init__(
        self,
        goal_spaces: Dict[str, Any],
        *,
        epsilon: float = 0.20,
        lp_alpha: float = 0.10,           # EMA smoothing factor
    ):
        self.spaces: Dict[str, Any] = goal_spaces
        self.epsilon = epsilon
        self.alpha   = lp_alpha

        # running exponential averages of intrinsic reward
        self.avg_lp = defaultdict(float)    # r̄_k  (initially 0)

    # ---------------------------------------------------------------- PUBLIC
    def next_goal(self, context: Any | None = None) -> Tuple[str, Any]:
        """
        Returns (space_id, goal_vector g).

        `context` kept for future use (e.g. context-dependent priors) but
        ignored in this simple implementation.
        """
        # ---------- choose a space -----------------------------------------
        if random.random() < self.epsilon:
            space_id = random.choice(list(self.spaces))           # pure explore
        else:
            # exploit: soft-probability ∝ max(avg_lp, 0)
            weights = {k: max(0.0, self.avg_lp[k]) for k in self.spaces}
            total   = sum(weights.values())
            if total == 0.0:                                       # no progress yet
                space_id = random.choice(list(self.spaces))
            else:
                r = random.random() * total
                acc = 0.0
                for k, w in weights.items():
                    acc += w
                    if acc >= r:
                        space_id = k
                        break

        # ---------- sample goal inside that space --------------------------
        g = self.spaces[space_id].sample()
        return space_id, g

    def update(self, space_id: str, intrinsic_reward: float):
        """
        Call AFTER an *exploitation* episode with Π.

        Updates the exponential moving average of learning-progress for
        the given space.
        """
        old = self.avg_lp[space_id]
        new = old * (1.0 - self.alpha) + intrinsic_reward * self.alpha
        self.avg_lp[space_id] = new

    # ---------------------------------------------------------------- HELPERS
    def refresh_spaces(self, new_spaces: Dict[str, Any]):
        """
        Replace the dict of goal-spaces (e.g. when you load a new layout).
        Keeps existing LP stats for overlapping keys, initialises new keys to 0.
        """
        self.spaces = new_spaces
        for k in new_spaces:
            self.avg_lp.setdefault(k, 0.0)

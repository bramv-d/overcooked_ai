# goal_policy.py
import random
from collections import defaultdict
from typing import Any, Dict, Tuple


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
        epsilon: float = 0.20,
        lp_alpha: float = 0.10,           # EMA smoothing factor
    ):
        self.spaces: Dict[str, Any] = goal_spaces
        self.epsilon = epsilon
        self.alpha   = lp_alpha

        # running exponential averages of intrinsic reward
        self.ir_by_space = defaultdict(float)  # r̄_k  (initially 0)

    # ---------------------------------------------------------------- PUBLIC
    def next_goal(self) -> Tuple[str, Any]:
        """
        Returns (space_id, goal_vector g).

        """
        # ---------- choose a space -----------------------------------------
        if random.random() < self.epsilon or not self.ir_by_space:
            space_id = random.choice(list(self.spaces))           # pure explore
        else:
            # exploit: soft-probability ∝ max(avg_lp, 0)
            space_id = self.ir_by_space
            space_id = max(self.ir_by_space, key=lambda k: max(self.ir_by_space[k], 0.0))

        # ---------- sample goal inside that space --------------------------
        g = self.spaces[space_id].sample()
        return space_id, g

    def update(self, space_id: str, intrinsic_reward: float):
        """
        Call AFTER an *exploitation* episode with Π.

        Updates the exponential moving average of learning-progress for
        the given space.
        """
        self.ir_by_space[space_id] = intrinsic_reward

    # ---------------------------------------------------------------- HELPERS
    def refresh_spaces(self, new_spaces: Dict[str, Any]):
        """
        Replace the dict of goal-spaces (e.g. when you load a new layout).
        Keeps existing LP stats for overlapping keys, initialises new keys to 0.
        """
        self.spaces = new_spaces
        for k in new_spaces:
            self.ir_by_space.setdefault(k, 0.0)

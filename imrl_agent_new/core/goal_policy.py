# goal_policy.py
import random
from collections import defaultdict
from typing import Any, Dict

import numpy as np


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
        self.ir_by_space = defaultdict(float, {space_id: 0 for space_id in self.spaces})

        self.shared_reward_by_space = defaultdict(float, {space_id: 0 for space_id in self.spaces})

    # ---------------------------------------------------------------- PUBLIC
    def next_goal(self, shared_episode) -> str:
        """
        Returns (space_id).
        """
        if shared_episode:
            return max(self.shared_reward_by_space, key=self.shared_reward_by_space.get)

        # ---------- choose a space -----------------------------------------
        if random.random() < self.epsilon or not self.ir_by_space:
            space_id = random.choice(list(self.spaces))           # pure explore
        else:
            ir_values = self.ir_by_space
            max_val = max(ir_values.values())
            # Use isclose to handle float comparisons
            best_spaces = [k for k, v in ir_values.items() if np.isclose(v, max_val)]
            space_id = random.choice(best_spaces)
        return space_id

    def update(self, space_id: str, intrinsic_reward: float, shared_episode_reward: float):
        """
        Update EMA of learning progress for the given goal space.
        """
        old_ir = self.ir_by_space.get(space_id)
        old_shared_reward = self.shared_reward_by_space.get(space_id)
        self.ir_by_space[space_id] = old_ir + self.alpha * (intrinsic_reward - old_ir)
        self.shared_reward_by_space[space_id] = old_shared_reward + self.alpha * (
                    shared_episode_reward - old_shared_reward)

from typing import List

import numpy as np

from base_dir.shared_files.goal_spaces import Goal


class GoalSpaceEMA:
    """Holds the EMA of the intrinsic reward for a goal space."""

    def __init__(self, goal: Goal, ema: float):
        self.goal: Goal = goal
        self.ema = ema


def _softmax_sample(goals: List[GoalSpaceEMA], tau: float = 0.5) -> GoalSpaceEMA:
    """
    Draw one GoalSpaceEMA with P(g) ∝ exp(ema / τ).
    Lower τ → greedier.  We subtract max logits for numerical stability.
    """
    logits = np.array([g.ema for g in goals])
    logits -= logits.max()  # avoid overflow
    probs = np.exp(logits / tau)
    probs /= probs.sum()
    idx = np.random.choice(len(goals), p=probs)
    return goals[idx]

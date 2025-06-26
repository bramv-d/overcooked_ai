import random
from collections import defaultdict
from typing import Dict, List

from IMGEP_agent.hyper_parameters import AgentConfig, IR_BONUS_CAP, IR_BONUS_SLOPE
from IMGEP_agent.shared_agent.goal_spaces import Goal
from IMGEP_agent.shared_agent.knowledge_base import KnowledgeBase, RolloutRecord

EPSILON = 1e-6  # numeric tolerance


class GoalSpaceEMA:
    """Holds the EMA of the intrinsic reward for a goal space."""

    def __init__(self, goal: Goal, ema: float):
        self.goal: Goal = goal
        self.ema = ema


def _top_ema_candidates(gs_emas: List[GoalSpaceEMA], eps: float = EPSILON) -> List[GoalSpaceEMA]:
    """Return all GS-EMA objects whose EMA is within eps of the maximum."""
    if not gs_emas:
        return []
    m = max(gs.ema for gs in gs_emas)
    return [gs for gs in gs_emas if abs(gs.ema - m) < eps]


def select_goal(
        goal_space_ema: List[GoalSpaceEMA],
        goal_spaces: List[Goal],
        config: AgentConfig,
) -> Goal:
    # Based on the goal space EMA, select a goal to pursue.
    goal_space_ema = _top_ema_candidates(goal_space_ema)
    return random.choice(goal_space_ema).goal if goal_space_ema else random.choice(goal_spaces)

def update_goal_space_ema(
        kb: KnowledgeBase,
        goal_spaces: List[Goal],
        config: AgentConfig,
) -> List[GoalSpaceEMA]:
    """
    Compute an (approximate) EMA of intrinsic reward per goal space,
    adding a small bonus for sparsely tried goals so they remain eligible.
    """
    # 1) get the N most recent exploit rollouts
    recent: List[RolloutRecord] = kb.nearest(config.n_recent, exploit=True)

    # 2) bucket by goal_space_id
    by_goal: Dict[int, List[RolloutRecord]] = defaultdict(list)
    for r in recent:
        by_goal[r.goal_space_id].append(r)

    # 3) compute a per-goal EMA + bonus
    bonus_cutoff = int(IR_BONUS_CAP / IR_BONUS_SLOPE)
    out: List[GoalSpaceEMA] = []
    for g in goal_spaces:
        recs = by_goal.get(g.goal_id, [])
        n = len(recs)
        avg_ir = (sum(r.intrinsic_reward for r in recs) / n) if n > 0 else 0.0

        # if very sparse, give a decreasing bonus
        if n < bonus_cutoff:
            bonus = IR_BONUS_CAP - IR_BONUS_SLOPE * n
            avg_ir += bonus

        out.append(GoalSpaceEMA(g, avg_ir))

    return out

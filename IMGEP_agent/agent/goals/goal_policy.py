import random
from collections import defaultdict
from typing import Dict, List

from IMGEP_agent.agent.goals.goal_spaces import Goal
from IMGEP_agent.agent.knowledge_base import KnowledgeBase, RolloutRecord
from IMGEP_agent.hyper_parameters import BONUS_CAP, BONUS_FLOOR, BONUS_SLOPE, GOAL_EMA_ALPHA, N_RECENT


class GoalSpaceEMA:
    # Holds the EMA of the intrinsic reward for a goal space
    def __init__(self, goal: Goal, ema):
        self.goal: Goal = goal
        self.ema = ema


def select_goal_space(goal_space_emas: List[GoalSpaceEMA]) -> Goal:
    # 20% exploration
    # if random.random() > GOAL_GREEDY_PROB:
    #     return random.choice(goal_space_emas).goal

    # find max EMA, allow for small floating point tolerance
    max_ema = max(gs.ema for gs in goal_space_emas)
    epsilon = 1e-6
    candidates = [gs.goal for gs in goal_space_emas if abs(gs.ema - max_ema) < epsilon]

    return random.choice(candidates)


def update_goal_space_ema(
        kb: KnowledgeBase,
        goal_spaces: List[Goal],
        alpha: float = GOAL_EMA_ALPHA
) -> List[GoalSpaceEMA]:
    """
    Compute an EMA of intrinsic reward for each GoalSpace.

    If we have fewer than 10 recent exploit roll-outs for a goal, add
    a small bonus so that very sparse goal spaces remain eligible for
    selection.

        bonus = max(0,  BONUS_CAP - (len(records) * BONUS_SLOPE))
               = 0.10 for 0  records
               = 0.01 for 9  records
               = 0    for 10 or more records
    """
    # 1) pull the N most-recent exploit records once
    recent: List[RolloutRecord] = kb.nearest(N_RECENT, exploit=True)

    # 2) bucket by goal_space_id
    by_goal: Dict[int, List[RolloutRecord]] = defaultdict(list)
    for r in recent:
        by_goal[r.goal_space_id].append(r)

    # 3) build EMA objects
    ema_list: List[GoalSpaceEMA] = []
    for g in goal_spaces:
        records = by_goal.get(g.goal_id, [])
        n = len(records)

        # mean intrinsic reward over the bucket (0 if empty)
        avg_ir = sum(r.intrinsic_reward for r in records) / n if n else 0.0

        # data-scarcity bonus
        if n < BONUS_FLOOR:
            bonus = max(0.0, BONUS_CAP - n * BONUS_SLOPE)
            avg_ir += bonus

        ema_list.append(GoalSpaceEMA(g, avg_ir))

    return ema_list

import random
from collections import defaultdict
from typing import Dict, List

from IMGEP_agent.agent.goals.goal_spaces import Goal
from IMGEP_agent.agent.knowledge_base import KnowledgeBase, RolloutRecord
from IMGEP_agent.hyper_parameters import GOAL_EMA_ALPHA


class GoalSpaceEMA:
    # Holds the EMA of the intrinsic reward for a goal space
    def __init__(self, goal: Goal, ema):
        self.goal: Goal = goal
        self.ema = ema


def select_goal_space(goal_space_emas: List[GoalSpaceEMA]) -> Goal:
    # 20% exploration
    # if random.random() > GOAL_GREEDY_PROB:
    #     return random.choice(goal_space_emas).goal


    # find max EMA
    max_ema = max(gs.ema for gs in goal_space_emas)
    candidates = [gs.goal for gs in goal_space_emas if gs.ema == max_ema]

    return random.choice(candidates)


def update_goal_space_ema(
        kb: KnowledgeBase,
        goal_spaces: List[Goal],
        alpha: float = GOAL_EMA_ALPHA
) -> List[GoalSpaceEMA]:
    # Pull N most-recent records once
    recent_records: List[RolloutRecord] = kb.nearest(100, exploit=True)

    # Bucket them by goal_space_id → [records...]
    records_by_goal: Dict[int, List[RolloutRecord]] = defaultdict(list)
    for r in recent_records:
        records_by_goal[r.goal_space_id].append(r)

    result: List[GoalSpaceEMA] = []

    for goal in goal_spaces:
        records = records_by_goal.get(goal.goal_id, [])
        if records:
            avg_reward = sum(rec.intrinsic_reward for rec in records) / len(records)
        else:
            avg_reward = 0.0
        result.append(GoalSpaceEMA(goal, avg_reward))

    return result

import random
from typing import List

from IMGEP_agent.agent.goals.goal_spaces import Goal
from IMGEP_agent.agent.knowledge_base import KnowledgeBase
from IMGEP_agent.hyper_parameters import GOAL_EMA_ALPHA, GOAL_EMA_K, GOAL_GREEDY_PROB


def goal_intrinsic_ema(kb: KnowledgeBase,
                       goal: Goal,
                       k: int = GOAL_EMA_K,
                       alpha: float = GOAL_EMA_ALPHA
                       ) -> float:
    recs = kb.nearest(goal, k, exploit=True) or []
    if not recs:
        return 0.0
    recs = list(reversed(recs))
    ema = recs[0].intrinsic_reward
    for rec in recs[1:]:
        ema = alpha * rec.intrinsic_reward + (1 - alpha) * ema
    return ema


def select_goal_space(all_goal_spaces: List[Goal],
                      kb: KnowledgeBase
                      ) -> Goal:
    # 20% exploration
    if random.random() > GOAL_GREEDY_PROB:
        return random.choice(all_goal_spaces)

    # compute EMAs
    emas = {
        gs: goal_intrinsic_ema(kb, gs)
        for gs in all_goal_spaces
    }
    # find max EMA
    max_ema = max(emas.values(), default=0.0)
    # tie-break uniformly among all that equal max_ema
    candidates = [gs for gs, e in emas.items() if e == max_ema]
    return random.choice(candidates)

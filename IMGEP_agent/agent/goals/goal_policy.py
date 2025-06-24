import random
from collections import defaultdict
from typing import Dict, List, Tuple

from IMGEP_agent.agent.goals.goal_spaces import Goal, get_goal_by_goal_id
from IMGEP_agent.agent.knowledge_base import KnowledgeBase, RolloutRecord
from IMGEP_agent.hyper_parameters import AgentConfig, IR_BONUS_CAP, IR_BONUS_SLOPE

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
        own_goal_space_emas: List[GoalSpaceEMA],
        other_goal_space_emas: List[GoalSpaceEMA],
        other_kb: KnowledgeBase,
        goal_spaces: List[Goal],
        own_kb: KnowledgeBase,
        config: AgentConfig,
        exploit: bool = False,
) -> Tuple[Goal, bool]:
    """
    Pick a goal either for ourselves or to accommodate the partner,
    based on EMA of intrinsic rewards and recent KB mappings.
    """
    own_top = _top_ema_candidates(own_goal_space_emas, EPSILON)
    other_top = _top_ema_candidates(other_goal_space_emas, EPSILON)

    if not own_top:
        raise ValueError("select_goal() called with no own_goal_space_emas")

    # Whose motivation is stronger?
    own_max = own_top[0].ema
    other_max = other_top[0].ema if other_top else float("-inf")
    our_turn = (own_max + EPSILON) >= other_max

    # If it's our turn (or partner has no goals), pick one of our top EMAs
    if our_turn or not other_top:
        candidates = [gs for gs in own_top if abs(gs.ema - own_max) < EPSILON]
        chosen = random.choice(candidates)
        return get_goal_by_goal_id(chosen.goal.goal_id, goal_spaces), exploit

    # Otherwise, try to accommodate partner
    partner_gs = random.choice(other_top)
    partner_goal = partner_gs.goal

    # Look up their most recent mapped partner-goal
    lookups = [
        other_kb.nearest(1, goal=partner_goal, exploit=True),
        other_kb.nearest(1, goal=partner_goal, exploit=False),
    ]
    other_recent = next((res for res in lookups if res), [])
    if other_recent:
        mapped_id = other_recent[0].partner_goal_id
        mapped_goal = get_goal_by_goal_id(mapped_id, goal_spaces)
        ours = own_kb.nearest(10, goal=mapped_goal, exploit=True)
        if ours and max(r.fitness for r in ours) > 0.5:
            return mapped_goal, True

    # Fallback: revert to one of our top
    fallback = random.choice(own_top)
    return get_goal_by_goal_id(fallback.goal.goal_id, goal_spaces), exploit


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

from collections import defaultdict
from typing import Dict

from IMGEP_agent.agent.goals.goal_spaces import Goal, get_goal_by_goal_id
from IMGEP_agent.agent.knowledge_base import KnowledgeBase, RolloutRecord
from IMGEP_agent.hyper_parameters import N_RECENT


class GoalSpaceEMA:
    # Holds the EMA of the intrinsic reward for a goal space
    def __init__(self, goal: Goal, ema):
        self.goal: Goal = goal
        self.ema = ema


import random
from typing import List

EPSILON = 1e-6  # default numeric tolerance


def _top_ema_candidates(gs_emas: List["GoalSpaceEMA"], eps: float = EPSILON) -> list["GoalSpaceEMA"]:
    """Return ALL goal-space EMA objects whose EMA is (max ± eps)."""
    if not gs_emas:
        return []
    m = max(gs.ema for gs in gs_emas)
    return [gs for gs in gs_emas if abs(gs.ema - m) < eps]


def select_goal(
        own_goal_space_emas: List["GoalSpaceEMA"],
        other_goal_space_emas: List["GoalSpaceEMA"],
        other_kb: "KnowledgeBase",
        goal_spaces: List["Goal"],
        own_kb: "KnowledgeBase",
        eps: float = EPSILON,
        exploit: bool = False,
) -> tuple[Goal, bool | None]:
    """
    Pick a goal for *this* agent to pursue.

    1. If we are at least as motivated (EMA ≥ other EMA within `eps`), choose
       randomly among our own top-EMA goal spaces.
    2. Otherwise (the other agent is more motivated), try to accommodate:
       a. Randomly pick one of *their* top-EMA goal spaces.
       b. Ask the other agent’s KB for the most recent partner‐goal mapping.
          First with `exploit=True`; if that returns nothing, fall back to
          `exploit=False`.
       c. Map the returned `partner_goal_id` back to an actual `Goal` object.
       d. If that mapping is missing or the KB had no memory at all, fall back
          to our own top-EMA goal spaces (rule 1).

    The function never raises and always returns *some* Goal as long as
    `own_goal_space_emas` is non-empty.
    """
    # ------------------------------------------------------------------ #
    # 1) Compute “top of chart” candidate lists for both agents
    # ------------------------------------------------------------------ #
    own_top = _top_ema_candidates(own_goal_space_emas, eps)
    other_top = _top_ema_candidates(other_goal_space_emas, eps)

    # Guards: we assume *we* have at least one goal space; if not, the design
    # of the caller is broken.
    if not own_top:
        raise ValueError("select_goal() called with no own_goal_space_emas")

    # ------------------------------------------------------------------ #
    # 2) Decide whose motivation wins
    # ------------------------------------------------------------------ #
    own_max_ema = own_top[0].ema  # all same EMA ± eps
    other_max_ema = other_top[0].ema if other_top else float("-inf")

    our_turn = (own_max_ema + eps) >= other_max_ema  # >= within tolerance
    if our_turn or not other_top:
        # -------------------------------------------------------------- #
        # Rule 1: pick one of *our* top-EMA goal spaces at random
        # -------------------------------------------------------------- #
        if exploit:
            return random.choice(own_top).goal, True
        else:
            # If not exploit, return a goal using the EMA as a probability
            # distribution, so that more motivated goal spaces are more likely
            # to be selected.
            # Convert each EMA to a strictly-positive weight
            min_prob = 0.05
            temperature = 0.5  # controls the "sharpness" of the distribution
            weights = [(itm.ema + min_prob) ** temperature for itm in own_goal_space_emas]
            chosen_goal_space_ema = random.choices(own_goal_space_emas, weights=weights, k=1)[0]
            # Return the goal from the chosen goal space

            # random.choices does the normalized soft-max for you
            return get_goal_space_from_goal_id(chosen_goal_space_ema.goal.goal_id, goal_spaces), False


    # ------------------------------------------------------------------ #
    # 3) Their turn — try to accommodate
    # ------------------------------------------------------------------ #
    other_goal_space = random.choice(other_top)
    other_goal = other_goal_space.goal

    # (a) Look for the most recent partner-goal mapping in their KB
    lookups = [
        other_kb.nearest(1, goal=other_goal, exploit=True),
        other_kb.nearest(1, goal=other_goal, exploit=False),
    ]
    other_kb_recent = next((res for res in lookups if res), [])  # first non-empty

    if other_kb_recent:
        partner_goal_id = other_kb_recent[0].partner_goal_id
        goal = get_goal_by_goal_id(partner_goal_id, goal_spaces)
        own_recent = own_kb.nearest(10, goal=goal, exploit=True)
        if max(r.fitness for r in own_recent) > 0.5:  # Only return a goal if it has a decent fitness
            return goal, True

    # ------------------------------------------------------------------ #
    # 4) Fallback — couldn’t map or KB was empty; revert to our own top
    # ------------------------------------------------------------------ #
    own_goal = random.choice(own_goal_space_emas)
    return get_goal_by_goal_id(own_goal.goal.goal_id, goal_spaces), True


def get_goal_space_from_goal_id(goal_id: int, goal_spaces: List[Goal]) -> Goal:
    """
    Get the GoalSpace object from a goal_id.

    Args:
        goal_id (int): The ID of the goal.
        goal_spaces (List[Goal]): List of all available GoalSpace objects.

    Returns:
        Goal: The GoalSpace object corresponding to the given goal_id.
    """
    for goal_space in goal_spaces:
        if goal_space.goal_id == goal_id:
            return goal_space
    raise ValueError(f"Goal with ID {goal_id} not found in provided goal spaces.")

def update_goal_space_ema(
        kb: KnowledgeBase,
        goal_spaces: List[Goal],
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

        ema_list.append(GoalSpaceEMA(g, avg_ir))

    return ema_list

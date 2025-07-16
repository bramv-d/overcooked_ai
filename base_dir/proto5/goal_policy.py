# -------------------- helpers -------------------- #
import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from base_dir.hyper_parameters import AgentConfig
from base_dir.proto5.knowledge_base import KnowledgeBase, RolloutRecord
from base_dir.shared_files.goal_spaces import Goal, get_goal_by_goal_id
from base_dir.shared_files.helpers.softmax_sampler import GoalSpaceEMA

EPSILON = 1e-6  # numeric tolerance
DEFAULT_TAU = 0.3  # soft-max temperature
DEFAULT_EPS = 0.10  # ε for ε-greedy
DEFAULT_FIT_WIN = 20  # recent roll-outs to average


def _top_ema_candidates(gs_emas: List["GoalSpaceEMA"], eps: float = EPSILON) -> List["GoalSpaceEMA"]:
    """Return all GoalSpaceEMA objects whose EMA is within *eps* of the maximum."""
    if not gs_emas:
        return []
    m = max(gs.ema for gs in gs_emas)
    return [gs for gs in gs_emas if abs(gs.ema - m) < eps]


def _epsilon_softmax_sample(goals: List["GoalSpaceEMA"],
                            tau: float,
                            eps: float) -> "GoalSpaceEMA":
    """
    ε-greedy soft-max:
        • with prob ε → uniform random goal
        • else        → soft-max on EMA scores
    Guarantees every goal ≥ ε / |goals| probability.
    """
    if random.random() < eps:  # pure exploration branch
        return random.choice(goals)

    logits = np.asarray([g.ema for g in goals], dtype=np.float64)
    logits -= logits.max()  # numerical stability
    probs = np.exp(logits / tau)
    probs /= probs.sum()
    return goals[np.random.choice(len(goals), p=probs)]


def _recent_mean_fitness(kb: "KnowledgeBase",
                         goal: "Goal",
                         k: int) -> float:
    """
    Average fitness over the last *k* **exploit** roll-outs for a given goal.
    Returns 0.0 if the KB has no such records.
    """
    records = kb.nearest(k, goal=goal, exploit=True)
    if not records:
        return 0.0
    return sum(r.fitness for r in records) / len(records)


# -------------------- main entry -------------------- #
def select_goal(
        own_goal_space_emas: List["GoalSpaceEMA"],
        other_goal_space_emas: List["GoalSpaceEMA"],
        other_kb: "KnowledgeBase",
        goal_spaces: List["Goal"],
        own_kb: "KnowledgeBase",
        config: "AgentConfig",
        exploit: bool = False,
) -> Tuple["Goal", bool, bool]:
    """
    Decide which goal to pursue.

    Returns
    -------
    Tuple(goal, exploit_flag, our_turn_flag)
    """
    # ---------- 0) guards --------------------------------------------------- #
    if not own_goal_space_emas:
        raise ValueError("select_goal() called with empty own_goal_space_emas")

    tau = getattr(config, "softmax_temperature", DEFAULT_TAU)
    eps = getattr(config, "epsilon_explore", DEFAULT_EPS)
    fit_win = getattr(config, "fitness_window", DEFAULT_FIT_WIN)

    # ---------- 1) priority: whose turn? ------------------------------------ #
    own_max = max(gs.ema for gs in own_goal_space_emas)
    other_max = max((gs.ema for gs in other_goal_space_emas), default=-math.inf)

    # default rule: higher EMA keeps the turn
    our_turn = (own_max + EPSILON) >= other_max

    if not our_turn and other_goal_space_emas:
        # examine partner's **best** goal (deterministic, no random tie-break)
        partner_best = max(other_goal_space_emas, key=lambda g: g.ema)

        partner_fit = _recent_mean_fitness(other_kb, partner_best.goal, fit_win)

        # if partner under-performs on their best goal, keep control
        if partner_fit < config.minimum_goal_fitness:
            our_turn = True

    # ---------- 2) if it's OUR turn ---------------------------------------- #
    if our_turn or not other_goal_space_emas:
        if exploit:
            # greedy: uniform among near-best EMAs
            best = _top_ema_candidates(own_goal_space_emas, eps=EPSILON)
            chosen_gs = random.choice(best)
        else:
            # exploration: ε-greedy soft-max
            chosen_gs = _epsilon_softmax_sample(own_goal_space_emas, tau, eps)

        return get_goal_by_goal_id(chosen_gs.goal.goal_id, goal_spaces), exploit, True

    # ---------- 3) attempt to help partner --------------------------------- #
    partner_best = max(other_goal_space_emas, key=lambda g: g.ema)
    partner_goal = partner_best.goal

    # ask KB whether we have a useful mapping for that partner goal
    for exploit_flag in (True, False):  # try exploit, then explore
        recs = other_kb.nearest(1, goal=partner_goal, exploit=exploit_flag)
        if recs:
            mapped_id = recs[0].partner_goal_id
            mapped_goal = get_goal_by_goal_id(mapped_id, goal_spaces)

            ours = own_kb.nearest(10, goal=mapped_goal, exploit=True)
            if ours and max(r.fitness for r in ours) > config.minimum_goal_fitness:
                # good mapping found → help partner (always exploit)
                return mapped_goal, True, False

    # ---------- 4) fallback = pick our own goal ---------------------------- #
    if exploit:
        fallback = max(own_goal_space_emas, key=lambda g: g.ema)
    else:
        fallback = _epsilon_softmax_sample(own_goal_space_emas, tau, eps)

    return get_goal_by_goal_id(fallback.goal.goal_id, goal_spaces), exploit, True


# -------------------- EMA updater -------------------- #
def update_goal_space_ema(
        kb: "KnowledgeBase",
        goal_spaces: List["Goal"],
        config: "AgentConfig",
) -> List["GoalSpaceEMA"]:
    """
    Re-compute EMA of intrinsic reward *per goal space* from the latest
    *config.n_recent* exploit roll-outs.  No sparsity bonus is injected;
    exploration is handled entirely in the selector.
    """
    recent: List["RolloutRecord"] = kb.nearest(config.n_recent, exploit=True)

    # bucket by goal_id
    by_goal: Dict[int, List["RolloutRecord"]] = defaultdict(list)
    for r in recent:
        by_goal[r.goal_space_id].append(r)

    result: List["GoalSpaceEMA"] = []
    for g in goal_spaces:
        recs = by_goal.get(g.goal_id, [])
        if recs:
            avg_ir = sum(r.intrinsic_reward for r in recs) / len(recs)
        else:
            avg_ir = 0.0
        result.append(GoalSpaceEMA(g, avg_ir))

    return result

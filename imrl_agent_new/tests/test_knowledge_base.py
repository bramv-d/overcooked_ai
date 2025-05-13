# test_knowledge_base.py
"""
Sanity-check for the ExperimentRecord dataclass and the KnowledgeBase API
defined in knowledge_base.py.

Run with:  python test_knowledge_base.py
"""

import numpy as np
from imrl_agent_new.core.knowledge_base import KnowledgeBase, ExperimentRecord, RunningMean

# --------------------------------------------------------------------------- #
# Global constants so tests are reproducible
CONTEXT_DIM  = 10
GOAL_DIM     = 3
THETA_DIM    = 8
OUTCOME_DIM  = 5
RNG          = np.random.default_rng(42)

def rand_vec(dim: int):
    """Utility: random float vector in [0,1)."""
    return RNG.random(dim)

def make_record():
    """Create a dummy ExperimentRecord with random values."""
    return ExperimentRecord(
        context  = rand_vec(CONTEXT_DIM),
        goal     = rand_vec(GOAL_DIM),
        theta    = rand_vec(THETA_DIM),
        outcome  = rand_vec(OUTCOME_DIM),
        fitness  = float(RNG.random()),
        intrinsic_reward = float(RNG.random())
    )

# --------------------------------------------------------------------------- #
# Individual tests
def test_add_and_len():
    kb = KnowledgeBase(CONTEXT_DIM, OUTCOME_DIM)
    assert len(kb) == 0, "KB should be empty on init"

    rec = make_record()
    kb.add_record(rec)
    assert len(kb) == 1, "add_record should increase length"
    print("✓ test_add_and_len passed")

def test_nearest_returns_self_for_identical_point():
    kb = KnowledgeBase(CONTEXT_DIM, OUTCOME_DIM)
    rec = make_record()
    kb.add_record(rec)

    idx, dist = kb.nearest(rec.context, rec.outcome)
    # Because there is only one record, nearest must return index 0 and distance 0
    assert idx[0] == 0, "nearest should return index 0 for the only record"
    assert np.isclose(dist[0], 0.0), "distance to identical point should be 0"
    print("✓ test_nearest_returns_self_for_identical_point passed")

def test_nearest_picks_closest_of_two():
    kb = KnowledgeBase(CONTEXT_DIM, OUTCOME_DIM)

    # First record at all-zeros
    recA = ExperimentRecord(
        context  = np.zeros(CONTEXT_DIM),
        goal     = np.zeros(GOAL_DIM),
        theta    = rand_vec(THETA_DIM),
        outcome  = np.zeros(OUTCOME_DIM),
        fitness  = 0.0,
        intrinsic_reward = 0.0
    )
    # Second record at all-ones
    recB = ExperimentRecord(
        context  = np.ones(CONTEXT_DIM),
        goal     = np.ones(GOAL_DIM),
        theta    = rand_vec(THETA_DIM),
        outcome  = np.ones(OUTCOME_DIM),
        fitness  = 1.0,
        intrinsic_reward = 0.1
    )
    kb.add_record(recA)
    kb.add_record(recB)

    # Query near zeros -> should get recA (index 0)
    idx, _ = kb.nearest(np.zeros(CONTEXT_DIM), np.zeros(OUTCOME_DIM))
    assert idx[0] == 0, "nearest should return record A for zero query"

    # Query near ones  -> should get recB (index 1)
    idx, _ = kb.nearest(np.ones(CONTEXT_DIM), np.ones(OUTCOME_DIM))
    assert idx[0] == 1, "nearest should return record B for ones query"
    print("✓ test_nearest_picks_closest_of_two passed")

def test_update_stat_and_running_mean():
    kb = KnowledgeBase(CONTEXT_DIM, OUTCOME_DIM)
    values = [1.0, 3.0, 5.0]
    for v in values:
        kb.update_stat("dish-goal", v)

    mean_est = kb.stats["dish-goal"].mean
    assert np.isclose(mean_est, np.mean(values)), "RunningMean should track average"
    print("✓ test_update_stat_and_running_mean passed")
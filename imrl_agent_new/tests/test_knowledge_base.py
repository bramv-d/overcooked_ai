# test_knowledge_base.py
"""
Quick sanity-check for ExperimentRecord and KnowledgeBase.

Run:  python test_knowledge_base.py
"""
import numpy as np
from imrl_agent_new.core.knowledge_base import KnowledgeBase, ExperimentRecord
CONTEXT_DIM  = 10
GOAL_DIM     = 3      # only used inside the record
THETA_DIM    = 8
OUTCOME_DIM  = 5

def rand_vec(dim):            # helper
    return np.random.random(dim)

def make_record():
    return ExperimentRecord(
        context=rand_vec(CONTEXT_DIM),
        goal=rand_vec(GOAL_DIM),
        theta=rand_vec(THETA_DIM),
        outcome=rand_vec(OUTCOME_DIM),
        fitness=np.random.random(),
        intrinsic_reward=np.random.random()
    )

# ---------------------------------------------------------------------------
def test_add_and_len():
    kb = KnowledgeBase(context_dim=CONTEXT_DIM, outcome_dim=OUTCOME_DIM)
    assert len(kb) == 0

    kb.add_record(make_record())
    assert len(kb) == 1
    print("✓ test_add_and_len passed")

def test_last():
    kb = KnowledgeBase(context_dim=CONTEXT_DIM, outcome_dim=OUTCOME_DIM)
    rec1 = make_record()
    rec2 = make_record()
    kb.add_record(rec1); kb.add_record(rec2)
    assert kb.last() is rec2
    print("✓ test_last passed")

def test_nearest():
    kb = KnowledgeBase(context_dim=CONTEXT_DIM, outcome_dim=OUTCOME_DIM)

    # two clearly separated points
    recA = ExperimentRecord(np.zeros(CONTEXT_DIM), np.zeros(GOAL_DIM),
                            rand_vec(THETA_DIM), np.zeros(OUTCOME_DIM),
                            0.1, 0.01)
    recB = ExperimentRecord(np.ones (CONTEXT_DIM), np.ones (GOAL_DIM),
                            rand_vec(THETA_DIM), np.ones (OUTCOME_DIM),
                            0.2, 0.02)
    kb.add_record(recA); kb.add_record(recB)

    # query near A  -> should get recA
    idx, best = kb.nearest(np.zeros(CONTEXT_DIM), np.zeros(OUTCOME_DIM))
    assert best is recA
    print("✓ test_nearest passed")

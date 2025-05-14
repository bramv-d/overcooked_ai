# tests/test_goal_policy.py
import random
import numpy as np
import pytest

from imrl_agent_new.core.goal_policy import GoalSpacePolicy
from imrl_agent_new.core.goal_spaces import make_nav_space
from overcooked_ai_py.data.layouts.layouts import layouts
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


# ---------------------------------------------------------------------------
#  minimal stub for a GoalSpace (only .sample() needed for the bandit)
class DummySpace:
    def __init__(self, name):
        self.name = name
    def sample(self):
        return np.array([42], dtype=np.int8)    # any vector is fine

# two goal-spaces A and B
SPACES = {"A": DummySpace("A"), "B": DummySpace("B")}

# handy helper
def fixed_seed():
    random.seed(12345)
    np.random.seed(12345)

# ---------------------------------------------------------------------------
def test_next_goal_returns_valid_key():
    fixed_seed()
    Γ = GoalSpacePolicy(SPACES, epsilon=0.0)   # exploit path
    space_id, g = Γ.next_goal()
    assert space_id in SPACES
    # sample() should have returned [42]
    assert np.array_equal(g, np.array([42], dtype=np.int8))

# ---------------------------------------------------------------------------
def test_uniform_sampling_when_no_progress():
    fixed_seed()
    Γ = GoalSpacePolicy(SPACES, epsilon=0.0)   # exploit but LP==0
    counts = {"A": 0, "B": 0}
    for _ in range(10_000):
        space_id, _ = Γ.next_goal()
        counts[space_id] += 1
    ratio = counts["A"] / (counts["A"] + counts["B"])
    # With seed and uniform random we expect about 0.5 ± 0.05
    assert 0.45 < ratio < 0.55

# ---------------------------------------------------------------------------
def test_lp_bias_after_update():
    fixed_seed()
    Γ = GoalSpacePolicy(SPACES, epsilon=0.0, lp_alpha=0.1)

    # Give space 'A' a big positive learning-progress
    Γ.update("A", 1.0)     # avg_lp['A'] = 0.1
    Γ.update("A", 1.0)     # avg_lp['A'] ≈ 0.19
    # space 'B' stays at 0

    counts = {"A": 0, "B": 0}
    for _ in range(5_000):
        sid, _ = Γ.next_goal()
        counts[sid] += 1

    assert counts["A"] > counts["B"] * 4       # strong bias toward 'A'

# ---------------------------------------------------------------------------
def test_exponential_moving_average():
    fixed_seed()
    Γ = GoalSpacePolicy(SPACES, lp_alpha=0.5)  # easy math

    Γ.update("A", 1.0)     # avg = 0.5
    assert pytest.approx(Γ.avg_lp["A"]) == 0.5

    Γ.update("A", 1.0)     # 0.5*(1-0.5)+1*0.5 = 0.75
    assert pytest.approx(Γ.avg_lp["A"]) == 0.75

    Γ.update("A", 0.0)     # 0.75*(1-0.5)+0*0.5 = 0.375
    assert pytest.approx(Γ.avg_lp["A"]) == 0.375


def test_policy_with_real_nav_space():
    fixed_seed()
    grid = OvercookedGridworld.from_layout_name(layouts[20])
    nav  = make_nav_space(grid_world=grid, length_of_trajectory=400)

    G = {"nav": nav}
    Γ = GoalSpacePolicy(G, epsilon=0.0)    # exploit path
    sid, g = Γ.next_goal()

    # sid should be 'nav', g should be a 2-vector that is one of the walkables
    assert sid == "nav"
    assert tuple(g) in grid.get_valid_player_positions()

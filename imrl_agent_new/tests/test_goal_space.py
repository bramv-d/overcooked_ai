# tests/test_goal_spaces.py
import numpy as np
import pytest

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.data.layouts.layouts import layouts


# ----------------- light-weight stubs ----------------------------------------

class DummyGrid:
    """Just enough for make_nav_space."""
    def __init__(self, walkable):
        # walkable = list[(x,y)]
        self._walkable = walkable
    def get_valid_player_positions(self):
        return self._walkable

class DummyPos:
    def __init__(self, x, y): self.x, self.y = x, y
class DummyPlayer:
    def __init__(self, x, y, held=None):
        self.pos = DummyPos(x, y)
        self._held = held
    def has_object(self):       return self._held is not None
    def get_object(self):       return self._held

# ----------------- import the spaces under test ------------------------------

from imrl_agent_new.core.goal_spaces import (
    make_nav_space,
    make_pick_object_space,
    ItemCode
)

# ----------------- constants -------------------------------------------------
HORIZON = 100
GRID    = OvercookedGridworld.from_layout_name(layouts[20])

# ----------------- tests for NAV ---------------------------------------------

def test_nav_sample_within_walkable():
    nav = make_nav_space(GRID, HORIZON)
    g = nav.sample()
    assert tuple(g) in GRID.get_valid_player_positions()

def test_nav_success_and_fitness():
    nav = make_nav_space(GRID, HORIZON)
    g   = np.array([1, 2])               # choose a known tile

    # player on the goal tile -> success True
    player = DummyPlayer(1, 2)
    assert nav.success(player, g)

    # fitness should be highest when reach_step = 0
    f0 = nav.fitness(np.array([1,2]), g, reach_step=0)
    assert pytest.approx(f0) == 1.0

    # and lower when reach_step is later
    f20 = nav.fitness(np.array([1,2]), g, reach_step=20)
    assert f20 < f0 and pytest.approx(f20) == 1 - 20/HORIZON

    # never reached -> 0 fitness
    f_none = nav.fitness(np.array([0,0]), g, reach_step=None)
    assert f_none == 0.0

# ----------------- tests for PICK_OBJECT -------------------------------------

def test_pick_object_sampler_and_success():
    pick = make_pick_object_space(HORIZON)
    g    = pick.sample()
    target_code = int(g[0])

    # build player holding that object
    held_name   = ItemCode(target_code).name.lower()
    player      = DummyPlayer(0, 0, held=held_name)

    assert pick.success(player, g)

def test_pick_object_time_shaped_fitness():
    pick = make_pick_object_space(HORIZON)
    g         = np.array([ItemCode.BOWL], dtype=np.int8)

    # outcome slot 1 must equal held item code
    o = np.array([0, ItemCode.BOWL], dtype=np.float32)

    f10 = pick.fitness(o, g, pick_step=10)
    f90 = pick.fitness(o, g, pick_step=90)

    assert f10 > f90                        # faster pickup ⇒ higher reward
    assert pytest.approx(f10) == 1 - 10/HORIZON
    assert pytest.approx(f90) == 1 - 90/HORIZON

    # wrong item -> zero
    o_wrong = np.array([0, ItemCode.ONION], dtype=np.float32)
    assert pick.fitness(o_wrong, g, pick_step=10) == 0.0

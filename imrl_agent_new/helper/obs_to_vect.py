from enum import IntEnum

import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, OvercookedGridworld, OvercookedState, PlayerState
from overcooked_ai_py.planning.planners import MotionPlanner


class ItemCode(IntEnum):
    """Compact integer codes for objects the chef can hold or deliver."""
    NOTHING = 0
    ONION = 1
    TOMATO = 2
    DISH = 3
    SOUP = 4


def item_to_int(item_name: ObjectState) -> int:
    """
    Map an Overcooked item string (or None) to a compact int code.
    Unknown items default to 0 (NOTHING).
    """
    if item_name is None:
        return ItemCode.NOTHING.value
    try:
        return ItemCode[(item_name.name.upper())].value
    except KeyError:
        return ItemCode.NOTHING.value


def int_to_item(code: int) -> str:
    """Reverse lookup: from int to canonical item name."""
    try:
        return ItemCode(code).name.lower()
    except ValueError:
        return "unknown"

def is_reachable(mp: MotionPlanner, pos_and_or, target_locations) -> float:
    """Return 1.0 if any target location is reachable, else 0.0."""
    return 1.0 if mp.min_cost_to_feature(pos_and_or, target_locations) != np.inf else 0.0

OBS_VEC_SIZE = len(ItemCode) + 8

def obs_to_vec(
        state: OvercookedState,
        mdp: OvercookedGridworld,
        mp: MotionPlanner,
        player_id: int,
) -> np.ndarray:
    me: PlayerState = state.players[player_id]
    pot_states = mdp.get_pot_states(state)

    # Held object one-hot
    held = item_to_int(me.get_object()) if me.has_object() else 0
    held_onehot = np.eye(len(ItemCode), dtype=np.float32)[held]  # shape (5,)

    # Binary reachable features
    d_onion = is_reachable(mp, me.pos_and_or, mdp.get_onion_dispenser_locations())
    d_tomato = is_reachable(mp, me.pos_and_or, mdp.get_tomato_dispenser_locations())
    d_pots_cooking = is_reachable(mp, me.pos_and_or, mdp.get_cooking_pots(pot_states))
    d_empty_pots = is_reachable(mp, me.pos_and_or, mdp.get_empty_pots(pot_states))
    d_ready_pots = is_reachable(mp, me.pos_and_or, mdp.get_ready_pots(pot_states))
    d_full_not_cooking_pots = is_reachable(mp, me.pos_and_or, mdp.get_full_but_not_cooking_pots(pot_states))
    d_counter = is_reachable(mp, me.pos_and_or, mdp.get_empty_counter_locations(state))
    d_serving = is_reachable(mp, me.pos_and_or, mdp.get_serving_locations())

    # Assemble feature vector
    feat = np.array([
        *held_onehot,  # (5,)
        d_onion,  # (1,)
        d_tomato,
        d_pots_cooking,
        d_empty_pots,
        d_full_not_cooking_pots,
        d_ready_pots,
        d_counter,
        d_serving,
    ], dtype=np.float32)

    return feat  # shape = (5 + 8,) = (13,)

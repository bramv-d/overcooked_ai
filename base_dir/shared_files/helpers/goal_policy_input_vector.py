import numpy as np

from base_dir.shared_files.helpers.item_codes import ItemCode
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState
from overcooked_ai_py.planning.planners import MotionPlanner


def is_reachable(mp: MotionPlanner, pos_and_or, target_locations) -> float:
    """Return 1.0 if any target location is reachable, else 0.0."""
    return 1.0 if mp.min_cost_to_feature(pos_and_or, target_locations) != np.inf else 0.0


def get_goal_policy_input_vector(
        state: OvercookedState,
        mdp: OvercookedGridworld,
        mp: MotionPlanner,
        player_id: int,
) -> np.ndarray:
    me: PlayerState = state.players[player_id]
    other: PlayerState = state.players[1 - player_id]
    pot_states = mdp.get_pot_states(state)

    # Held object one-hot
    held = ItemCode.ItemCodeValue(me.get_object().name) if me.has_object() else ItemCode.NOTHING
    held_onehot = np.eye(len(ItemCode), dtype=np.float32)[held]  # shape (5,)

    counter_objects = mdp.get_counter_objects_dict(state)

    # Binary reachable features
    d_onion = is_reachable(mp, me.pos_and_or, mdp.get_onion_dispenser_locations())
    d_onion_counter = is_reachable(mp, me.pos_and_or, counter_objects["onion"])
    d_dish_pickup = is_reachable(mp, me.pos_and_or, mdp.get_dish_dispenser_locations())
    d_dish_counter = is_reachable(mp, me.pos_and_or, counter_objects["dish"])
    d_pots_cooking = is_reachable(mp, me.pos_and_or, mdp.get_cooking_pots(pot_states))
    d_full_not_cooking_pots = is_reachable(mp, me.pos_and_or, mdp.get_full_but_not_cooking_pots(pot_states))
    d_empty_pots = is_reachable(mp, me.pos_and_or, mdp.get_empty_pots(pot_states))
    d_partly_full_pots = is_reachable(mp, me.pos_and_or, mdp.get_partially_full_pots(pot_states))
    d_ready_pots = is_reachable(mp, me.pos_and_or, mdp.get_ready_pots(pot_states))

    # Assemble feature vector
    feat = np.array([
        *held_onehot,  # (5,# )
        d_dish_pickup,
        d_onion_counter,
        d_dish_counter,
        d_onion,
        d_partly_full_pots,
        d_pots_cooking,
        d_empty_pots,
        d_full_not_cooking_pots,
        d_ready_pots,
    ], dtype=np.float32)

    return feat


GOAL_POLICY_INPUT_VECTOR_SIZE = len(ItemCode) + 9

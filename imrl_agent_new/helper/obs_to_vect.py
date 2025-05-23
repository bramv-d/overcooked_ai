import numpy as np

from imrl_agent_new.overcooked.outcome import ItemCode, item_to_int
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState
from overcooked_ai_py.planning.planners import MotionPlanner


def obs_to_vec(
        state: OvercookedState,
        mdp: OvercookedGridworld,
        mp: MotionPlanner,
        player_id: int,
        diam: int  # longest path in the layout, used for normalization
) -> np.ndarray:
    w, h = mdp.width, mdp.height
    me: PlayerState = state.players[player_id]
    px, py = me.position
    held = item_to_int(me.get_object()) if me.has_object() else 0
    held_onehot = np.eye(len(ItemCode), dtype=np.float32)[held]  # shape (5,)
    pot_states = mdp.get_pot_states(state)
    # ----------- pre-computed distances (already integers) ------------------
    # Could include things like "is_potting_useful" but not sure if that is nice
    d_onion = mp.min_cost_to_feature(me.pos_and_or, mdp.get_onion_dispenser_locations())
    d_tomato = mp.min_cost_to_feature(me.pos_and_or, mdp.get_tomato_dispenser_locations())
    d_pots_cooking = mp.min_cost_to_feature(me.pos_and_or, mdp.get_cooking_pots(pot_states))
    d_empty_pots = mp.min_cost_to_feature(me.pos_and_or, mdp.get_empty_pots(mdp.get_pot_states(state)))
    d_ready_pots = mp.min_cost_to_feature(me.pos_and_or, mdp.get_ready_pots(mdp.get_pot_states(state)))
    d_counter = mp.min_cost_to_feature(me.pos_and_or, mdp.get_empty_counter_locations(state))
    d_serving = mp.min_cost_to_feature(me.pos_and_or, mdp.get_serving_locations())

    # -------------- partner relative offset (normalised) --------------------
    partner = state.players[1 - player_id]
    rel_dx = np.clip(partner.position[0] - px, -(w - 1), w - 1) / (w - 1)  # ∈[-1,1]
    rel_dy = np.clip(partner.position[1] - py, -(h - 1), h - 1) / (h - 1)
    rel_dx = (rel_dx + 1) / 2  # shift to [0,1]
    rel_dy = (rel_dy + 1) / 2

    # -------------- assemble feature vector ---------------------------------
    feat = np.array([
        *held_onehot,  # 0-4 held object one-hot, unpack using *
        min(d_onion, diam) / diam,  # 5 dist to onion dispenser
        min(d_tomato, diam) / diam,  # 6 dist to tomato dispenser
        min(d_pots_cooking, diam) / diam,  # 7 dist to cooking pot
        min(d_empty_pots, diam) / diam,  # 8 dist to empty pot
        min(d_ready_pots, diam) / diam,  # 9 dist to ready pot
        min(d_counter, diam) / diam,  # 10 dist to empty counter
        min(d_serving, diam) / diam,  # 11 dist to serving/pass window
        rel_dx,  # 12 partner dx normalised
        rel_dy,  # 13 partner dy normalised
    ], dtype=np.float32)

    return feat  # shape = (14,)

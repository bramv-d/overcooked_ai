import numpy as np
from typing import List
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState


OUTCOME_DIM = 15  # must match the tests and goals.py


def outcome_from_trajectory(traj: List[OvercookedState]) -> np.ndarray:
    """
    Convert the *final* OvercookedState in `traj` into a 15‑dim vector:

      [p0_x, p0_y, p1_x, p1_y, holding0, holding1,
       pot0_contents, pot0_timer,
       pot1_contents, pot1_timer,
       pot2_contents, pot2_timer,
       dishes_served, time_elapsed, collisions]

    Only the fields we currently need are filled; the rest stay zero.
    """
    s_final = traj[-1]
    grid = s_final.mdp
    pots = grid.all_pot_indices  # e.g. [(5,3), (6,3), (7,3)]

    vec = np.zeros(OUTCOME_DIM, dtype=np.float32)

    # players ---------------------------------------------------------------
    vec[0], vec[1] = s_final.players[0].position
    vec[2], vec[3] = s_final.players[1].position
    vec[4] = s_final.players[0].held_object.name_as_int()  # 0‑n mapping
    vec[5] = s_final.players[1].held_object.name_as_int()

    # pots ------------------------------------------------------------------
    for pot_i, (x, y) in enumerate(pots[:3]):          # cap at 3 pots
        pot = grid.get_pot_at_loc_if_exists((x, y))
        base = 6 + 2 * pot_i
        if pot:
            vec[base] = pot.get_num_ingredients()
            vec[base + 1] = pot.cooking_tick if pot.is_cooking else 0

    # global stats ----------------------------------------------------------
    vec[12] = s_final.soup_served
    vec[13] = s_final.timestep
    vec[14] = s_final.collisions

    return vec

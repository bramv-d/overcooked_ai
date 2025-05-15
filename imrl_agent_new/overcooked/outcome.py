# outcome.py
import string

import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, OvercookedGridworld, OvercookedState, PlayerState


# outcome.py
import numpy as np

def extract_outcome(
    state: OvercookedState,
    player: PlayerState,
    grid_world: OvercookedGridworld,
    soups_delivered: int
) -> np.ndarray:
    """
    Returns a 4-dim outcome vector  φ(τ) = [x, y, inv_code, soups]  for now.
    Slot 3 (pot_state) will be added later.
    """

    # ----- 0-1 : agent (x, y) ------------------------------------------------
    x, y = player.position[0], player.position[1]        # ints 0 … width-1 / height-1

    # ----- 2   : held-object code -------------------------------------------
    inv_code = item_to_int(player.get_object()) if player.has_object() else 0

    # ----- 3   : placeholder for pot state (to be filled in later) ----------
    pot_code = 0                             # PotCode.EMPTY for now

    # ----- 4   : soups delivered --------------------------------------------
    soups = soups_delivered

    return np.array([x, y, inv_code, pot_code, soups], dtype=np.float32)

from enum import IntEnum

class ItemCode(IntEnum):
    """Compact integer codes for objects the chef can hold or deliver."""
    NOTHING   = 0
    ONION     = 1
    TOMATO    = 2
    BOWL      = 3
    SOUP      = 4

# ---------- helpers ----------------------------------------------------------
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
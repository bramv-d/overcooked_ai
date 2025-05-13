# outcome.py
import numpy as np

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState


def extract_outcome(state: OvercookedState, player: PlayerState, grid_world: OvercookedGridworld, soups_delivered) -> np.ndarray:
    """
    Build a 9-dim outcome vector from the final env state.
    The saved outcome of a rollout.
    """

    return np.array([
        player.pos_and_or,                                                              # Agent position
        item_to_int(player.get_object()) if player.has_object() else 0,                 # Agent object
        grid_world.get_pot_states(state),                                               # Numer of items in the pot //TODO Im unsure how to save the pot_state
        soups_delivered                                                                 # The number of correctly delivered soups
    ], dtype=np.float32)

from enum import IntEnum

class ItemCode(IntEnum):
    """Compact integer codes for objects the chef can hold or deliver."""
    NOTHING   = 0
    ONION     = 1
    TOMATO    = 2
    BOWL      = 3
    SOUP      = 4
    DIRTY_DISH= 5          # optional – add more if your env uses them

# ---------- helpers ----------------------------------------------------------
def item_to_int(item_name: str | None) -> int:
    """
    Map an Overcooked item string (or None) to a compact int code.
    Unknown items default to 0 (NOTHING).
    """
    if item_name is None:
        return ItemCode.NOTHING.value
    try:
        return ItemCode[item_name.upper()].value
    except KeyError:
        return ItemCode.NOTHING.value

def int_to_item(code: int) -> str:
    """Reverse lookup: from int to canonical item name."""
    try:
        return ItemCode(code).name.lower()
    except ValueError:
        return "unknown"
from enum import IntEnum

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.planning.planners import MediumLevelActionManager


class HighLevelActions(IntEnum):
    GO_ONION, GO_DISH, PUT_ONION, GO_READY_POT, GO_SERVE, START_COOKING, WAIT, ONION_FROM_DISPENSER, GO_SHARED_COUNTER = range(
        9)


def get_motion_goals(mlam: MediumLevelActionManager, mdp: OvercookedGridworld,
                     high_level_action: int, state: OvercookedState):
    all_counters = mdp.get_counter_locations()
    counter_objects = mdp.get_counter_objects_dict(state, all_counters)
    pots_object = mdp.get_pot_states(state)

    match high_level_action:
        case HighLevelActions.GO_ONION:
            return mlam.pickup_onion_actions(counter_objects)
        case HighLevelActions.ONION_FROM_DISPENSER:
            return mlam.pickup_onion_actions(counter_objects, only_use_dispensers=True)
        case HighLevelActions.GO_DISH:
            return mlam.pickup_dish_actions(counter_objects)
        case HighLevelActions.PUT_ONION:
            return mlam.put_onion_in_pot_actions(pots_object)
        case HighLevelActions.GO_READY_POT:
            return mlam.pickup_soup_with_dish_actions(pots_object)
        case HighLevelActions.GO_SERVE:
            return mlam.deliver_soup_actions()
        case HighLevelActions.START_COOKING:
            return mlam.start_cooking_actions(pots_object)
        case HighLevelActions.GO_SHARED_COUNTER:
            return mlam._get_ml_actions_for_positions(mdp.find_free_counters_valid_for_both_players(state, mlam))
        case HighLevelActions.WAIT:
            # If the agent is waiting, return an empty list to indicate no motion goals
            return []
    return None

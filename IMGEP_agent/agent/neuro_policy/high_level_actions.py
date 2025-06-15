from enum import IntEnum

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.planning.planners import MediumLevelActionManager


class HighLevelActions(IntEnum):
    GO_ONION, GO_TOMATO, GO_DISH, PUT_ONION, PUT_TOMATO, GO_READY_POT, GO_SERVE, START_COOKING, GO_COUNTER, WAIT = range(
        10)


def get_motion_goals(mlam: MediumLevelActionManager, mdp: OvercookedGridworld,
                     high_level_action: HighLevelActions, state: OvercookedState):
    all_counters = mdp.get_counter_locations()
    counter_objects = mdp.get_counter_objects_dict(state, all_counters)
    pots_object = mdp.get_pot_states(state)

    match high_level_action:
        case HighLevelActions.GO_ONION:
            return mlam.pickup_onion_actions(counter_objects)
        case HighLevelActions.GO_TOMATO:
            return mlam.pickup_tomato_actions(counter_objects)
        case HighLevelActions.GO_DISH:
            return mlam.pickup_dish_actions(counter_objects)
        case HighLevelActions.PUT_ONION:
            return mlam.put_onion_in_pot_actions(pots_object)
        case HighLevelActions.PUT_TOMATO:
            return mlam.put_tomato_in_pot_actions(pots_object)
        case HighLevelActions.GO_READY_POT:
            return mlam.pickup_soup_with_dish_actions(pots_object)
        case HighLevelActions.GO_SERVE:
            return mlam.deliver_soup_actions()
        case HighLevelActions.START_COOKING:
            return mlam.start_cooking_actions(pots_object)
        case HighLevelActions.GO_COUNTER:
            return mlam.place_obj_on_counter_actions(state)
        case HighLevelActions.WAIT:
            # If the agent is waiting, return an empty list to indicate no motion goals
            return []
    return None

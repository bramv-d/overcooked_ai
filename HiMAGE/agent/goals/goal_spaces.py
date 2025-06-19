from enum import IntEnum
from typing import List

from HiMAGE.agent.helpers.item_codes import ItemCode
from HiMAGE.hyper_parameters import HORIZON
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.layout_generator import COUNTER, POT, TYPE_TO_CODE
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState


class GoalSpaceEnum(IntEnum):
    OVERALL_GOAL = 0
    PLACE_ONION = 1
    START_COOKING = 2
    PICKUP_SOUP = 3
    DELIVER_SOUP = 4
    PICKUP_DISH = 5
    PICKUP_ONION = 6
    PLACE_ONION_IN_POT = 7

    @classmethod
    def get_goal_space_name(cls, value: int) -> str:
        return cls(value).name if value in cls._value2member_map_ else "UNKNOWN"

    @classmethod
    def get_goal_space_value(cls, name: str) -> int:
        return cls[name].value if name in cls._member_map_ else -1

class Goal:
    def __init__(self, goal_id: GoalSpaceEnum, fitness_fn, success_fn, reset):
        self.goal_id = goal_id
        self.fitness = fitness_fn
        self.success = success_fn
        self.reset = reset


def pick_object_space(goal_id: GoalSpaceEnum, object_code: int) -> Goal:
    performed_goal_amount = 0
    def fitness(pick_step: int, state: OvercookedState, previous_state: OvercookedState,
                agent_id: int, mdp: OvercookedGridworld, start_time):
        if pick_step is None:
            return 0.0
        time_taken = pick_step - start_time
        fitness_value = 1.0 - (time_taken / HORIZON)
        nonlocal performed_goal_amount
        if performed_goal_amount:
            fitness_value /= performed_goal_amount
        return fitness_value

    def success(agent_id: int, state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        player: PlayerState = state.players[agent_id]
        if not player.has_object():
            return False
        held_object = ItemCode.ItemCodeValue(player.held_object.name)
        nonlocal performed_goal_amount
        performed_goal_amount += 1
        return held_object == object_code

    def reset():
        nonlocal performed_goal_amount
        performed_goal_amount = 0

    return Goal(goal_id, fitness_fn=fitness, success_fn=success, reset=reset)


def place_object_space(goal_id: GoalSpaceEnum, object_code: int, terrain_type: int) -> Goal:
    performed_goal_amount = 0
    def fitness(pick_step: int, state: OvercookedState, previous_state: OvercookedState,
                agent_id: int, mdp: OvercookedGridworld, start_time):
        if pick_step is None:
            return 0.0
        time_taken = pick_step - start_time
        fitness_value = 1.0 - (time_taken / HORIZON)
        nonlocal performed_goal_amount
        if performed_goal_amount:
            fitness_value /= performed_goal_amount
        return fitness_value

    def success(agent_id: int, state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        prev_agent: PlayerState = previous_state.players[agent_id]
        curr_agent: PlayerState = state.players[agent_id]
        if curr_agent.has_object() or not prev_agent.has_object():
            return False
        held_prev = ItemCode.ItemCodeValue(prev_agent.held_object.name)
        if held_prev != object_code:
            return False
        facing_cell = Action.move_in_direction(prev_agent.position, prev_agent.orientation)
        terrain_type_code = mdp.get_terrain_type_at_pos(facing_cell)
        environment_type_code = TYPE_TO_CODE[terrain_type_code]
        nonlocal performed_goal_amount
        performed_goal_amount += 1
        return environment_type_code == terrain_type

    def reset():
        nonlocal performed_goal_amount
        performed_goal_amount = 0

    return Goal(goal_id, fitness_fn=fitness, success_fn=success, reset=reset)


def start_cooking_space() -> Goal:
    place_onion = place_object_space(GoalSpaceEnum.PLACE_ONION, ItemCode.ONION.value, TYPE_TO_CODE[POT])

    def fitness(pick_step: int, state: OvercookedState, previous_state: OvercookedState,
                agent_id: int, mdp: OvercookedGridworld, start_time):
        prev_pots = set(mdp.get_cooking_pots(mdp.get_pot_states(previous_state)))
        curr_pots = set(mdp.get_cooking_pots(mdp.get_pot_states(state)))
        new = curr_pots - prev_pots

        prev_agent = previous_state.players[agent_id]
        adj = mdp.get_adjacent_features(prev_agent)
        adj_positions = {pos for pos, _ in adj}
        if place_onion.success(agent_id, state, previous_state, mdp):
            return 0.1
        for p in new:
            if p in adj_positions:
                current_pot_state = state.get_object(p)
                return current_pot_state.value / 65 if current_pot_state else 0.0
        return 0.0

    def success(agent_id: int, state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        prev_pots = set(mdp.get_cooking_pots(mdp.get_pot_states(previous_state)))
        curr_pots = set(mdp.get_cooking_pots(mdp.get_pot_states(state)))
        new = curr_pots - prev_pots
        if not new:
            return False
        curr_agent = state.players[agent_id]
        adj = mdp.get_adjacent_features(curr_agent)
        adj_positions = {pos for pos, _ in adj}
        is_success = any(p in adj_positions for p in new)
        return is_success

    def reset():
        # Reset any specific state if needed
        pass

    return Goal(GoalSpaceEnum.START_COOKING, fitness_fn=fitness, success_fn=success, reset=reset)


def pickup_soup_space() -> Goal:
    start_cooking = start_cooking_space()
    pickup_dish = pick_object_space(GoalSpaceEnum.PICKUP_SOUP, ItemCode.DISH.value)

    def fitness(pick_step: int, state: OvercookedState, previous_state: OvercookedState,
                agent_id: int, mdp: OvercookedGridworld, start_time):
        player: PlayerState = state.players[agent_id]
        held = player.get_object() if player.has_object() else None

        if held and held.name == 'soup':
            return held.value / 65
        return 0.0

    def success(agent_id: int, state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        player: PlayerState = state.players[agent_id]
        held = player.get_object() if player.has_object() else None
        if held and held.name == 'soup':
            return True
        return False

    def reset():
        # Reset any specific state if needed
        pass

    return Goal(GoalSpaceEnum.PICKUP_SOUP, fitness_fn=fitness, success_fn=success, reset=reset)

def create_goal_spaces() -> List[Goal]:
    return [
        place_object_space(GoalSpaceEnum.PLACE_ONION, ItemCode.ONION.value, TYPE_TO_CODE[COUNTER]),
        place_object_space(GoalSpaceEnum.PLACE_ONION_IN_POT, ItemCode.ONION.value, TYPE_TO_CODE[POT]),
        start_cooking_space(),
        pickup_soup_space(),
        pick_object_space(GoalSpaceEnum.PICKUP_DISH, ItemCode.DISH.value),
        pick_object_space(GoalSpaceEnum.PICKUP_ONION, ItemCode.ONION.value),
    ]


def reset_goal_spaces(goal_spaces: List[Goal]) -> None:
    for goal in goal_spaces:
        goal.reset()

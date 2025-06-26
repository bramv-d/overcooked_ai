from enum import IntEnum
from typing import List

from IMGEP_agent.agent.helpers.item_codes import ItemCode
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.layout_generator import COUNTER, POT, TYPE_TO_CODE
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState


class GoalSpaceEnum(IntEnum):
    SHARE_ONION = 1
    START_COOKING = 2
    PICKUP_SOUP = 3
    PICKUP_ONION = 4
    PLACE_IN_POT = 5

    @classmethod
    def get_goal_space_name(cls, value: int) -> str:
        return cls(value).name if value in cls._value2member_map_ else "UNKNOWN"

    @classmethod
    def get_goal_space_value(cls, name: str) -> int:
        return cls[name].value if name in cls._member_map_ else -1


class Goal:
    def __init__(self, goal_id: GoalSpaceEnum, fitness_fn, success_fn=None, reset=None):
        self.goal_id = goal_id
        self.fitness = fitness_fn
        self.success = success_fn
        self.reset = reset


def pick_object_space(goal_id: GoalSpaceEnum, object_code: int) -> Goal:
    def fitness(state: OvercookedState, previous_state: OvercookedState, mdp: OvercookedGridworld, pick_step):

        if pick_step is None:
            return 0.0
        return 1.0

    def success(agent_id: int, state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        player: PlayerState = state.players[agent_id]
        if not player.has_object():
            return False
        held_object = ItemCode.ItemCodeValue(player.held_object.name)

        return held_object == object_code

    return Goal(goal_id, fitness_fn=fitness, success_fn=success)


def place_object_space(goal_id: GoalSpaceEnum, object_code: int, terrain_type: int) -> Goal:
    def fitness(state: OvercookedState, previous_state: OvercookedState, mdp: OvercookedGridworld, pick_step):
        if pick_step is None:
            return 0.0
        return 1.0

    def success(state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        for agent_id, agent in enumerate(state.players):
            prev_agent: PlayerState = previous_state.players[agent_id]
            curr_agent: PlayerState = state.players[agent_id]
            if curr_agent.has_object() or not prev_agent.has_object():
                continue
            held_prev = ItemCode.ItemCodeValue(prev_agent.held_object.name)
            if held_prev != object_code:
                continue
            facing_cell = Action.move_in_direction(prev_agent.position, prev_agent.orientation)
            terrain_type_code = mdp.get_terrain_type_at_pos(facing_cell)
            environment_type_code = TYPE_TO_CODE[terrain_type_code]
            if environment_type_code == terrain_type:
                return True
        return False

    return Goal(goal_id, fitness_fn=fitness, success_fn=success)


def start_cooking_space() -> Goal:
    place_onion_in_pot = place_object_space(GoalSpaceEnum.SHARE_ONION, ItemCode.ONION.value, TYPE_TO_CODE[POT])
    place_onion_on_counter = place_object_space(GoalSpaceEnum.SHARE_ONION, ItemCode.ONION.value, TYPE_TO_CODE[COUNTER])
    onions_placed = 0

    def fitness(state: OvercookedState, previous_state: OvercookedState, mdp: OvercookedGridworld, pick_step):
        prev_pots = set(mdp.get_cooking_pots(mdp.get_pot_states(previous_state)))
        curr_pots = set(mdp.get_cooking_pots(mdp.get_pot_states(state)))
        new = curr_pots - prev_pots
        total_fitness = 0.0
        nonlocal onions_placed
        if place_onion_in_pot.success(state, previous_state, mdp):
            if onions_placed < 3:
                total_fitness += 0.2
                onions_placed += 1
            else:
                total_fitness -= 0.01
        if place_onion_on_counter.success(state, previous_state, mdp):
            if onions_placed < 3:
                total_fitness += 0.05
                onions_placed += 1
            else:
                total_fitness -= 0.01
        for p in new:
            current_pot_state = state.get_object(p)
            total_fitness += current_pot_state.value / 65 if current_pot_state else 0.0
        return total_fitness

    def success(state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        curr_pots = set(mdp.get_cooking_pots(mdp.get_pot_states(state)))
        if not curr_pots:
            return False
        prev_pots = set(mdp.get_cooking_pots(mdp.get_pot_states(previous_state)))
        new = curr_pots - prev_pots
        if not new:
            return False
        return True

    def reset():
        nonlocal onions_placed
        onions_placed = 0

    return Goal(GoalSpaceEnum.START_COOKING, fitness_fn=fitness, success_fn=success, reset=reset)


def pickup_soup_space() -> Goal:
    start_cooking = start_cooking_space()

    def fitness(state: OvercookedState, previous_state: OvercookedState, mdp: OvercookedGridworld, pick_step):
        curr_pots = set(mdp.get_cooking_pots(mdp.get_pot_states(state)))
        if not curr_pots:
            cooking_fitness = start_cooking.fitness(state, previous_state, mdp, pick_step)
            return cooking_fitness / 2
        for agent_id, agent in enumerate(state.players):
            curr_agent: PlayerState = state.players[agent_id]
            held = curr_agent.get_object() if curr_agent.has_object() else None
            if held and held.name == 'soup':
                return held.value / 65
        return 0.0

    def success(state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        for agent_id, agent in enumerate(state.players):
            curr_agent: PlayerState = state.players[agent_id]
            held = curr_agent.get_object() if curr_agent.has_object() else None
            if held and held.name == 'soup':
                return True
        return False


    def reset():
        nonlocal start_cooking
        start_cooking.reset()

    return Goal(GoalSpaceEnum.PICKUP_SOUP, fitness_fn=fitness, success_fn=success, reset=reset)

def create_goal_spaces() -> List[Goal]:
    return [
        start_cooking_space(),
        # pickup_soup_space(),
        place_object_space(GoalSpaceEnum.SHARE_ONION, ItemCode.ONION.value, TYPE_TO_CODE[COUNTER]),
        # place_object_space(GoalSpaceEnum.PLACE_IN_POT, ItemCode.ONION.value, TYPE_TO_CODE[POT]),
    ]

def reset_goal_spaces(goal_spaces: List[Goal]):
    """
    Reset the goal spaces to their initial state.
    """
    for goal in goal_spaces:
        if goal.reset:
            goal.reset()


def get_goal_by_goal_id(goal_id: int, goal_spaces: List[Goal]) -> Goal:
    """
    Get a goal by its ID from the list of goal spaces.
    """
    for goal in goal_spaces:
        if goal.goal_id == goal_id:
            return goal
    raise ValueError(f"Goal with ID {goal_id} not found in the provided goal spaces.")

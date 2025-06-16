from enum import IntEnum
from typing import List

from IMGEP_agent.agent.helpers.item_codes import ItemCode
from IMGEP_agent.hyper_parameters import HORIZON
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState


class GoalSpaceEnum(IntEnum):
    PLACE_ONION = 1
    START_COOKING = 2
    PICKUP_SOUP = 3

    @classmethod
    def get_goal_space_name(cls, value: int) -> str:
        return cls(value).name if value in cls._value2member_map_ else "UNKNOWN"

    @classmethod
    def get_goal_space_value(cls, name: str) -> int:
        return cls[name].value if name in cls._member_map_ else -1


class Goal:
    def __init__(self, goal_id: GoalSpaceEnum, fitness_fn, success_fn=None):
        self.goal_id = goal_id
        self.fitness = fitness_fn
        self.success = success_fn


def pick_object_space(goal_id: GoalSpaceEnum, object_code: int) -> Goal:
    def fitness(pick_step: int, state: OvercookedState, previous_state: OvercookedState,
                agent_id: int, mdp: OvercookedGridworld):
        if pick_step is None:
            return 0.0
        return 1.0 - (pick_step / max(HORIZON, 1))

    def success(agent_id: int, state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        player: PlayerState = state.players[agent_id]
        if not player.has_object():
            return False
        held_object = ItemCode.ItemCodeValue(player.held_object.name)

        return held_object == object_code

    return Goal(goal_id, fitness_fn=fitness, success_fn=success)


def place_object_space(goal_id: GoalSpaceEnum, object_code: int) -> Goal:
    def fitness(pick_step: int, state: OvercookedState, previous_state: OvercookedState,
                agent_id: int, mdp: OvercookedGridworld):
        if pick_step is None:
            return 0.0
        return 1.0 - (pick_step / max(HORIZON, 1))

    def success(agent_id: int, state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        prev_agent: PlayerState = previous_state.players[agent_id]
        curr_agent: PlayerState = state.players[agent_id]
        if curr_agent.has_object() or not prev_agent.has_object():
            return False
        held_prev = ItemCode.ItemCodeValue(prev_agent.held_object.name)
        if held_prev != object_code:
            return False

        # check adjacency
        adjacent = mdp.get_adjacent_features(curr_agent)
        adj_positions = {pos for pos, _ in adjacent}

        for pot in mdp.get_pot_locations():
            prev_obj = previous_state.get_object(pot) if previous_state.has_object(pot) else None
            curr_obj = state.get_object(pot) if state.has_object(pot) else None
            if pot in adj_positions and prev_obj != curr_obj:
                return True
        return False

    return Goal(goal_id, fitness_fn=fitness, success_fn=success)


def start_cooking_space() -> Goal:
    place_onion = place_object_space(GoalSpaceEnum.PLACE_ONION, ItemCode.ONION.value)
    def fitness(pick_step: int, state: OvercookedState, previous_state: OvercookedState,
                agent_id: int, mdp: OvercookedGridworld):
        prev_pots = set(mdp.get_cooking_pots(mdp.get_pot_states(previous_state)))
        curr_pots = set(mdp.get_cooking_pots(mdp.get_pot_states(state)))
        new = curr_pots - prev_pots

        prev_agent = previous_state.players[agent_id]
        adj = mdp.get_adjacent_features(prev_agent)
        adj_positions = {pos for pos, _ in adj}
        if place_onion.success(agent_id, state, previous_state, mdp):
            place_onion_fitness = place_onion.fitness(pick_step, state, previous_state, agent_id, mdp)
            return place_onion_fitness / 4
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

    return Goal(GoalSpaceEnum.START_COOKING, fitness_fn=fitness, success_fn=success)


def pickup_soup_space() -> Goal:
    start_cooking = start_cooking_space()
    def fitness(pick_step: int, state: OvercookedState, previous_state: OvercookedState,
                agent_id: int, mdp: OvercookedGridworld):
        player: PlayerState = state.players[agent_id]
        previous_player: PlayerState = previous_state.players[agent_id]
        held = player.get_object() if player.has_object() else None

        if held and held.name == 'soup':
            return held.value / 65

        if start_cooking.success(agent_id, state, previous_state, mdp):
            start_cooking_fitness = start_cooking.fitness(pick_step, state, previous_state, agent_id, mdp)
            return start_cooking_fitness / 2
        return 0.0

    def success(agent_id: int, state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        player: PlayerState = state.players[agent_id]
        held = player.get_object() if player.has_object() else None
        if held and held.name == 'soup':
            return True
        return False

    return Goal(GoalSpaceEnum.PICKUP_SOUP, fitness_fn=fitness, success_fn=success)


def create_goal_spaces() -> List[Goal]:
    return [
        place_object_space(GoalSpaceEnum.PLACE_ONION, ItemCode.ONION.value),
        start_cooking_space(),
        pickup_soup_space(),
    ]

import random

from imrl_agent_new.overcooked.outcome import ItemCode, item_to_int
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, OvercookedGridworld, OvercookedState, PlayerState


class GoalSpace:
    def __init__(self, name, sampler, fitness_fn, success_fn=None):
        self.name = name
        self.sample = sampler
        self.fitness = fitness_fn
        # if None, the space never triggers early stop
        self.success = success_fn or (lambda obs, g: False)


PICKABLE_OBJECTS = [
    ItemCode.ONION,
    # ItemCode.TOMATO,
    # ItemCode.DISH,
]


def make_pick_object_space(length_of_trajectory: int) -> GoalSpace:
    """Goal: hold a requested object; reward falls as pickup time ↑."""

    def sampler():
        return random.choice(PICKABLE_OBJECTS).value

    def fitness(pick_step: int, g: int, end_of_episode: bool, state: OvercookedState, previous_state: OvercookedState,
                agent_id: int, mdp: OvercookedGridworld):
        if pick_step is None:
            return 0.0
        return 1.0 - (pick_step / max(length_of_trajectory, 1))  # fast = high

    def success(agent_id: int, g: int, state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        player: PlayerState = state.players[agent_id]
        held = item_to_int(player.get_object()) if player.has_object() else 0
        if held == int(g) and not previous_state.players[agent_id].has_object():
            return True
        return False

    return GoalSpace("PICK_OBJECT", sampler=sampler, fitness_fn=fitness, success_fn=success)


def place_object_space(length_of_trajectory: int) -> GoalSpace:
    """Goal: place a requested object; reward falls as placement time ↑."""

    def sampler():
        return ItemCode.ONION

    def fitness(pick_step: int, g: int, end_of_episode: bool, state: OvercookedState, previous_state: OvercookedState,
                agent_id: int, mdp: OvercookedGridworld):
        if make_pick_object_space(length_of_trajectory).success(agent_id, g, state, previous_state, mdp):
            # The agent has picked up an onion
            return 0.2
        if pick_step is None:
            return 0.0
        return 1.0 - (pick_step / max(length_of_trajectory, 1)) - 0.2  # Subtract 0.2 for picking up the object

    def success(agent_id: int, g: int, state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        current_agent: PlayerState = state.players[agent_id]
        previous_agent: PlayerState = previous_state.players[agent_id]
        previous_held: ObjectState = previous_agent.held_object if previous_agent.has_object() else 0
        current_held: ObjectState = current_agent.held_object if current_agent.has_object() else 0
        if not previous_held or current_held or item_to_int(
                previous_held) != g:  # The agent did not hold the correct object or is still holding it
            return False
        # At this point, the agent held the correct object and is not holding it anymore
        pot_locations = mdp.get_pot_locations()
        for index, pot in enumerate(pot_locations):
            current_pot = state.get_object(pot) if state.has_object(pot) else None
            previous_pot = previous_state.get_object(pot) if previous_state.has_object(pot) else None
            if previous_pot != current_pot:
                # The agent placed the object in the pot
                return True
        return False

    return GoalSpace("PLACE_OBJECT", sampler=sampler, fitness_fn=fitness, success_fn=success)


def start_cooking_space(length_of_trajectory: int) -> GoalSpace:
    """Goal: start cooking a requested object; reward falls as cooking time ↑."""

    def sampler():
        return None

    def fitness(pick_step: int, g: int, end_of_episode: bool, state: OvercookedState, previous_state: OvercookedState,
                agent_id: int, mdp: OvercookedGridworld):
        if end_of_episode:
            if pick_step is None:
                return 0.0
            return 1.0 - (pick_step / max(length_of_trajectory, 1))  # fast = high

        previous_agent: PlayerState = previous_state.players[agent_id]
        current_agent = state.players[agent_id]
        if not previous_agent.has_object() or current_agent.has_object():
            return 0.0

        if make_pick_object_space(length_of_trajectory).success(agent_id, ItemCode.ONION.value, state, previous_state,
                                                                mdp):
            # The agent has picked up an onion
            return 0.1
        if place_object_space(length_of_trajectory).success(agent_id, ItemCode.ONION.value, state, previous_state, mdp):
            # The agent has placed an onion in a pot
            return 0.1

        return 0


    def success(agent_id: int, g: int, state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        current_agent = state.players[agent_id]
        previous_agent: PlayerState = previous_state.players[agent_id]

        previous_pot_state = mdp.get_pot_states(previous_state)
        current_pot_state = mdp.get_pot_states(state)

        previous_cooking_pots = mdp.get_cooking_pots(previous_pot_state)
        current_cooking_pots = mdp.get_cooking_pots(current_pot_state)
        if previous_cooking_pots or not current_cooking_pots:  # If the agent had a pot cooking or is currently not cooking
            return False

        if not '3_items' in previous_pot_state:
            return False

        adjacent_features = mdp.get_adjacent_features(previous_agent)
        adjacent_positions = [feature[0] for feature in adjacent_features]  # Extract pot positions
        for pot in current_cooking_pots:
            if pot in adjacent_positions:
                # The agent is adjacent to the cooking pot
                return True
        return False

    return GoalSpace("START_COOKING", sampler=sampler, fitness_fn=fitness, success_fn=success)


def pickup_soup_space(length_of_trajectory: int) -> GoalSpace:
    def sampler():
        return None

    def fitness(pick_step=None, *args):
        if pick_step is None:
            return 0.0
        return 1.0

    def success(agent_id: int, g: int, state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        player: PlayerState = state.players[agent_id]
        held = player.get_object() if player.has_object() else 0
        if held == ItemCode.SOUP:
            return True
        return None

    return GoalSpace("PICKUP_SOUP", sampler=sampler, fitness_fn=fitness, success_fn=success)


def create_goal_space(length_of_trajectory):
    return {
        "place_object": place_object_space(length_of_trajectory),
        "pick_object": make_pick_object_space(length_of_trajectory),
        "start_cooking": start_cooking_space(length_of_trajectory),
        # "pickup_soup": pickup_soup_space(length_of_trajectory),
    }

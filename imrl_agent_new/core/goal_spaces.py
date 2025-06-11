from imrl_agent_new.overcooked.outcome import ItemCode, item_to_int
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, OvercookedGridworld, OvercookedState, PlayerState


class Goal:
    def __init__(self, name, fitness_fn, success_fn=None):
        self.name = name
        self.fitness = fitness_fn
        self.success = success_fn


def pick_object_space(length_of_trajectory: int, name, object_code) -> Goal:
    def fitness(pick_step: int, end_of_episode: bool, state: OvercookedState, previous_state: OvercookedState,
                agent_id: int, mdp: OvercookedGridworld):
        if pick_step is None:
            return 0.0
        return 1.0 - (pick_step / max(length_of_trajectory, 1))  # fast = high

    def success(agent_id: int, state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        player: PlayerState = state.players[agent_id]
        held = item_to_int(player.get_object()) if player.has_object() else 0
        if held == object_code and not previous_state.players[agent_id].has_object():
            return True
        return False

    return Goal(name, fitness_fn=fitness, success_fn=success)


def place_object_space(length_of_trajectory: int, name, object_code) -> Goal:
    def fitness(pick_step: int, end_of_episode: bool, state: OvercookedState, previous_state: OvercookedState,
                agent_id: int, mdp: OvercookedGridworld):
        if pick_object_space(length_of_trajectory, name, object_code).success(agent_id, state, previous_state, mdp):
            # The agent has picked up an onion
            return 0.2
        if pick_step is None:
            return 0.0
        return 1.0 - (pick_step / max(length_of_trajectory, 1)) - 0.2  # Subtract 0.2 for picking up the object

    def success(agent_id: int, state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        current_agent: PlayerState = state.players[agent_id]
        previous_agent: PlayerState = previous_state.players[agent_id]
        previous_held: ObjectState = previous_agent.held_object if previous_agent.has_object() else 0
        current_held: ObjectState = current_agent.held_object if current_agent.has_object() else 0
        if not previous_held or current_held or item_to_int(
                previous_held) != object_code:  # The agent did not hold the correct object or is still holding it
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

    return Goal(name, fitness_fn=fitness, success_fn=success)


def start_cooking_space(length_of_trajectory: int) -> Goal:
    """Goal: start cooking a requested object; reward falls as cooking time ↑."""

    def fitness(pick_step: int, end_of_episode: bool, state: OvercookedState, previous_state: OvercookedState,
                agent_id: int, mdp: OvercookedGridworld):
        if end_of_episode:
            if pick_step is None:
                return 0.0
            return 1.0 - (pick_step / max(length_of_trajectory, 1)) - 0.2  # fast = high

        previous_agent: PlayerState = previous_state.players[agent_id]
        current_agent = state.players[agent_id]
        if not previous_agent.has_object() or current_agent.has_object():
            return 0.0

        if pick_object_space(length_of_trajectory, "PICK_ONION", ItemCode.ONION.value).success(agent_id, state,
                                                                                               previous_state,
                                                                mdp):
            # The agent has picked up an onion
            return 0.1
        if place_object_space(length_of_trajectory, "PICK_ONION", ItemCode.ONION.value).success(agent_id, state,
                                                                                                previous_state, mdp):
            # The agent has placed an onion in a pot
            return 0.1

        return 0

    def success(agent_id: int, state: OvercookedState, previous_state: OvercookedState,
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

    return Goal("START_COOKING", fitness_fn=fitness, success_fn=success)


def pickup_soup_space(length_of_trajectory: int) -> Goal:
    def fitness(pick_step: int, end_of_episode: bool, state: OvercookedState, previous_state: OvercookedState,
                agent_id: int, mdp: OvercookedGridworld):
        if not end_of_episode:
            return start_cooking_space(length_of_trajectory).fitness(pick_step, end_of_episode, state, previous_state,
                                                                     agent_id, mdp)
        if end_of_episode and pick_step is None:
            return 0.0

        return 1.0 - (pick_step / max(length_of_trajectory, 1)) - 0.2  # fast = high

    def success(agent_id: int, state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        player: PlayerState = state.players[agent_id]
        held = player.get_object() if player.has_object() else 0
        if held == ItemCode.SOUP:
            return True
        return None

    return Goal("PICKUP_SOUP", fitness_fn=fitness, success_fn=success)


def create_goal_space(length_of_trajectory: int) -> dict[str, Goal]:
    """
    Create a goal space with various goals for the Overcooked environment.
    :param length_of_trajectory: The length of the trajectory for normalization.
    :return: A dictionary mapping goal names to Goal objects.
    """
    return {
        "PICK_ONION": pick_object_space(length_of_trajectory, "PICK_ONION", ItemCode.ONION.value),
        "PLACE_ONION": place_object_space(length_of_trajectory, "PLACE_ONION", ItemCode.ONION.value),
        "START_COOKING": start_cooking_space(length_of_trajectory),
        "PICKUP_SOUP": pickup_soup_space(length_of_trajectory)
    }
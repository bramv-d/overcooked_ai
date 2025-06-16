from imrl_agent_new.helper.obs_to_vect import ItemCode, item_to_int
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, OvercookedGridworld, OvercookedState, PlayerState


class Goal:
    def __init__(self, name, fitness_fn, success_fn=None):
        self.name = name
        self.fitness = fitness_fn
        self.success = success_fn


def pick_object_space(length_of_trajectory: int, name, object_code) -> Goal:
    def fitness(pick_step: int, state: OvercookedState, previous_state: OvercookedState,
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
    def fitness(pick_step: int, state: OvercookedState, previous_state: OvercookedState,
                agent_id: int, mdp: OvercookedGridworld):
        if pick_step is None:
            return 0.0
        return 1.0 - (pick_step / max(length_of_trajectory, 1))  # fast = high

    def success(agent_id: int, state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        current_agent: PlayerState = state.players[agent_id]
        previous_agent: PlayerState = previous_state.players[agent_id]
        previous_held: ObjectState = previous_agent.held_object if previous_agent.has_object() else 0
        current_held: ObjectState = current_agent.held_object if current_agent.has_object() else 0

        if not previous_held or current_held or item_to_int(previous_held) != object_code:
            return False  # Agent was not holding correct object, or still holding it

        pot_locations = mdp.get_pot_locations()

        # Get agent's adjacent positions (in CURRENT state)
        adjacent_features = mdp.get_adjacent_features(current_agent)
        adjacent_positions = {feature[0] for feature in adjacent_features}

        for pot in pot_locations:
            current_pot = state.get_object(pot) if state.has_object(pot) else None
            previous_pot = previous_state.get_object(pot) if previous_state.has_object(pot) else None
            if previous_pot != current_pot and pot in adjacent_positions:
                # The agent placed the object in this pot and was next to it
                return True

        return False

    return Goal(name, fitness_fn=fitness, success_fn=success)


def start_cooking_space(length_of_trajectory: int) -> Goal:
    """Goal: start cooking a requested object; reward falls as cooking time ↑."""

    def fitness(pick_step: int, state: OvercookedState, previous_state: OvercookedState,
                agent_id: int, mdp: OvercookedGridworld):
        previous_cooking_pots = set(mdp.get_cooking_pots(mdp.get_pot_states(previous_state)))  # set of positions
        current_cooking_pots = set(mdp.get_cooking_pots(mdp.get_pot_states(state)))  # set of positions
        # Get positions adjacent to the agent (CURRENT state)
        previous_agent: PlayerState = previous_state.players[agent_id]
        adjacent_features = mdp.get_adjacent_features(previous_agent)
        adjacent_positions = {feature[0] for feature in adjacent_features}  # convert to set for fast lookup
        # Check: any NEW pot that started cooking, AND agent is adjacent to it
        new_cooking_pots = current_cooking_pots - previous_cooking_pots  # set difference

        for pot_pos in new_cooking_pots:
            if pot_pos in adjacent_positions:
                held = state.get_object(pot_pos)

                # Final reward for successful pickup
                if not held or held.name != 'soup':
                    return 0.0
                return held.value / 65

        return 0

    def success(agent_id: int, state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        previous_cooking_pots = set(mdp.get_cooking_pots(mdp.get_pot_states(previous_state)))  # set of positions
        current_cooking_pots = set(mdp.get_cooking_pots(mdp.get_pot_states(state)))  # set of positions

        # Get positions adjacent to the agent (CURRENT state)
        current_agent = state.players[agent_id]
        adjacent_features = mdp.get_adjacent_features(current_agent)
        adjacent_positions = {feature[0] for feature in adjacent_features}  # convert to set for fast lookup

        # Check: any NEW pot that started cooking, AND agent is adjacent to it
        new_cooking_pots = current_cooking_pots - previous_cooking_pots  # set difference

        for pot_pos in new_cooking_pots:
            if pot_pos in adjacent_positions:
                return True

        return False

    return Goal("START_COOKING", fitness_fn=fitness, success_fn=success)


def pickup_soup_space(length_of_trajectory: int) -> Goal:
    start_cooking_goal = start_cooking_space(length_of_trajectory)
    pick_dish_goal = pick_object_space(length_of_trajectory, "PICK_DISH", ItemCode.DISH.value)
    pick_onion_goal = pick_object_space(length_of_trajectory, "PICK_ONION", ItemCode.ONION.value)

    def fitness(pick_step: int, state: OvercookedState, previous_state: OvercookedState,
                agent_id: int, mdp: OvercookedGridworld):
        # Phase 1: Start cooking → use start_cooking_goal's fitness
        # Check if cooking has started
        current_cooking_pots = set(mdp.get_cooking_pots(mdp.get_pot_states(state)))
        cooking_started = len(current_cooking_pots) > 0
        player: PlayerState = state.players[agent_id]
        held = player.get_object() if player.has_object() else 0

        # Final reward for successful pickup
        if held and held.name == 'soup':
            return held.value / 65  # Normalize by max soup value (65)

        if not cooking_started:
            return start_cooking_goal.fitness(pick_step, state, previous_state,
                                              agent_id, mdp)
        # Punish picking up something other than a dish
        if pick_onion_goal.success(agent_id, state, previous_state, mdp):
            # The agent has picked up an onion
            return -0.2

        # Detect bowl pickup
        if pick_dish_goal.success(agent_id, state, previous_state, mdp):
            # The agent is holding a dish
            return 0.2
        return 0.0  # No soup picked up yet

    def success(agent_id: int, state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        player: PlayerState = state.players[agent_id]
        held = player.get_object() if player.has_object() else 0
        if held and held.name == 'soup':
            return True
        return False

    return Goal("PICKUP_SOUP", fitness_fn=fitness, success_fn=success)


def deliver_soup_space(length_of_trajectory: int) -> Goal:
    pickup_soup_goal = pickup_soup_space(length_of_trajectory)

    # Goal: deliver a soup to a counter; reward falls as delivery time ↑.
    # This goalspace is agent independent, so it the agent can collaborate with other agents to deliver soup.
    def fitness(pick_step: int, state: OvercookedState, previous_state: OvercookedState,
                agent_id: int, mdp: OvercookedGridworld):
        gathered_fitness = pickup_soup_goal.fitness(pick_step, state, previous_state, agent_id, mdp)
        gathered_fitness += pickup_soup_goal.fitness(pick_step, state, previous_state, 1 - agent_id, mdp)
        return gathered_fitness

    def success(agent_id: int, state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        player: PlayerState = state.players[agent_id]
        held = player.get_object() if player.has_object() else 0
        mdp.get_serving_locations()
        return False

    return Goal("DELIVER_SOUP", fitness_fn=fitness, success_fn=success)

def create_goal_space(length_of_trajectory: int) -> dict[str, Goal]:
    """
    Create a goal space with various goals for the Overcooked environment.
    :param length_of_trajectory: The length of the trajectory for normalization.
    :return: A dictionary mapping goal names to Goal objects.
    """
    return {
        "PICK_ONION": pick_object_space(length_of_trajectory, "PICK_ONION", ItemCode.ONION.value),
        "PICK_DISH": pick_object_space(length_of_trajectory, "PICK_DISH", ItemCode.DISH.value),
        "PLACE_ONION": place_object_space(length_of_trajectory, "PLACE_ONION", ItemCode.ONION.value),
        "START_COOKING": start_cooking_space(length_of_trajectory),
        "PICKUP_SOUP": pickup_soup_space(length_of_trajectory),
        # "DELIVER_SOUP": deliver_soup_space(length_of_trajectory)
    }

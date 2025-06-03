import random

from imrl_agent_new.overcooked.outcome import ItemCode, item_to_int
from overcooked_ai_py.mdp.overcooked_mdp import ObjectState, OvercookedGridworld, OvercookedState, PlayerState


class GoalSpace:
    def __init__(self, name, dim, sampler, fitness_fn, success_fn=None):
        self.name    = name
        self.dim     = dim
        self.sample  = sampler
        self.fitness = fitness_fn
        # if None, the space never triggers early stop
        self.success = success_fn or (lambda obs, g: False)


PICKABLE_OBJECTS = [
    ItemCode.ONION,
    # ItemCode.TOMATO,
    ItemCode.DISH,
]

def make_pick_object_space(length_of_trajectory: int) -> GoalSpace:
    """Goal: hold a requested object; reward falls as pickup time ↑."""
    def sampler():
        return random.choice(PICKABLE_OBJECTS).value

    def fitness(pick_step=None):
        if pick_step is None:
            return 0.0
        return 1.0 - (pick_step / max(length_of_trajectory, 1))           # fast = high

    def success(agent_id: int, g, state: OvercookedState, previous_state: OvercookedState, mdp: OvercookedGridworld):
        player: PlayerState = state.players[agent_id]
        held = item_to_int(player.get_object()) if player.has_object() else 0
        if held == int(g):
            return True
        return None

    return GoalSpace("PICK_OBJECT", 1, sampler=sampler, fitness_fn=fitness, success_fn=success)


def place_object_space(length_of_trajectory: int) -> GoalSpace:
    """Goal: place a requested object; reward falls as placement time ↑."""

    def sampler():
        return ItemCode.ONION

    def fitness(pick_step=None):
        if pick_step is None:
            return 0.0
        return 1.0 - (pick_step / max(length_of_trajectory, 1))  # fast = high

    def success(agent_id: int, g: int, state: OvercookedState, previous_state: OvercookedState,
                mdp: OvercookedGridworld):
        current_agent = state.players[agent_id]
        previous_agent: PlayerState = previous_state.players[agent_id]
        previous_held: ObjectState = previous_agent.held_object if previous_agent.has_object() else 0
        current_held: ObjectState = current_agent.held_object if current_agent.has_object() else 0
        if not previous_held or current_held or item_to_int(
                previous_held) != g:  # The agent did not hold the correct object or is still holding it
            return None
        # At this point, the agent held the correct object and is not holding it anymore
        pot_locations = mdp.get_pot_locations()
        for index, pot in enumerate(pot_locations):
            previous_pot = state.get_object(pot) if state.has_object(pot) else None
            current_pot = previous_state.get_object(pot) if previous_state.has_object(pot) else None
            if previous_pot != current_pot:
                # The agent placed the object in the pot
                return True
        return False

    return GoalSpace("PLACE_OBJECT", 1, sampler=sampler, fitness_fn=fitness, success_fn=success)


def create_goal_space(length_of_trajectory):
    return {
        "pick_object" : make_pick_object_space(length_of_trajectory),
        "place_object": place_object_space(length_of_trajectory),
    }

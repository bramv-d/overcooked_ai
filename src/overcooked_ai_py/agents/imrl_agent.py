from typing import Tuple

import numpy as np

from overcooked_ai_py.agents.agent import Agent, AgentPair, GreedyHumanModel
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.data.layouts.layouts import layouts
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, PlayerState
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS, MotionPlanner
from overcooked_ai_py.utils import manhattan_distance
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.actions import Direction, Action
from testing.agent_test import force_compute


class IMRLAgent(Agent):
    def __init__(self, layout, player_id, sim_threads=None):
        super().__init__()
        self.mdp: OvercookedGridworld = OvercookedGridworld.from_layout_name(layout)
        self.motion_planner = MotionPlanner(self.mdp)
        self.medium_level_action_manager = MediumLevelActionManager(self.mdp, NO_COUNTERS_PARAMS)
        self.sim_threads = sim_threads
        self.player_id = player_id

    def get_player(self, state: OvercookedState) -> PlayerState:
        """
        Get the player state from the OvercookedState.
        """
        return state.players[self.player_id]

    def action(self, state: OvercookedState) -> Tuple[Action, dict]:
        if self.get_player(state).has_object():
            return self.go_to_soup_action(state)
        else:
            return self.get_nearest_onion_action(state)

    def get_nearest_onion_action(self, state: OvercookedState) -> Tuple[Action, dict]:
        """
        Get the nearest onion action from the state.
        """
        start_pos_and_or = self.get_player(state).pos_and_or
        counter_objects = self.mdp.get_counter_objects_dict(state)
        onion_locations = self.medium_level_action_manager.pickup_onion_actions(counter_objects)

        if not onion_locations:
            return None  # or handle appropriately

        # Find the location with the smallest Manhattan distance
        nearest_onion = min(onion_locations, key=lambda pos: manhattan_distance(start_pos_and_or[0], pos[0]))

        action_plan = self.motion_planner.get_plan(start_pos_and_or, nearest_onion)[0]

        action = action_plan[0]
        action_probs = Agent.a_probs_from_action(action)

        return action, {"action_probs": action_probs}

    def go_to_soup_action(self, state: OvercookedState) -> Tuple[Action, dict]:
        current_pos_and_or = self.get_player(state).pos_and_or
        pot_states_dict = self.medium_level_action_manager.mdp.get_pot_states(state)
        next_to_pot = self.medium_level_action_manager.put_onion_in_pot_actions(pot_states_dict)
        action_plan = self.motion_planner.get_plan(current_pos_and_or, next_to_pot[0])[0]

        action = action_plan[0]
        action_probs = Agent.a_probs_from_action(action)
        return action, {"action_probs": action_probs}

    def actions(self, states, agent_indices):
        return [self.action(state) for state in states]


def evaluate_agent_pair(agent_pair, layout):
    """
    Evaluate a pair of agents on the given layout.
    """
    ae = AgentEvaluator.from_layout_name(mdp_params={"layout_name": layout, "old_dynamics": False},
                                         env_params={"horizon": 20})  # Set horizon to 5
    return ae.evaluate_agent_pair(agent_pair, 1)  # Run 5 games


# Make example usage
if __name__ == "__main__":
    layout_name = layouts[20]  # Choose a layout from the list of layouts
    agent1 = IMRLAgent(layout=layout_name, player_id=0)
    agent2 = IMRLAgent(layout=layout_name, player_id=1)
    # Evaluate the agent pair
    results = evaluate_agent_pair(AgentPair(agent1, agent2), layout_name)

    # Desired base directory
    base_dir = "/Users/bram/Documents/Afstuderen/images/trajectories"

    StateVisualizer().display_rendered_trajectory(results, img_directory_path=base_dir, ipython_display=False)

# Next steps
# 1. How to measure the change in state? How to measure the effect of an action?
# 2. How to transfer the effect of the agent towards the competence?
# 3. Progress is the difference between current and old competence
# 4. Based on the progress model the interest in the action.
# 5. How to make the agent learn based on its own actions and the changes it provokes?
# 6. How to determine whether the agent should explore new actions or continue learning the current action?

# Eventual idea
# Feed the important state properties to the agent
# Let it check with which action it can provoke the biggest change in the state
# Perform the action with the biggest change and save this in some way
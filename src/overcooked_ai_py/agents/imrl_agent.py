from collections import defaultdict
from typing import Tuple, Any, Optional

import numpy as np

from overcooked_ai_py.agents.agent import Agent, AgentPair, GreedyHumanModel
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.data.layouts.layouts import layouts
from overcooked_ai_py.mdp.layout_generator import CODE_TO_TYPE
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, PlayerState, ObjectState
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS, MotionPlanner
from overcooked_ai_py.utils import manhattan_distance
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.actions import Direction, Action
from testing.agent_test import force_compute

class EnvironmentTypeRepresentation:
    """
    Class to represent the Environment type the agent is facing
    """

    def __init__(
            self,
            environment_type: CODE_TO_TYPE, # The type of the environment (e.g., 0 for empty, 1 for counter, etc.)
            is_interactable: bool,  # Can agent interact?
            can_receive_object: bool,  # Will it accept held object?
            current_contents: Optional[str]  # e.g., "onion", "2_onions", "empty", "cooking"
    ):
        self.environment_type = environment_type
        self.is_interactable = is_interactable
        self.can_receive_object = can_receive_object
        self.current_contents = current_contents

    def to_key(self):
        return self.environment_type, self.is_interactable, self.can_receive_object, self.current_contents

    def __eq__(self, other):
        return self.to_key() == other.to_key()

    def __hash__(self):
        return hash(self.to_key())

    def __repr__(self):
        return (
            f"Target(type={self.environment_type}, interactable={self.is_interactable}, "
            f"receives={self.can_receive_object}, contents={self.current_contents})"
        )


class ActionEnvironmentCoupling:
    """
    Class to represent the coupling between actions, agent_object_state and environment_object.
    The certainty of the coupling can be calculated based on the number of times the action was performed and the result it had
    """
    def __init__(
        self,
        action: Action, #The performed action of the agent
        agent_object_state: ObjectState, # The state of the object the agent is holding
        environment_type_representation: EnvironmentTypeRepresentation, # The type of the environment and its properties
    ):
        self.action = action
        self.agent_object_state = agent_object_state
        self.environment_type_representation = environment_type_representation

    def to_key(self):
        return self.action, self.agent_object_state, self.environment_type_representation
    def __eq__(self, other):
        return self.to_key() == other.to_key()
    def __hash__(self):
        return hash(self.to_key())

class IMRLAgent(Agent):
    def __init__(self, layout, player_id, sim_threads=None):
        super().__init__()
        self.mdp: OvercookedGridworld = OvercookedGridworld.from_layout_name(layout)
        self.motion_planner = MotionPlanner(self.mdp)
        self.medium_level_action_manager = MediumLevelActionManager(self.mdp, NO_COUNTERS_PARAMS)
        self.sim_threads = sim_threads
        self.player_id = player_id

        # Memory: stores coupling → list of observed outcomes
        self.memory = defaultdict(list)

    def get_player(self, state: OvercookedState) -> PlayerState:
        """
        Get the player state from the OvercookedState.
        """
        return state.players[self.player_id]

    def action(self, state: OvercookedState) -> Tuple[Action, dict]:
        possible_actions = self.mdp.get_actions(state)[self.player_id]

        return Action.STAY, {}  # Placeholder for the action

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
# 4. Based on the progress model, the interest in the action.
# 5. How to make the agent learn based on its own actions and the changes it provokes?
# 6. How to determine whether the agent should explore new actions or continue learning the current action?

# Eventual idea
# Feed the important state properties to the agent
# Let it check with which action it can provoke the biggest change in the state
# Perform the action with the biggest change and save this in some way


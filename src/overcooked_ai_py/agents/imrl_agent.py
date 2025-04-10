from datetime import datetime
from typing import Tuple

import numpy as np

from overcooked_ai_py.agents.agent import Agent, AgentPair, GreedyHumanModel
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.data.layouts.layouts import layouts
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, PlayerState
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.actions import Direction, Action
from testing.agent_test import force_compute


class IMRLAgent(Agent):
    def __init__(self, layout, player_id, sim_threads=None):
        super().__init__()
        self.simple_mdp: OvercookedGridworld = OvercookedGridworld.from_layout_name(layout)
        self.mlam: MediumLevelActionManager = MediumLevelActionManager.from_pickle_or_compute(
            self.simple_mdp, NO_COUNTERS_PARAMS, force_compute=True, info=False
        )
        self.sim_threads = sim_threads
        self.player_id = player_id

    def get_player(self, state: OvercookedState):
        """
        Get the player state from the OvercookedState.
        """
        return state.players[self.player_id]

    def action(self, state: OvercookedState) -> Tuple[Action, dict]:
        closest_feature_actions = self.mlam.go_to_closest_feature_actions(self.get_player(state))
        action = closest_feature_actions[0][1]
        action_probs = Agent.a_probs_from_action(action)
        print(closest_feature_actions)

        return action, {"action_probs": action_probs}

    def actions(self, states, agent_indices):
        return [self.action(state) for state in states]


def evaluate_agent_pair(agent_pair, layout):
    """
    Evaluate a pair of agents on the given layout.
    """
    print('Evaluating agent pair...')
    ae = AgentEvaluator.from_layout_name(mdp_params={"layout_name": layout, "old_dynamics": False},
                                         env_params={"horizon": 5})  # Set horizon to 5
    return ae.evaluate_agent_pair(agent_pair, 1)  # Run 5 games

# Make example usage
if __name__ == "__main__":
    layout_name = layouts[16]  # Choose a layout from the list of layouts
    agent1 = IMRLAgent(layout=layout_name, player_id=0)
    agent2 = IMRLAgent(layout=layout_name, player_id=1)
    # Evaluate the agent pair
    results = evaluate_agent_pair(AgentPair(agent1, agent2), layout_name)

    # Desired base directory
    base_dir = "/Users/bram/Documents/Afstuderen/images/trajectories"

    StateVisualizer().display_rendered_trajectory(results, img_directory_path=base_dir, ipython_display=False)

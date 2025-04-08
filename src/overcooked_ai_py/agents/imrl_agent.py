from datetime import datetime

import numpy as np

from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.overcooked_mdp import SoupState, OvercookedState
from overcooked_ai_py.planning.planners import MotionPlanner
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action, Direction


class IMRLAgent(Agent):
    def __init__(self, sim_threads=None):
        self.sim_threads = sim_threads

    def action(self, state):
        action = Direction.EAST
        action_probs = Agent.a_probs_from_action(action)
        self.go_to_nearest_soup(state)
        return action, {"action_probs": action_probs}

    def actions(self, states, agent_indices):
        return [self.action(state) for state in states]

    def go_to_nearest_soup(self, state):
        print(state.all_objects_by_type.get("pot"))


def evaluate_agent_pair(agent_pair):
    """
    Evaluate a pair of agents on the given layout.
    """
    print('Evaluating agent pair...')
    layout = "corridor"
    ae = AgentEvaluator.from_layout_name(mdp_params={"layout_name": layout, "old_dynamics": True},
                                         env_params={"horizon": 400})

    return ae.evaluate_agent_pair(agent_pair, 10)


# Make example usage
if __name__ == "__main__":
    agent1 = IMRLAgent()
    agent2 = IMRLAgent()
    # Evaluate the agent pair
    results = evaluate_agent_pair(AgentPair(agent1, agent2))

    # Desired base directory
    base_dir = "/Users/bram/Documents/Afstuderen/images/trajectories"

    StateVisualizer().display_rendered_trajectory(results, img_directory_path=base_dir, ipython_display=False)

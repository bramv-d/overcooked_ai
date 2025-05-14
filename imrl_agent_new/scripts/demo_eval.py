"""
Evaluate TWO ModularIMGEPAgent instances for one game
and save a rendered trajectory GIF.

Run:
    python -m imrl_agent_new.scripts.demo_eval
"""

from __future__ import annotations
import pathlib

from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.data.layouts.layouts import layouts
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from imrl_agent_new.overcooked.agent import IMGEPAgent


def evaluate_agent_pair(agent_pair, layout):
    ae = AgentEvaluator.from_layout_name(mdp_params={"layout_name": layout, "old_dynamics": False},
                                         env_params={"horizon": 400})  # Set horizon to 5
    return ae.evaluate_agent_pair(agent_pair, 2)  # Run 5 games


# Make example usage
if __name__ == "__main__":
    layout_name = layouts[20]  # Choose a layout from the list of layouts
    grid_world = OvercookedGridworld.from_layout_name(layout_name)
    agent1 = IMGEPAgent(grid_world, 0)
    agent2 = IMGEPAgent(grid_world, 1)
    # Evaluate the agent pair
    results = evaluate_agent_pair(AgentPair(agent1, agent2), layout_name)

    # Desired base directory
    base_dir = "/Users/bram/Documents/Afstuderen/Overcooked/imrl_agent_new/trajectory"

    StateVisualizer().display_rendered_trajectory(results, img_directory_path=base_dir + "/images",
                                                  ipython_display=False)
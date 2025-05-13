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
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

from imrl_agent_new.overcooked.agent import ModularIMGEPAgent


# --------------------------------------------------------------------------- #
def build_agent_pair(layout_name: str) -> AgentPair:



def evaluate(agent_pair: AgentPair, layout: str, horizon: int = 400):
    ae = AgentEvaluator.from_layout_name(
        mdp_params={"layout_name": layout, "old_dynamics": False},
        env_params={"horizon": horizon},
    )
    return ae.evaluate_agent_pair(agent_pair, 1)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    layout_name = layouts[20]  # e.g. "large_room"
    print(f"Evaluating IMGEP agents on layout: {layout_name}")

    pair = build_agent_pair(layout_name)
    results = evaluate(pair, layout_name, horizon=500)

    out_dir = pathlib.Path(
        "/Users/bram/Documents/Afstuderen/imrl_agent_new/trajectory/images"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    StateVisualizer().display_rendered_trajectory(
        results, img_directory_path=str(out_dir), ipython_display=False
    )
    print(f"GIF & frames saved to: {out_dir}")

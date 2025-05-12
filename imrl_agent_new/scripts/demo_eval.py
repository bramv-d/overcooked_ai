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

from imrl_agent_new.core.memory import EpisodicMemory
from imrl_agent_new.core.goal_policy import EpsilonGreedyBandit
from imrl_agent_new.core.exploration_policy import LinearSamplingPolicy
from imrl_agent_new.overcooked.agent import ModularIMGEPAgent


# --------------------------------------------------------------------------- #
def build_agent_pair(layout_name: str) -> AgentPair:
    memory = EpisodicMemory()
    Gamma = EpsilonGreedyBandit(epsilon=0.3)
    # θ  = 4 motion actions × 20 obs dims = 80
    Pi_prime = LinearSamplingPolicy(feat_dim=24, theta_dim=80)

    a0 = ModularIMGEPAgent(0, memory, Gamma, Pi_prime)
    a1 = ModularIMGEPAgent(1, memory, Gamma, Pi_prime)
    for a in (a0, a1):
        a.reset_episode(layout_name)

    return AgentPair(a0, a1)


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

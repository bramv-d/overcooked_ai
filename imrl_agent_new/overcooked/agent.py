# imrl_agent_new/imgep_agent.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from imrl_agent_new.core.goal_policy import GoalSpacePolicy
from imrl_agent_new.core.goal_spaces import create_goal_space
from imrl_agent_new.core.knowledge_base import ExperimentRecord, KnowledgeBase
from imrl_agent_new.core.neuro_policy import NeuroPolicy
from imrl_agent_new.core.population_explorer import PopulationExplorer
from imrl_agent_new.helper.obs_to_vect import obs_to_vec
from imrl_agent_new.overcooked.outcome import extract_outcome
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.planning.planners import MotionPlanner


# --------------------------------------------------------------------------- #
@dataclass
class RolloutStats:
    goal_space : str
    goal_vec: List[int]
    dishes     : int
    fitness    : float
    intrinsic  : float
    steps      : int


# ------------ utility -------------------------------------------------------
G_ENC = 3  # length of the padded goal-encoding vector


def pad_goal(enc: np.ndarray, length: int = G_ENC) -> np.ndarray:
    out = np.zeros(length, np.float32)
    out[:len(enc)] = enc
    return out


# --------------------------------------------------------------------------- #
class IMGEPAgent(Agent):
    """
    Intrinsically-Motivated Goal-Exploration Agent with neuro-evolution
    controllers.
    """

    def __init__(self, grid_world: OvercookedGridworld, agent_id: int, horizon: int = 400, epsilon: float = 1.0,
                 mp: MotionPlanner | None = None, max_dist: int = 0):
        # ---------- static members -----------------------------------------
        self.agent_id = agent_id
        self.horizon      = horizon
        self.mdp = grid_world
        self.mp = mp
        self.max_dist = max_dist

        # ---------- goal-spaces & bandit -----------------------------------
        self.G = create_goal_space(grid_world, horizon)
        self.bandit = GoalSpacePolicy(self.G, epsilon=epsilon)

        # ---------- KB + explorer ------------------------------------------
        self.kb = KnowledgeBase(context_dim=1, outcome_dim=5)
        self.obs_dim = 11  # from obs_to_vec
        self.explorer = PopulationExplorer(self.kb, NeuroPolicy, self.obs_dim)

        # ---------- per-rollout fields -------------------------------------
        self.goal_space_id: str  | None = None
        self.goal_vec     : np.ndarray | None = None
        self.theta: NeuroPolicy | None = None
        self.t            : int = 0
        self.meta         : Dict[str, Any] = {}
        self.use_pi: bool = False
        self.last_stats: RolloutStats | None = None

        super().__init__()

    @property
    def rollout_stats(self) -> RolloutStats | None:
        return self.last_stats

    # ---------------------------------------------------------------- reset
    def reset(self):
        """Called by Overcooked-AI at roll-out start."""
        # pick new goal
        self.goal_space_id, self.goal_vec = self.bandit.next_goal(None)
        self.t = 0
        self.meta = {"reach_step": None, "pick_step": None, "fill_step": None}

        # decide exploration vs exploitation (20 % exploit)
        self.use_pi = (random.random() > 0.8)

        if self.use_pi and len(self.kb) > 0:
            # reuse best θ so far
            best_idx = max(range(len(self.kb)),
                           key=lambda i: self.kb.buffer[i].fitness)
            theta_vec = self.kb.buffer[best_idx].theta

            self.theta = NeuroPolicy(obs_dim=self.obs_dim, theta=theta_vec)
        else:
            # exploration: mutate nearest or random
            self.theta = NeuroPolicy(obs_dim=self.obs_dim)

            # ---------------------------------------------------------------- action

    def action(self, state):
        obs_vec = obs_to_vec(state, self.mdp, self.mp, self.agent_id, self.max_dist)
        g_enc = pad_goal(self.G[self.goal_space_id].encode(self.goal_vec))

        act_enum = self.theta.act(obs_vec, g_enc)

        # ------ update meta for pick_object ----------------------------------
        if self.goal_space_id == "pick_object":
            target = int(self.goal_vec[0])
            held = obs_vec[3] * 5  # undo scaling
            if held == target and self.meta["pick_step"] is None:
                self.meta["pick_step"] = self.t

        self.t += 1
        return act_enum, {}

    # ---------------------------------------------------------------- finish
    def finish_rollout(self,
                       final_state: OvercookedState,
                       soups_delivered: int):
        """Call once the environment signals done=True."""
        outcome = extract_outcome(final_state,
                                  final_state.players[self.agent_id],
                                  self.mdp, soups_delivered)

        gs = self.G[self.goal_space_id]
        fitness = gs.fitness(outcome, self.goal_vec, **self.meta)

        # intrinsic reward = improvement over best previous fitness
        if len(self.kb) == 0:
            prev_f = 0.0
        else:
            idx, _ = self.kb.nearest(np.array([0]), outcome)
            prev_f = self.kb.buffer[idx].fitness
        r_i = fitness - prev_f

        # save rollout stats
        self.last_stats = RolloutStats(
            goal_space=self.goal_space_id,
            goal_vec=self.goal_vec.tolist(),
            dishes=soups_delivered,
            fitness=fitness,
            intrinsic=r_i,
            steps=self.t
        )

        # save experiment record
        self.kb.add_record(ExperimentRecord(
            context=np.array([0]),
            goal=self.goal_vec,
            theta=self.theta.theta,
            outcome=outcome,
            fitness=fitness,
            intrinsic_reward=r_i
        ))

        # bandit update only on exploitation runs
        if self.use_pi:
            self.bandit.update(self.goal_space_id, r_i)

    def save_kb(self, path: str):
        np.savez_compressed(
            path,
            outcomes=np.stack([r.outcome for r in self.kb.buffer]),
            fitness=np.array([r.fitness for r in self.kb.buffer]),
            goals=np.stack([r.goal for r in self.kb.buffer])
        )

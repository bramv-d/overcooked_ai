# imrl_agent_new/imgep_agent.py
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from imrl_agent_new.core.goal_policy import GoalSpacePolicy
from imrl_agent_new.core.goal_spaces import create_goal_space
from imrl_agent_new.core.knowledge_base import ExperimentRecord, KnowledgeBase
from imrl_agent_new.core.neuro_policy import NeuroPolicy  # <-- back to NN
from imrl_agent_new.core.population_explorer import PopulationExplorer
from imrl_agent_new.helper.choose_goal import get_plan
from imrl_agent_new.helper.high_level_actions import HighLevelActions
from imrl_agent_new.helper.obs_to_vect import obs_to_vec
from imrl_agent_new.overcooked.outcome import extract_outcome
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.planning.planners import MediumLevelActionManager, MotionPlanner

G_ENC = 3  # padded goal-encoding length


@dataclass
class RolloutStats:
    goal_space: str
    goal_vec: List[int]
    dishes: int
    fitness: float
    intrinsic: float
    steps: int


def pad_goal(enc: np.ndarray, length: int = G_ENC) -> np.ndarray:
    out = np.zeros(length, np.float32)
    out[:len(enc)] = enc
    return out


# --------------------------------------------------------------------------- #
class IMGEPAgent(Agent):
    """
    IMGEP agent with a *neuro-evolution* controller (NeuroPolicy).
    """

    def __init__(
            self,
            mlam: MediumLevelActionManager,
            grid_world: OvercookedGridworld,
            agent_id: int,
            horizon: int = 400,
            epsilon: float = 1.0,
            mp: MotionPlanner | None = None,
            max_dist: int = 0,
    ):
        # ---------- env refs ----------------------------------------------
        self.agent_id = agent_id
        self.mdp = grid_world
        self.horizon = horizon
        self.mp = mp
        self.max_dist = max_dist
        self.mlam = mlam
        # ---------- IMGEP machinery --------------------------------------
        self.G = create_goal_space(grid_world, horizon)
        self.bandit = GoalSpacePolicy(self.G, epsilon=epsilon)

        self.kb = KnowledgeBase(context_dim=1, outcome_dim=5)
        self.obs_dim = 14  # from obs_to_vec()
        self.explorer = PopulationExplorer(self.kb, NeuroPolicy,
                                           self.obs_dim,  # ← unchanged
                                           G_ENC)
        # ---------- per-rollout fields -----------------------------------
        self.goal_space_id: str | None = None
        self.goal_vec     : np.ndarray | None = None
        self.theta: NeuroPolicy | None = None
        self.use_pi: bool = False
        self.t: int = 0
        self.meta         : Dict[str, Any] = {}
        self.last_stats: RolloutStats | None = None
        self.goal_reach_time_step = None

        # ---------- path planning --------------------------------------
        self.path = []
        super().__init__()

    # ---------------------------------------------------------------- reset
    def reset(self):
        """Called by Overcooked-AI at episode start."""
        self.goal_space_id, self.goal_vec = self.bandit.next_goal(None)
        self.goal_reach_time_step = None
        self.t = 0
        self.meta = {"reach_step": None, "pick_step": None, "fill_step": None}

        # 20 % exploitation, 80 % exploration
        self.use_pi = (random.random() > 0.8)

        if self.use_pi and len(self.kb):
            best_idx = max(range(len(self.kb)), key=lambda i: self.kb.buffer[i].fitness)
            theta_vec = self.kb.buffer[best_idx].theta
            self.theta = NeuroPolicy(obs_dim=self.obs_dim, goal_enc_dim=G_ENC, theta=theta_vec)
        else:
            self.theta = self.explorer.sample_or_mutate(self.goal_vec)  # NeuroPolicy

    # ---------------------------------------------------------------- action
    def action(self, state: OvercookedState):
        # optional success time-step logging
        if (self.G[self.goal_space_id].success(state.players[self.agent_id], self.goal_vec)
                and self.goal_reach_time_step is None):
            self.goal_reach_time_step = self.t

        if self.path:
            step = self.path.pop(0)
            return step, {}

        obs_vec = obs_to_vec(state, self.mdp, self.mp,
                             self.agent_id, self.max_dist)
        goal_enc = pad_goal(self.G[self.goal_space_id].encode(self.goal_vec))

        # ---------------- legality mask & token selection ------------------
        token = HighLevelActions(
            self.theta.select_token(obs_vec, goal_enc, greedy=True))

        # ----------------- path planning ----------------------------------
        motion_goals = self._get_motion_goals(token, state)
        self.path = get_plan(state.players[self.agent_id].pos_and_or, motion_goals, self.mlam)
        if not self.path:
            return Action.STAY, {}
        action = self.path.pop(0)
        return action, {}

    # ---------------------------------------------------------------- finish
    def finish_rollout(self, final_state: OvercookedState, soups_delivered: int):
        outcome = extract_outcome(final_state,
                                  final_state.players[self.agent_id],
                                  self.mdp, soups_delivered)

        gs = self.G[self.goal_space_id]
        fitness = gs.fitness(outcome, self.goal_vec,
                             pick_step=self.goal_reach_time_step)

        # intrinsic reward = Δ fitness vs nearest prior experiment
        if len(self.kb) == 0:
            prev_f = 0.0
        else:
            idx, _ = self.kb.nearest(np.array([0]), outcome)
            prev_f = self.kb.buffer[idx].fitness
        r_i = fitness - prev_f

        self.last_stats = RolloutStats(
            goal_space=self.goal_space_id,
            goal_vec=self.goal_vec.tolist(),
            dishes=soups_delivered,
            fitness=fitness,
            intrinsic=r_i,
            steps=self.t
        )

        self.kb.add_record(ExperimentRecord(
            context=np.array([0]),
            goal=self.goal_vec,
            theta=self.theta.theta,  # store network parameters
            outcome=outcome,
            fitness=fitness,
            intrinsic_reward=r_i
        ))

        if self.use_pi:
            self.bandit.update(self.goal_space_id, r_i)

    def _get_motion_goals(self, high_level_action: HighLevelActions, state: OvercookedState):
        all_counters = self.mdp.get_counter_locations()
        counter_objects = self.mdp.get_counter_objects_dict(state, all_counters)
        pots_object = self.mdp.get_pot_states(state)
        match high_level_action:
            case HighLevelActions.GO_ONION:
                return self.mlam.pickup_onion_actions(counter_objects)
            case HighLevelActions.GO_TOMATO:
                return self.mlam.pickup_tomato_actions(counter_objects)
            case HighLevelActions.GO_DISH:
                return self.mlam.pickup_dish_actions(counter_objects)
            case HighLevelActions.PUT_ONION:
                return self.mlam.put_onion_in_pot_actions(pots_object)
            case HighLevelActions.PUT_TOMATO:
                return self.mlam.put_tomato_in_pot_actions(pots_object)
            case HighLevelActions.GO_READY_POT:
                return self.mlam.pickup_soup_with_dish_actions(pots_object)
            case HighLevelActions.GO_SERVE:
                return self.mlam.deliver_soup_actions()
            case HighLevelActions.GO_COUNTER:
                return self.mlam.place_obj_on_counter_actions(state)
            case HighLevelActions.START_COOKING:
                return self.mlam.start_cooking_actions(pots_object)
            case HighLevelActions.WAIT:
                return []
        return None

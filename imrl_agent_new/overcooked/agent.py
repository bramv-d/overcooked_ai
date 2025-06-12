# imrl_agent_new/imgep_agent.py
from __future__ import annotations

import random

import numpy as np

from imrl_agent_new.core.goal_policy import GoalSpacePolicy
from imrl_agent_new.core.goal_spaces import create_goal_space
from imrl_agent_new.core.knowledge_base import ExperimentRecord, KnowledgeBase
from imrl_agent_new.core.neuro_policy import NeuroPolicy
from imrl_agent_new.helper.choose_goal import get_plan
from imrl_agent_new.helper.high_level_actions import HighLevelActions
from imrl_agent_new.helper.obs_to_vect import obs_to_vec
from imrl_agent_new.overcooked.outcome import extract_outcome
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.planning.planners import MediumLevelActionManager, MotionPlanner


# --------------------------------------------------------------------------- #
class IMGEPAgent(Agent):
    """
    IMGEP agent with a *neuro-evolution* controller (NeuroPolicy).
    """

    def __init__(
            self,
            env: OvercookedEnv,
            mdp: OvercookedGridworld,
            agent_id: int,
            horizon: int,
            max_dist: int,
    ):
        # ---------- env refs ----------------------------------------------
        self.agent_id = agent_id
        self.env: OvercookedEnv = env
        self.mdp: OvercookedGridworld = mdp
        self.horizon = horizon
        self.mp: MotionPlanner = env.mp
        self.max_dist = max_dist
        self.mlam: MediumLevelActionManager = env.mlam
        self.previous_state: OvercookedState | None = None
        # ---------- IMGEP machinery --------------------------------------
        self.G = create_goal_space(horizon)
        self.bandit = GoalSpacePolicy(self.G)
        self.kb = KnowledgeBase()
        # ---------- per-rollout fields -----------------------------------
        self.goal_space_id: str | None = None
        self.neuro_policy: NeuroPolicy | None = None
        self.use_pi: bool = False
        self.t: int = 0
        self.goal_reach_time_step = None
        self.parent_policy: ExperimentRecord | None = None
        self.previous_state: OvercookedState | None = None
        self.rollout_fitness = 0.0
        # ---------- path planning --------------------------------------
        self.path = []
        super().__init__()

    # ---------------------------------------------------------------- reset
    def reset(self, mdp: OvercookedGridworld | None = None):
        """Called by Overcooked-AI at episode start."""
        super().reset()
        self.mdp = mdp
        self.goal_space_id = self.bandit.next_goal()
        self.goal_reach_time_step = None
        self.t = 0
        self.rollout_fitness = 0.0
        self.path = []

        self.use_pi = (random.random() > 0.8)

        if not self.kb.nearest(
                self.goal_space_id, 1):
            # --- explore: create new policy --
            self.neuro_policy = NeuroPolicy()
        elif self.use_pi:
            # --- exploit: clone best policy ----
            relevant_policies = self.kb.nearest(
                self.goal_space_id, 50)
            best_idx = np.argmax([r.fitness for r in relevant_policies]) if relevant_policies else 0
            self.parent_policy = relevant_policies[best_idx]
            theta_vec = self.parent_policy.theta
            self.neuro_policy = NeuroPolicy(theta=theta_vec)
        else:
            # --- explore:  mutate policy --
            relevant_policies = []
            pick_other_policy = random.random() < 0.3
            if pick_other_policy:  # Pick a random policy from an (other) goal space which is performing well
                relevant_policies = self.kb.well_performing_policies(
                    fitness_threshold=0.5,
                    record_amount=20,
                    exploit=True
                )
            if not relevant_policies:
                relevant_policies = self.kb.nearest(
                    self.goal_space_id, 50)

            self.parent_policy = random.choice(relevant_policies)
            adaptive_noise = 0.1 / (self.parent_policy.intrinsic_reward + 0.1)  # more noise when progress is low
            child_theta = self.parent_policy.theta + np.random.normal(0, adaptive_noise, self.parent_policy.theta.shape)
            self.neuro_policy = NeuroPolicy(theta=child_theta)

    # ---------------------------------------------------------------- action
    def action(self, state: OvercookedState):
        self.t += 1

        gs = self.G[self.goal_space_id]

        if self.previous_state and gs.success(self.agent_id, state, self.previous_state,
                                              self.mdp) and self.goal_reach_time_step is None:
            self.goal_reach_time_step = self.t

        if self.previous_state and (self.goal_reach_time_step is None or self.t == self.goal_reach_time_step):
            self.rollout_fitness += gs.fitness(
                pick_step=self.goal_reach_time_step,
                state=state,
                previous_state=self.previous_state,
                agent_id=self.agent_id,
                mdp=self.mdp
            )

        if self.path:
            step = self.path.pop(0)
            return self.return_action(step, state)

        obs_vec = obs_to_vec(state, self.mdp, self.mp,
                             self.agent_id)

        # ----------------- high-level action selection ----------------------
        if self.use_pi or self.parent_policy is None:
            greedy = True
        else:
            greedy = self.parent_policy.intrinsic_reward > 0.4

        token = HighLevelActions(
            self.neuro_policy.select_token(obs_vec, greedy=greedy))

        # ----------------- path planning ----------------------------------
        motion_goals = self._get_motion_goals(token, state)
        self.path = get_plan(state.players[self.agent_id].pos_and_or, motion_goals, self.mlam)

        if not self.path:
            legal_actions = list(Action.MOTION_ACTIONS)
            # Pick a random action from the legal actions
            random_action = random.choice(legal_actions)
            return self.return_action(random_action, state)
        action = self.path.pop(0)
        return self.return_action(action, state)

    def return_action(self, action, state):
        self.previous_state = state
        return action, {}

    # ---------------------------------------------------------------- finish
    def finish_rollout(self, final_state: OvercookedState, start_state: OvercookedState):
        outcome = extract_outcome(final_state,
                                  final_state.players[self.agent_id],
                                  self.mdp)


        # intrinsic reward = Δ fitness vs nearest prior experiment
        if len(self.kb) == 0 or not self.parent_policy:
            prev_f = 0.0
        else:
            nearest = self.kb.nearest(self.goal_space_id, 1, self.use_pi)
            prev_f = np.mean([r.fitness for r in nearest]) if nearest else 0.0

        r_i = max(0.0, self.rollout_fitness - prev_f)

        if self.use_pi:
            record_amount = 1
        else:
            record_amount = max(1, int(10 * r_i))

        rollout_idx = self.kb.buffer[-1].rollout_idx + 1 if self.kb.buffer else 0

        if self.use_pi:
            self.bandit.update(self.goal_space_id, r_i)

        # Save the amount of records in the knowledge base based on the r_i
        # This follows the idea of neuro evolution where successful policies are recorded more often and bad policies are recorded less often
        for _ in range(record_amount):
            self.kb.add_record(ExperimentRecord(
                goal_space=self.goal_space_id,
                theta=self.neuro_policy.theta,
                outcome=outcome,
                fitness=self.rollout_fitness,
                intrinsic_reward=r_i,
                exploit=self.use_pi,
                rollout_idx=rollout_idx,
            ))


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
            case HighLevelActions.START_COOKING:
                return self.mlam.start_cooking_actions(pots_object)
            case HighLevelActions.GO_COUNTER:
                return self.mlam.place_obj_on_counter_actions(state)
            case HighLevelActions.WAIT:
                # If the agent is waiting, return an empty list to indicate no motion goals
                return []
        return None

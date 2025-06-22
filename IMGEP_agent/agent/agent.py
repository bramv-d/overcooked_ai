import random
from typing import List

from IMGEP_agent.agent.goals.goal_policy import GoalSpaceEMA, select_goal_space, update_goal_space_ema
from IMGEP_agent.agent.goals.goal_spaces import Goal, create_goal_spaces
from IMGEP_agent.agent.helpers.get_plan import get_plan
from IMGEP_agent.agent.helpers.obs_to_vect import obs_to_vec
from IMGEP_agent.agent.knowledge_base import KnowledgeBase, RolloutRecord
from IMGEP_agent.agent.neuro_policy.high_level_actions import HighLevelActions, get_motion_goals
from IMGEP_agent.agent.neuro_policy.neuro_policy import GoalSpaceNeuroPolicy, get_neuro_policy
from IMGEP_agent.hyper_parameters import EXPLOIT_PROB
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.planning.planners import MotionPlanner


# TODO 1. Moet ik de keuze van het goal space afhankelijk maken van de state?
# TODO 1a. Heeft 1 episode meerdere goals?
# TODO 2. Hoe spelen de goal space intrinsic rewards van de andere agent een rol in de keuze van het goal space?
# TODO 3. Hoe voorspel je de goal space van de andere agent op basis van de goal space intrinsic reward?

class IMGEPAgent(Agent):
    def __init__(
            self,
            env: OvercookedEnv,
            mdp: OvercookedGridworld,
            agent_id: int,
    ):
        self.agent_id = agent_id
        self.env: OvercookedEnv = env
        self.mdp: OvercookedGridworld = mdp
        self.mp: MotionPlanner = env.mp
        self.mlam = env.mlam

        self.goal_spaces: List[Goal] = create_goal_spaces()
        self.kb = KnowledgeBase()
        self.goal_space_emas: List[GoalSpaceEMA] = []
        self.update_goal_space_emas()
        # Rollout bookkeeping
        self.previous_state = None
        self.goal_space_neuro_policies: List[
            GoalSpaceNeuroPolicy] = []  # List of neuro policies the 0th element being active
        self.t = 0
        self.goal_reach_time_step = None
        self.rollout_fitness = 0.0
        self.path = []  # This will hold the path of actions to take

    def update_goal_space_emas(self):
        """
        Update the goal space EMA based on the knowledge base.
        """
        self.goal_space_emas = update_goal_space_ema(self.kb, self.goal_spaces)

    def reset(self, mdp: OvercookedGridworld | None = None):
        super().reset()
        self.mdp = mdp
        self.t = 0
        self.path = []
        self.goal_reach_time_step = None
        self.rollout_fitness = 0.0
        self.previous_state = None
        chosen_goal = select_goal_space(self.goal_space_emas)
        exploit = random.random() < EXPLOIT_PROB  # Decide whether to exploit or explore
        neuro_policy = get_neuro_policy(selected_goal=chosen_goal, kb=self.kb, exploit=exploit,
                                        goal_spaces=self.goal_spaces)
        self.goal_space_neuro_policies = [
            GoalSpaceNeuroPolicy(goal=chosen_goal, neuro_policy=neuro_policy, exploit=exploit)
        ]  # Put the main neuro policy in the first position

    def action(self, state: OvercookedState) -> Action:
        self.t += 1

        gs = self.goal_space_neuro_policies[-1].goal
        exploit = self.goal_space_neuro_policies[-1].exploit
        if self.goal_reach_time_step:
            # If we have reached the goal, perform a random action
            legal_actions = list(Action.MOTION_ACTIONS)
            # Pick a random action from the legal actions
            random_action = random.choice(legal_actions)
            return self.return_action(random_action, state)

        if self.previous_state and gs.success(self.agent_id, state, self.previous_state,
                                              self.mdp) and self.goal_reach_time_step is None:
            if len(self.goal_space_neuro_policies) == 1:
                self.goal_reach_time_step = self.t
            else:
                self.goal_space_neuro_policies.pop()

        if self.previous_state and (self.goal_reach_time_step is None or self.t == self.goal_reach_time_step) and len(
                self.goal_space_neuro_policies) == 1:
            self.rollout_fitness += self.goal_space_neuro_policies[0].goal.fitness(
                # Only add the fitness of the first goal space
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
        neuro_policy_token = self.goal_space_neuro_policies[-1].neuro_policy.select_token(obs_vec, greedy=exploit)
        if neuro_policy_token < len(HighLevelActions):
            # If the token is a high-level action, we need to get the motion goals for that action
            token = HighLevelActions(neuro_policy_token)
            motion_goals = get_motion_goals(self.mlam, self.mdp, token, state)
            self.path = get_plan(state.players[self.agent_id].pos_and_or, motion_goals, self.mlam)
        else:
            self.goal_space_neuro_policies.append(
                GoalSpaceNeuroPolicy(
                    goal=self.goal_spaces[neuro_policy_token - len(HighLevelActions)],
                    neuro_policy=get_neuro_policy(
                        selected_goal=self.goal_spaces[neuro_policy_token - len(HighLevelActions)],
                        kb=self.kb,
                        exploit=True,
                        goal_spaces=self.goal_spaces
                    ),
                    exploit=True
                )
            )

        if not self.path:
            legal_actions = list(Action.MOTION_ACTIONS)
            # Pick a random action from the legal actions
            random_action = random.choice(legal_actions)
            return self.return_action(random_action, state)

        action = self.path.pop(0)
        return self.return_action(action, state)

    def return_action(self, action, state):
        self.previous_state = state
        return action

    def finish_rollout(self, final_state: OvercookedState, info):
        # intrinsic reward = Δ fitness vs nearest prior experiment
        goal = self.goal_space_neuro_policies[0].goal  # The first goal is the main goal space
        neuro_policy = self.goal_space_neuro_policies[0].neuro_policy
        exploit = self.goal_space_neuro_policies[0].exploit
        prev_record = self.kb.nearest(goal=goal, k=1, exploit=exploit)
        prev_f = prev_record[0].fitness if prev_record else 0.0

        fitness_difference = self.rollout_fitness - prev_f
        r_i = max(0.0, fitness_difference)

        if exploit:
            record_amount = 1
        else:
            record_amount = max(1, int(10 * r_i))

        rollout_idx = self.kb.buffer[-1].rollout_idx + 1 if self.kb.buffer else 0

        # Save the number of records in the knowledge base based on the r_i
        # This follows the idea of neuro evolution where successful policies are recorded more often and bad policies are recorded less often
        for _ in range(record_amount):
            self.kb.add_record(RolloutRecord(
                goal_space_id=goal.goal_id.value,
                theta=neuro_policy.theta,
                fitness=self.rollout_fitness,
                intrinsic_reward=r_i,
                exploit=exploit,
                rollout_idx=rollout_idx,
            ))
        self.update_goal_space_emas()

import random
from typing import List

from base_dir.hyper_parameters import AgentConfig, GREEDY
from base_dir.proto1.goal_policy import GoalSpaceEMA, select_goal, update_goal_space_ema
from base_dir.proto1.knowledge_base import KnowledgeBase, RolloutRecord
from base_dir.proto1.neuro_policy import GoalSpaceNeuroPolicy, get_neuro_policy
from base_dir.shared_files.goal_spaces import Goal, create_goal_spaces, reset_goal_spaces
from base_dir.shared_files.helpers.get_plan import get_plan
from base_dir.shared_files.helpers.goal_policy_input_vector import get_goal_policy_input_vector
from base_dir.shared_files.helpers.high_level_actions import HighLevelActions, get_motion_goals
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.planning.planners import MotionPlanner


class IMGEPAgent(Agent):
    def __init__(
            self,
            env: OvercookedEnv,
            mdp: OvercookedGridworld,
            agent_id: int,
            config: AgentConfig
    ):
        self.agent_id = agent_id
        self.env: OvercookedEnv = env
        self.mdp: OvercookedGridworld = mdp
        self.mp: MotionPlanner = env.mp
        self.mlam = env.mlam

        self.goal_spaces: List[Goal] = create_goal_spaces()
        self.kb = KnowledgeBase(len(self.goal_spaces), config=config)
        self.goal_space_emas: List[GoalSpaceEMA] = []
        self.config = config
        self.update_goal_space_emas()
        # Rollout bookkeeping
        self.previous_state = None
        self.goal_space_neuro_policies: List[GoalSpaceNeuroPolicy] = []
        self.t = 0
        self.goal_reach_time_step = None
        self.rollout_fitness = 0.0
        self.path = []

    def update_goal_space_emas(self):
        """
        Update the goal space EMA based on the knowledge base.
        """
        self.goal_space_emas = update_goal_space_ema(self.kb, self.goal_spaces, self.config)

    def reset(self, mdp: OvercookedGridworld | None = None, exploit: bool = False):
        super().reset()
        self.mdp = mdp
        self.t = 0
        self.path = []
        self.goal_reach_time_step = None
        self.rollout_fitness = 0.0
        self.previous_state = None
        reset_goal_spaces(self.goal_spaces)
        exploit = random.random() < self.config.exploit_prob
        chosen_goal, exploit = select_goal(self.goal_space_emas, goal_spaces=self.goal_spaces,
                                           exploit=exploit)

        neuro_policy = get_neuro_policy(selected_goal=chosen_goal, kb=self.kb, exploit=exploit,
                                        goal_spaces=self.goal_spaces, config=self.config)
        self.goal_space_neuro_policies = [
            GoalSpaceNeuroPolicy(goal=chosen_goal, neuro_policy=neuro_policy, exploit=exploit)
        ]  # Put the main neuro policy in the first position

    def action(self, state: OvercookedState) -> Action:
        if self.goal_reach_time_step:
            return Action.STAY
        self.t += 1

        gs = self.goal_space_neuro_policies[-1].goal
        exploit = self.goal_space_neuro_policies[-1].exploit

        if self.previous_state and gs.success(self.agent_id, state, self.previous_state,
                                              self.mdp) and self.goal_reach_time_step is None:
            if len(self.goal_space_neuro_policies) == 1:
                self.goal_reach_time_step = self.t
            else:
                self.goal_space_neuro_policies.pop()

        if self.t == self.goal_reach_time_step:
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

        obs_vec = get_goal_policy_input_vector(state, self.mdp, self.mp, self.agent_id)

        # ----------------- high-level action selection ----------------------
        greedy = GREEDY or exploit
        neuro_policy_token = self.goal_space_neuro_policies[-1].neuro_policy.select_token(obs_vec, greedy=greedy)
        if neuro_policy_token >= len(HighLevelActions):
            selected_goal = self.goal_spaces[neuro_policy_token - len(HighLevelActions)]
            # If the chosen goal is itself or the fitness is not high enough, do not use the decision to use this
            rec = self.kb.nearest(goal=selected_goal, k=1, exploit=True)

            if rec and rec[0].fitness >= self.config.minimum_goal_fitness and selected_goal != gs:
                self.goal_space_neuro_policies.append(
                    GoalSpaceNeuroPolicy(
                        goal=selected_goal,
                        neuro_policy=get_neuro_policy(
                            selected_goal=selected_goal,
                            kb=self.kb,
                            exploit=True,
                            goal_spaces=self.goal_spaces,
                            config=self.config
                        ),
                        exploit=True
                    )
                )
            neuro_policy_token = self.goal_space_neuro_policies[-1].neuro_policy.select_token(obs_vec, greedy=greedy)

        if neuro_policy_token < len(HighLevelActions):
            # If the token is a high-level action, we need to get the motion goals for that action
            token = HighLevelActions(neuro_policy_token)
            if token != HighLevelActions.WAIT:
                self.rollout_fitness -= 0.01
            motion_goals = get_motion_goals(self.mlam, self.mdp, token, state)
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
        return action

    def finish_rollout(self, info, partner_goal_id):
        # intrinsic reward = Δ fitness vs nearest prior experiment
        goal = self.goal_space_neuro_policies[0].goal  # The first goal is the main goal space
        neuro_policy = self.goal_space_neuro_policies[0].neuro_policy
        exploit = self.goal_space_neuro_policies[0].exploit
        prev_record = self.kb.nearest(goal=goal, k=self.config.ir_avg_prev_records, exploit=exploit)
        # Calculate avarage fitness of the previous records
        prev_f = sum(r.fitness for r in prev_record) / len(prev_record) if prev_record else 0.0
        rollout_fitness = max(0.0, self.rollout_fitness)
        # rollout_fitness += partner_fitness / 4
        fitness_difference = rollout_fitness - prev_f
        r_i = max(0.0, fitness_difference)
        # r_i =fitness_difference
        record_amount = max(1, int(r_i * self.config.neuro_evolution_multiplier))

        # Get the previous rollout index from the kb and increment it
        prev_record = self.kb.nearest(1)
        rollout_idx = prev_record[0].rollout_idx + 1 if prev_record else 0

        # Save the number of records in the knowledge base based on the r_i
        # This follows the idea of neuro evolution where successful policies are recorded more often and bad policies are recorded less often
        for _ in range(record_amount):
            self.kb.add_record(RolloutRecord(
                goal_space_id=goal.goal_id.value,
                theta=neuro_policy.theta,
                fitness=rollout_fitness,
                intrinsic_reward=r_i,
                exploit=exploit,
                rollout_idx=rollout_idx,
                partner_goal_id=partner_goal_id,
            ))
        self.update_goal_space_emas()

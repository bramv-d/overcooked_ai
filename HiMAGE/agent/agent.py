from typing import List

from HiMAGE.agent.goals.goal_policy_input_vector import get_goal_policy_input_vector
from HiMAGE.agent.goals.goal_spaces import Goal, GoalSpaceEnum, create_goal_spaces, reset_goal_spaces
from HiMAGE.agent.helpers.action_policy_input_vector import get_action_policy_input_vector
from HiMAGE.agent.helpers.get_plan import get_plan
from HiMAGE.agent.knowledge_base import KnowledgeBase, RolloutRecord
from HiMAGE.agent.neuro_policy.high_level_actions import get_motion_goals
from HiMAGE.agent.neuro_policy.neuro_policy import ActionPolicy, NeuroPolicy, get_action_policy, get_goal_policy
from HiMAGE.hyper_parameters import ACTION_POLICY_NEUROEVOLUTION_MULTIPLIER
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState
from overcooked_ai_py.planning.planners import MotionPlanner


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

        # Rollout bookkeeping
        self.fitness = 0.0
        self.previous_state = None
        self.goal_policy: NeuroPolicy | None = None
        self.action_policy: ActionPolicy | None = None
        self.t = 0
        self.action_policy_start_time = 0
        self.path = []  # This will hold the path of actions to take
        self.rollout_idx = 0
        self.exploit = False
        super().__init__()

    def reset(self, mdp: OvercookedGridworld | None = None, rollout_idx: int = 0, exploit: bool = False):
        super().reset()
        self.rollout_idx = rollout_idx
        self.goal_policy = get_goal_policy(self.kb, self.goal_spaces, exploit)
        self.mdp = mdp
        self.t = 0
        self.fitness = 0.0
        self.action_policy = None
        self.action_policy_start_time = 0
        self.path = []
        self.exploit = exploit
        self.previous_state = None
        reset_goal_spaces(self.goal_spaces)

    def action(self, state: OvercookedState) -> Action:
        if self.path:
            action = self.path.pop(0)
            return self.return_action(action, state)

        if self.action_policy and self.action_policy.goal.success(self.agent_id, state, self.previous_state, self.mdp):
            # If the current action policy's goal is successful, reset the action policy
            self.add_action_policy_to_kb(state)
            self.action_policy = None

        if self.action_policy is None:
            goal_policy_input_vector = get_goal_policy_input_vector(state, self.mdp, self.mp, self.mlam, self.agent_id)
            output_token = self.goal_policy.select_token(goal_policy_input_vector, self.exploit)
            goal: Goal = self.goal_spaces[output_token]
            action_policy = get_action_policy(self.kb, self.exploit, goal.goal_id)
            self.action_policy = ActionPolicy(action_policy, goal)
            self.action_policy_start_time = self.t

        action_policy_input_vector = get_action_policy_input_vector(state, self.mdp, self.mp, self.mlam, self.agent_id)
        action_token = self.action_policy.neuro_policy.select_token(action_policy_input_vector, self.exploit)
        motion_goals = get_motion_goals(self.mlam, self.mdp, action_token, state)
        player: PlayerState = state.players[self.agent_id]
        self.path = get_plan(player.pos_and_or, motion_goals, self.mlam)
        if self.path:
            action = self.path.pop(0)
            return self.return_action(action, state)
        # Fallback
        return self.return_action(Action.STAY, state)


    def return_action(self, action, state):
        self.previous_state = state
        return action

    def finish_rollout(self, info):
        self.add_goal_policy_to_kb(info)

    def add_action_policy_to_kb(self, state) -> None:
        fitness = self.action_policy.goal.fitness(
            pick_step=self.t,
            start_time=self.action_policy_start_time,
            state=state,
            previous_state=self.previous_state,
            agent_id=self.agent_id,
            mdp=self.mdp
        )
        prev_record = self.kb.nearest(
            goal_id=self.action_policy.goal.goal_id,
            k=1,
            exploit=self.exploit
        )
        self.fitness += fitness
        prev_fitness = prev_record[0].fitness if prev_record else 0.0
        intrinsic_reward = max(0, fitness - prev_fitness)
        rec: RolloutRecord = RolloutRecord(
            goal_id=self.action_policy.goal.goal_id,
            fitness=fitness,
            intrinsic_reward=intrinsic_reward,
            theta=self.action_policy.neuro_policy.theta,
            exploit=self.exploit,
            rollout_idx=self.rollout_idx,
        )
        record_amount = int(max(1, intrinsic_reward * ACTION_POLICY_NEUROEVOLUTION_MULTIPLIER))
        for _ in range(record_amount):
            self.kb.add_record(rec)

    def add_goal_policy_to_kb(self, info) -> None:
        fitness = info['episode']['ep_sparse_r'] + info['episode']['ep_shaped_r'] + self.fitness
        fitness /= 100
        prev_r = self.kb.nearest(
            goal_id=GoalSpaceEnum.OVERALL_GOAL,
            k=1,
            exploit=self.exploit
        )
        intrinsic_reward = max(0, (fitness - prev_r[0].fitness if prev_r else 0.0))
        rec: RolloutRecord = RolloutRecord(
            goal_id=GoalSpaceEnum.OVERALL_GOAL,
            fitness=fitness,
            intrinsic_reward=intrinsic_reward,
            theta=self.goal_policy.theta,
            exploit=self.exploit,
            rollout_idx=self.rollout_idx,
        )
        record_amount = int(max(1, intrinsic_reward * ACTION_POLICY_NEUROEVOLUTION_MULTIPLIER))
        for _ in range(record_amount):
            self.kb.add_record(rec)

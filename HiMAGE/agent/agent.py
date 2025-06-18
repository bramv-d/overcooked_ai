from typing import List

from HiMAGE.agent.goals.goal_policy_input_vector import get_goal_policy_input_vector
from HiMAGE.agent.goals.goal_spaces import Goal, create_goal_spaces
from HiMAGE.agent.helpers.get_plan import get_plan
from HiMAGE.agent.helpers.obs_to_vect import get_action_policy_input_vector
from HiMAGE.agent.knowledge_base import KnowledgeBase, RolloutRecord
from HiMAGE.agent.neuro_policy.high_level_actions import HighLevelActions
from HiMAGE.agent.neuro_policy.neuro_policy import ActionPolicy, NeuroPolicy, get_action_policy, get_goal_policy
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
        self.action_policy_start_time = 0
        self.path = []
        self.exploit = exploit
        self.previous_state = None

    def action(self, state: OvercookedState) -> Action:
        if self.path:
            action = self.path.pop(0)
            return self.return_action(action, state)

        if self.action_policy and self.action_policy.goal.success(self.agent_id, state, self.previous_state, self.mdp):
            # If the current action policy's goal is successful, reset the action policy
            self.action_policy = None
            self.add_action_policy_to_kb(state)

        if self.action_policy is None:
            goal_policy_input_vector = get_goal_policy_input_vector(state, self.mdp, self.mp, self.agent_id)
            selected_goal_id = self.goal_policy.select_token(goal_policy_input_vector, self.exploit)
            action_policy = get_action_policy(self.kb, self.exploit, selected_goal_id)
            goal = self.goal_spaces[selected_goal_id]
            self.action_policy = ActionPolicy(action_policy, goal)
            self.action_policy_start_time = self.t

        action_policy_input_vector = get_action_policy_input_vector(state, self.mdp, self.mp, self.agent_id)
        action_token = self.action_policy.neuro_policy.select_token(action_policy_input_vector, self.exploit)
        motion_goals = HighLevelActions(action_token)
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

    def finish_rollout(self, final_state: OvercookedState, info):
        print("rollout finished")

    def add_action_policy_to_kb(self, state) -> None:
        fitness = self.action_policy.goal.fitness(
            pick_step=self.t,  # Assuming pick_step is not used in this context
            state=state,
            previous_state=self.previous_state,  # Assuming no previous state for initial action policy
            agent_id=self.agent_id,
            mdp=self.mdp
        )

        intrinsic_reward = 0.0  # Assuming intrinsic reward is not used in this context
        rec: RolloutRecord = RolloutRecord(
            goal_id=self.action_policy.goal.goal_id,
            fitness=fitness,
            intrinsic_reward=intrinsic_reward,
            theta=self.action_policy.neuro_policy.theta,
            exploit=self.exploit,
            rollout_idx=self.rollout_idx,
        )
        self.kb.add_record(rec)

import random

from IMGEP_agent.agent.helpers.get_plan import get_plan
from IMGEP_agent.agent.neuro_policy.high_level_actions import HighLevelActions, get_motion_goals
from IMGEP_agent.agent.neuro_policy.neuro_policy import GoalSpaceNeuroPolicy, get_neuro_policy
from IMGEP_agent.agent.neuro_policy.neuro_policy_input_vector import get_policy_input_vector
from IMGEP_agent.hyper_parameters import AgentConfig, GREEDY
from IMGEP_agent.shared_agent.goal_spaces import Goal
from IMGEP_agent.shared_agent.knowledge_base import KnowledgeBase
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

        self.config = config
        self.goal_space_neuro_policies: GoalSpaceNeuroPolicy | None = None
        self.path = []

    def reset(self, chosen_goal: Goal, mdp: OvercookedGridworld | None = None, exploit: bool = False,
              kb: KnowledgeBase | None = None):
        super().reset()
        self.mdp = mdp
        self.path = []
        neuro_policy = get_neuro_policy(selected_goal=chosen_goal, kb=kb, exploit=exploit,
                                        config=self.config, agent_id=self.agent_id)
        self.goal_space_neuro_policies = GoalSpaceNeuroPolicy(goal=chosen_goal, neuro_policy=neuro_policy,
                                                              exploit=exploit)

    def action(self, state: OvercookedState) -> Action:
        if self.path:
            step = self.path.pop(0)
            return self.return_action(step, state)

        obs_vec = get_policy_input_vector(state, self.mdp, self.mp, self.agent_id)

        # ----------------- high-level action selection ----------------------
        greedy = GREEDY or self.goal_space_neuro_policies.exploit
        neuro_policy_token = self.goal_space_neuro_policies.neuro_policy.select_token(obs_vec, greedy=greedy)
        token = HighLevelActions(neuro_policy_token)
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


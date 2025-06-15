import random
from typing import List

from IMGEP_agent.agent.goal_policy import select_goal_space
from IMGEP_agent.agent.goals.goal_spaces import Goal, create_goal_spaces
from IMGEP_agent.agent.knowledge_base import KnowledgeBase
from IMGEP_agent.agent.neuro_policy.neuro_policy import GoalSpaceNeuroPolicy
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
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

        self.goal_spaces: List[Goal] = create_goal_spaces()
        self.kb = KnowledgeBase()

        # Rollout bookkeeping
        self.previous_state = None
        self.goal_space_neuro_policies: List[
            GoalSpaceNeuroPolicy] = []  # List of neuro policies the 0th element being active

        self.path = []  # This will hold the path of actions to take
        super().__init__()

    def reset(self, mdp: OvercookedGridworld | None = None):
        super().reset()

        chosen_goal = select_goal_space(all_goal_spaces=self.goal_spaces, kb=self.kb)

        self.goal_space_neuro_policies = [
            GoalSpaceNeuroPolicy(goal=chosen_goal)
        ]

    def action(self, state):
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

    def finish_rollout(self, state, info):
        pass

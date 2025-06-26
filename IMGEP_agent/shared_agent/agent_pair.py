import random
from typing import List

from IMGEP_agent.agent.agent import IMGEPAgent
from IMGEP_agent.hyper_parameters import AgentConfig
from IMGEP_agent.shared_agent.goal_policy import GoalSpaceEMA, select_goal, update_goal_space_ema
from IMGEP_agent.shared_agent.goal_spaces import Goal, create_goal_spaces, reset_goal_spaces
from IMGEP_agent.shared_agent.knowledge_base import KnowledgeBase, RolloutRecord
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState


class AgentPair:
    def __init__(self,
                 env: OvercookedEnv,
                 mdp: OvercookedGridworld,
                 config: AgentConfig
                 ):
        self.agents = [IMGEPAgent(env, mdp, agent_id, config=config) for agent_id in range(2)]

        self.goal_spaces: List[Goal] = create_goal_spaces()
        self.kb = KnowledgeBase(len(self.goal_spaces), config=config)
        self.goal_space_ema: List[GoalSpaceEMA] = []
        self.config = config
        # Rollout variables
        self.mdp = mdp
        self.chosen_goal: Goal | None = None
        self.t = 0
        self.previous_state = None
        self.exploit = False
        self.goal_reach_time_step = None
        self.rollout_fitness = 0

    def load_kb(self):
        kb_path = f"agent/kb/buffer_rollout.npz"
        self.kb = KnowledgeBase.load_buffer(kb_path, config=self.config)
        self.goal_space_ema = update_goal_space_ema(self.kb, self.goal_spaces, self.config)

    def reset(self, mdp: OvercookedGridworld):
        reset_goal_spaces(self.goal_spaces)
        self.exploit = random.random() < self.config.exploit_prob
        self.mdp = mdp
        self.rollout_fitness = 0
        self.t = 0
        self.chosen_goal = select_goal(self.goal_space_ema, goal_spaces=self.goal_spaces, config=self.config)
        for agent in self.agents:
            agent.reset(chosen_goal=self.chosen_goal, mdp=mdp, exploit=self.exploit, kb=self.kb)

    def action(self, state: OvercookedState):
        self.t += 1
        joint = []
        if self.previous_state:
            self.rollout_fitness += self.chosen_goal.fitness(state, self.previous_state, self.mdp)
        for agent in self.agents:
            joint.append(agent.action(state))
        return self.return_action(joint, state)

    def return_action(self, joint, state):
        self.previous_state = state
        return joint

    def finish_rollout(self):
        prev_record = self.kb.nearest(goal=self.chosen_goal, k=self.config.ir_avg_prev_records, exploit=self.exploit)
        avg_prev_f = sum(r.fitness for r in prev_record) / len(prev_record) if prev_record else 0.0
        rollout_fitness = max(0.0, self.rollout_fitness)
        fitness_difference = rollout_fitness - avg_prev_f
        r_i = max(0.0, fitness_difference)
        record_amount = max(1, int(r_i * self.config.neuro_evolution_multiplier))
        prev_record = self.kb.nearest(1)
        rollout_idx = prev_record[0].rollout_idx + 1 if prev_record else 0
        if self.rollout_fitness:
            print(f"Rollout fitness: {self.rollout_fitness}, "
                  f"Avg prev fitness: {avg_prev_f}, "
                  f"Intrinsic reward: {r_i}, "
                  f"Exploit: {self.exploit}, "
                  f"Rollout idx: {rollout_idx}")
        for _ in range(record_amount):
            self.kb.add_record(RolloutRecord(
                goal_space_id=self.chosen_goal.goal_id.value,
                theta0=self.agents[0].goal_space_neuro_policies.neuro_policy.theta,
                theta1=self.agents[1].goal_space_neuro_policies.neuro_policy.theta,
                fitness=rollout_fitness,
                intrinsic_reward=r_i,
                exploit=self.exploit,
                rollout_idx=rollout_idx,
            ))
        self.goal_space_ema = update_goal_space_ema(self.kb, self.goal_spaces, self.config)

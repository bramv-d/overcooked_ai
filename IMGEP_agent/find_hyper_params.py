import copy
from typing import Any, Dict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

from IMGEP_agent.agent.agent import IMGEPAgent
from IMGEP_agent.agent.knowledge_base import KnowledgeBase
from IMGEP_agent.hyper_parameters import AgentConfig, HORIZON, LAYOUT_ID, LOAD_KB, ROLLOUTS
from overcooked_ai_py.data.layouts.layouts import layouts
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld


class AgentConfigEstimator(BaseEstimator):
    """
    A scikit-learn wrapper for IMGEPAgent training+evaluation.
    The hyperparameters are the fields of AgentConfig.
    """

    def __init__(self,
                 exploit_prob: float = 0.2,
                 neuro_policy_hidden_dim: int = 20,
                 adaptive_noise_std: float = 0.1,
                 parent_policy_recent: int = 50,
                 mutate_records: int = 64,
                 goal_ema_k: int = 50,
                 goal_greedy_prob: float = 0.8,
                 n_recent: int = 100,
                 ir_bonus_cap: float = 0.20,
                 ir_bonus_slope: float = 0.01,
                 random_state: int = 42):
        # These will be tuned by sklearn
        self.exploit_prob = exploit_prob
        self.neuro_policy_hidden_dim = neuro_policy_hidden_dim
        self.adaptive_noise_std = adaptive_noise_std
        self.parent_policy_recent = parent_policy_recent
        self.mutate_records = mutate_records
        self.goal_ema_k = goal_ema_k
        self.goal_greedy_prob = goal_greedy_prob
        self.n_recent = n_recent
        self.ir_bonus_cap = ir_bonus_cap
        self.ir_bonus_slope = ir_bonus_slope
        self.random_state = random_state

        self.best_score_ = None

    def fit(self, X=None, y=None):
        # 1) build the dataclass config from current params
        config = AgentConfig(
            exploit_prob=self.exploit_prob,
            neuro_policy_hidden_dim=self.neuro_policy_hidden_dim,
            adaptive_noise_std=self.adaptive_noise_std,
            parent_policy_recent=self.parent_policy_recent,
            mutate_records=self.mutate_records,
            n_recent=self.n_recent,
        )

        # 2) set up env + agents exactly as in your script
        layout_name = layouts[LAYOUT_ID]
        mdp = OvercookedGridworld.from_layout_name(layout_name)
        base_params = {
            "start_orientations": False,
            "wait_allowed": False,
            "counter_goals": mdp.terrain_pos_dict["X"],
            "counter_drop": mdp.terrain_pos_dict["X"],
            "counter_pickup": mdp.terrain_pos_dict["X"],
            "same_motion_goals": True,
        }
        env = OvercookedEnv.from_mdp(mdp,
                                     horizon=HORIZON,
                                     info_level=0,
                                     mlam_params=base_params)

        # instantiate agents
        rng = np.random.RandomState(self.random_state)
        agents = [IMGEPAgent(env, mdp, aid, config=config)
                  for aid in range(2)]

        # optionally load KB + initial EMA
        if LOAD_KB:
            for ag in agents:
                kb_path = f"agent/kb/b.buffer_rollouts{ag.agent_id}.npz"
                ag.kb = KnowledgeBase.load_buffer(kb_path, config=config)
                ag.update_goal_space_emas()

        # 3) run rollouts & collect a performance measure
        total_fitness = 0.0
        for roll in range(ROLLOUTS):
            env.reset(regen_mdp=True)
            exploit_flag = rng.rand() < config.exploit_prob

            # reset agents with partner state
            for ag in agents:
                partner = agents[1 - ag.agent_id]
                ag.reset(other_goal_space_emas=partner.goal_space_emas,
                         other_kb=partner.kb,
                         mdp=mdp,
                         exploit=exploit_flag)

            done = False
            state = env.state
            ep_states = [copy.deepcopy(state)]

            # simulate episode
            while not done:
                joint_actions = [ag.action(state) for ag in agents]
                state, _, done, info = env.step(joint_actions)
                ep_states.append(copy.deepcopy(state))

            # finish bookkeeping and tally fitness
            for idx, ag in enumerate(agents):
                partner = agents[1 - idx]
                ag.finish_rollout(info,
                                  partner.goal_space_neuro_policies[0].goal.goal_id,
                                  partner.rollout_fitness)
                total_fitness += partner.rollout_fitness

        # average fitness per rollout
        avg_fitness = total_fitness / (ROLLOUTS * 2)
        self.best_score_ = avg_fitness

        return self

    def score(self, X=None, y=None) -> float:
        # scikit-learn tries to maximize score
        return self.best_score_ if self.best_score_ is not None else -np.inf


# === USAGE WITH RANDOMIZEDSEARCHCV ===

param_distributions: Dict[str, Any] = {
    'exploit_prob': np.linspace(0.0, 1.0, 11),
    'neuro_policy_hidden_dim': [10, 20, 50, 100],
    'adaptive_noise_std': np.linspace(0.0, 0.5, 11),
    'parent_policy_recent': [10, 50, 100],
    'mutate_records': [20, 40, 80, 120],
    'n_recent': [30, 50, 80],
}
ps = PredefinedSplit(test_fold=[0])
optimizer = RandomizedSearchCV(
    estimator=AgentConfigEstimator(random_state=42),
    param_distributions=param_distributions,
    n_iter=20,
    cv=ps,
    scoring=None,  # uses estimator.score()
    random_state=42,
    verbose=2,
)

# run the hyperparameter search
optimizer.fit([[0]], [0])
print("Best avg. fitness:", optimizer.best_score_)
print("Best hyperparameters:\n", optimizer.best_params_)

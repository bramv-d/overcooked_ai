import copy
from typing import Any, Dict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

from IMGEP_agent.hyper_parameters import AgentConfig, HORIZON, LAYOUT_ID, ROLLOUTS
from IMGEP_agent.shared_agent.agent_pair import AgentPair
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
                 neuro_evolution_multiplier: int = 10,
                 n_recent: int = 100,
                 ir_avg_prev_records: int = 10,
                 random_state: int = 42):
        # These will be tuned by sklearn
        self.exploit_prob = exploit_prob
        self.neuro_policy_hidden_dim = neuro_policy_hidden_dim
        self.adaptive_noise_std = adaptive_noise_std
        self.parent_policy_recent = parent_policy_recent
        self.mutate_records = mutate_records
        self.neuro_evolution_multiplier = neuro_evolution_multiplier
        self.n_recent = n_recent
        self.ir_avg_prev_records = ir_avg_prev_records
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
            neuro_evolution_multiplier=self.neuro_evolution_multiplier,
            ir_avg_prev_records=self.ir_avg_prev_records,

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
        agent_pair = AgentPair(env, mdp, config)

        # 3) run rollouts & collect a performance measure
        total_fitness = 0.0
        for roll in range(ROLLOUTS):
            env.reset(regen_mdp=True)
            agent_pair.reset(mdp)
            done = False
            state = env.state

            # -------- record trajectory -----------------------------------------
            ep_states = [copy.deepcopy(state)]  # include start state
            while not done:
                joint = agent_pair.action(state)
                state, _, done, info = env.step(joint)
                ep_states.append(copy.deepcopy(state))  # save each next state

            # -------- finish roll-out bookkeeping -------------------------------
            agent_pair.finish_rollout()
            total_fitness += agent_pair.rollout_fitness

        # average fitness per rollout
        avg_fitness = total_fitness / ROLLOUTS
        self.best_score_ = avg_fitness

        return self

    def score(self, X=None, y=None) -> float:
        # scikit-learn tries to maximize score
        return self.best_score_ if self.best_score_ is not None else -np.inf


# === USAGE WITH RANDOMIZEDSEARCHCV ===

param_distributions: Dict[str, Any] = {
    'exploit_prob': np.linspace(0.0, 0.5, 20),
    'neuro_policy_hidden_dim': [50, 100, 125, 150, 200],
    'adaptive_noise_std': np.linspace(0.05, 0.3, 20),
    'parent_policy_recent': [40, 50, 60, 70, 80, 90, 100],
    'mutate_records': [5, 10, 20, 40],
    'n_recent': [25, 50, 75, 100],
    'neuro_evolution_multiplier': [2, 5, 10, 15],
    'ir_avg_prev_records': [5, 10, 20, 40],
}
ps = PredefinedSplit(test_fold=[0])
optimizer = RandomizedSearchCV(
    estimator=AgentConfigEstimator(random_state=42),
    param_distributions=param_distributions,
    n_iter=50,
    cv=ps,
    scoring=None,  # uses estimator.score()
    random_state=42,
    verbose=2,
)

# run the hyperparameter search
optimizer.fit([[0]], [0])
print("Best avg. fitness:", optimizer.best_score_)
print("Best hyperparameters:\n", optimizer.best_params_)

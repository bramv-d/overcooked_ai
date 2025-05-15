# imrl_agent_new/imgep_agent.py
from __future__ import annotations
import random, math, numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List

from overcooked_ai_py.agents.agent import Agent          # base class
from overcooked_ai_py.mdp.actions import Action          # primitive enum
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from imrl_agent_new.core.goal_policy      import GoalSpacePolicy
from imrl_agent_new.core.goal_spaces      import create_goal_space  # factory we wrote
from imrl_agent_new.core.knowledge_base   import KnowledgeBase, ExperimentRecord
from imrl_agent_new.overcooked.outcome import extract_outcome  # 5-slot outcome


@dataclass
class RolloutStats:
    goal_space : str
    goal_vec   : list[int]
    # score      : int
    dishes     : int
    fitness    : float
    intrinsic  : float
    steps      : int

# --------------------------------------------------------------------------- #
class IMGEPAgent(Agent):
    """
    A minimal Intrinsically-Motivated Goal-Exploration Agent.
    • chooses a goal with Γ at env-reset
    • random-walks until rollout ends or success predicate fires
    • logs outcome + intrinsic reward into KB
    """

    def __init__(self, grid_world: OvercookedGridworld, agent_id: int, horizon: int = 400, epsilon: float = 0.20):
        # 1) Core components
        self.last_stats = None
        self.horizon      = horizon
        self.G            = create_goal_space(grid_world, horizon)
        self.bandit       = GoalSpacePolicy(self.G, epsilon=epsilon)
        self.kb           = KnowledgeBase(context_dim=1, outcome_dim=5)  # simple ctxt
        self.mdp          = grid_world

        # 2) rollout-specific state (filled in reset())
        self.goal_space_id: str  | None = None
        self.goal_vec     : np.ndarray | None = None
        self.t            : int = 0
        self.meta         : Dict[str, Any] = {}
        self.agent_id     = agent_id
        super().__init__()

    @property
    def rollout_stats(self) -> RolloutStats:
        return self.last_stats

    # ------------------------------------------------------------------ Agent API
    def reset(self):
        """
        Called by Overcooked-AI at the beginning of each rollout.
        """
        # pick goal with Γ
        self.goal_space_id, self.goal_vec = self.bandit.next_goal(context=None)
        self.t       = 0
        self.meta    = {"reach_step": None,
                        "pick_step" : None,
                        "fill_step" : None}

    def action(self, state):
        """
        Returns (Action, info_dict).  Here we just random-walk.
        Replace with Πε / Π later.
        """
        self.t += 1

        # ---------- random primitive (stay, interact, N,E,S,W) --------------
        a = random.choice(list(Action.ALL_ACTIONS))
        return a, {}

    # ---------------------------------------------------------------- post-rollout
    def finish_rollout(self,
                       final_state: OvercookedState,
                       soups_delivered: int):



        """
        Call this once the environment signals done=True.
        Computes outcome, intrinsic reward and logs to KB + bandit.
        """
        # ------ extract outcome & compute fitness ---------------------------
        outcome = extract_outcome(final_state,
                                  final_state.players[self.agent_id],
                                  self.mdp,
                                  soups_delivered)

        gs      = self.G[self.goal_space_id]
        fitness = gs.fitness(outcome,
                             self.goal_vec,
                             **self.meta)

        # ------ intrinsic reward = LP vs nearest prior experiment ----------------
        if len(self.kb) == 0:  # first roll-out ever
            prev_f = 0.0
        else:
            idx, _ = self.kb.nearest(  # SAFE to call now
                context=np.array([0]),  # trivial context
                outcome=outcome
            )
            prev_f = self.kb.buffer[idx].fitness  # idx is a Python int

        r_i = fitness - prev_f

        self.last_stats = RolloutStats(
            goal_space=self.goal_space_id,
            goal_vec=self.goal_vec.tolist(),
            # score=final_state.score, TODO potentially add the score again
            dishes=soups_delivered,
            fitness=fitness,
            intrinsic=r_i,
            steps=self.t
        )
        # ------ record ------------------------------------------------------
        rec = ExperimentRecord(
            context          = np.array([0]),
            goal             = self.goal_vec,
            theta            = np.zeros(1),          # placeholder θ
            outcome          = outcome,
            fitness          = fitness,
            intrinsic_reward = r_i,
            trajectory       = []                    # not saved in this skeleton
        )
        self.kb.add_record(rec)

        # ------ update Γ if this was an exploitation run --------------------
        # (Here every rollout is “exploitation” for simplicity)
        self.bandit.update(self.goal_space_id, r_i)

    # ---------------------------------------------------------------- utility
    def save_kb(self, path: str):
        """Persist the KB as npz (tiny helper)."""
        np.savez_compressed(path,
                            outcomes=np.stack([r.outcome for r in self.kb.buffer]),
                            fitness =np.array([r.fitness for r in self.kb.buffer]),
                            goals   =np.stack([r.goal    for r in self.kb.buffer]))

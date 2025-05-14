# imrl_agent_new/imgep_agent.py
from __future__ import annotations
import random, math, numpy as np
from typing import Dict, Any, Tuple, List

from overcooked_ai_py.agents.agent import Agent          # base class
from overcooked_ai_py.mdp.actions import Action          # primitive enum
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from imrl_agent_new.core.goal_policy      import GoalSpacePolicy
from imrl_agent_new.core.goal_spaces      import create_goal_space  # factory we wrote
from imrl_agent_new.core.knowledge_base   import KnowledgeBase, ExperimentRecord
from imrl_agent_new.overcooked.outcome import extract_outcome  # 5-slot outcome


# --------------------------------------------------------------------------- #
class IMGEPAgent(Agent):
    """
    A minimal Intrinsically-Motivated Goal-Exploration Agent.
    • chooses a goal with Γ at env-reset
    • random-walks until episode ends or success predicate fires
    • logs outcome + intrinsic reward into KB
    """

    def __init__(self, grid_world: OvercookedGridworld, agent_id, horizon: int = 400, epsilon: float = 0.20):
        # 1) Core components
        self.horizon      = horizon
        self.G            = create_goal_space(grid_world, horizon)
        self.bandit       = GoalSpacePolicy(self.G, epsilon=epsilon)
        self.kb           = KnowledgeBase(context_dim=1, outcome_dim=5)  # simple ctxt

        # 2) Episode-specific state (filled in reset())
        self.goal_space_id: str  | None = None
        self.goal_vec     : np.ndarray | None = None
        self.t            : int = 0
        self.meta         : Dict[str, Any] = {}
        super().__init__()

    # ------------------------------------------------------------------ Agent API
    def reset(self):
        """
        Called by Overcooked-AI at the beginning of each episode.
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

    # ---------------------------------------------------------------- post-episode
    def finish_episode(self,
                       final_state,
                       soups_delivered: int):
        """
        Call this once the environment signals done=True.
        Computes outcome, intrinsic reward and logs to KB + bandit.
        """
        # ------ extract outcome & compute fitness ---------------------------
        outcome = extract_outcome(final_state,
                                  final_state.players[0],
                                  final_state.mdp,
                                  soups_delivered)

        gs      = self.G[self.goal_space_id]
        fitness = gs.fitness(outcome,
                             self.goal_vec,
                             **self.meta)

        # ------ intrinsic reward = LP vs nearest prior experiment -----------
        idx, _dist = self.kb.nearest(context=np.array([0]),   # trivial ctxt
                                     outcome=outcome)
        prev_f = self.kb.buffer[idx].fitness if idx is not None else 0.0
        r_i    = fitness - prev_f

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
        # (Here every episode is “exploitation” for simplicity)
        self.bandit.update(self.goal_space_id, r_i)

    # ---------------------------------------------------------------- utility
    def save_kb(self, path: str):
        """Persist the KB as npz (tiny helper)."""
        np.savez_compressed(path,
                            outcomes=np.stack([r.outcome for r in self.kb.buffer]),
                            fitness =np.array([r.fitness for r in self.kb.buffer]),
                            goals   =np.stack([r.goal    for r in self.kb.buffer]))

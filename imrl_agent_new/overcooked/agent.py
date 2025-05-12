"""
ModularIMGEPAgent
-----------------
A *very lightweight* agent that:

1. Chooses a goal-space module index with Γ.
2. Samples θ from the exploration policy Π′.
3. Interprets θ as a linear   (actions × obs_dim)   weight matrix.
4. Outputs a motion action each step.

This version is intentionally simple — it does **not** yet use concrete
goal objects; that will be added later, but it is sufficient to generate
rollouts and exercise the memory / intrinsic reward pipeline.
"""

from __future__ import annotations
from typing import Any, Sequence

import numpy as np
from overcooked_ai_py.agents.agent import Agent
from overcooked_ai_py.mdp.actions import Action

from imrl_agent_new.core.memory import EpisodicMemory
from imrl_agent_new.core.exploration_policy import LinearSamplingPolicy
from imrl_agent_new.core.goal_policy import EpsilonGreedyBandit


class ModularIMGEPAgent(Agent):
    def __init__(self, role_id: int, shared_E: EpisodicMemory, Γ: EpsilonGreedyBandit, Π_prime: LinearSamplingPolicy,
                 all_actions: bool = False):
        super().__init__()
        self.role = role_id
        self.E = shared_E
        self.Γ = Γ
        self.Pi_prime = Π_prime
        self.all_actions = all_actions

        # filled on reset
        self.context: str | None = None
        self.theta: np.ndarray | None = None
        self._w: np.ndarray | None = None

        # embedding size (p0x, p0y, p1x, p1y) + 16 zeros padding
        self.obs_dim = 20

    # ------------------------------------------------------------------ #
    def reset_episode(self, context: str):
        """Call once per episode before stepping."""
        self.context = context

        # ---- (a) choose module (we ignore the concrete goal for now) --- #
        module_id = self.Γ.select([0, 1, 2])

        # ---- (b) sample θ --------------------------------------------- #
        self.theta = self.Pi_prime.sample(goal=module_id, context=context)

        # ---- (c) map θ → weight matrix -------------------------------- #
        self.legal_actions: Sequence[str] = (
            Action.ALL_ACTIONS if self.all_actions else Action.MOTION_ACTIONS
        )
        act_dim = len(self.legal_actions)
        need = act_dim * self.obs_dim
        if len(self.theta) < need:
            self.theta = np.pad(self.theta, (0, need - len(self.theta)))
        elif len(self.theta) > need:
            self.theta = self.theta[:need]
        self._w = self.theta.reshape(act_dim, self.obs_dim)

    # Overcooked-AI calls this
    def reset(self):
        if self.context is None:
            raise RuntimeError("reset_episode(context) must be called first")

    # ------------------------------------------------------------------ #
    def _encode_state(self, state) -> np.ndarray:
        p0, p1 = state.players
        return np.array([*p0.position, *p1.position] + [0] * 16, dtype=np.float32)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        z = x - np.max(x)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()

    # ------------------------------------------------------------------ #
    def action(self, state) -> tuple[str, dict[str, Any]]:
        if self._w is None:
            # fallback (shouldn’t happen in proper driver)
            self.reset_episode(context="unknown")

        logits = self._w @ self._encode_state(state)
        probs = self._softmax(logits)
        idx = np.random.choice(len(probs), p=probs)
        return self.legal_actions[idx], {"action_probs": probs}

    def actions(self, states, agent_indices):
        return [self.action(s) for s in states]

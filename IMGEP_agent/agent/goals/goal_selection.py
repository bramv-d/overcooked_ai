"""goal_selection.py

Utility to choose collaborative goals for two agents given their goalspaces.

A *goalspace* is a list[dict] where each dict at least contains:
    id:       any hashable identifier
    ema:      float        # Exponential moving average of intrinsic reward / competence progress
    embed:    np.ndarray   # Continuous embedding of the goal for compatibility computation
Optional keys are ignored by the algorithm.

Algorithm (perfect‑info prototype):
1. Predict partner's goal as the one with highest EMA in their space.
2. For each candidate g in own space compute
       score = zscore(ema_g)  +  delta * phi(embed_g, embed_partner_pred)
   where phi = -cosine(embed_g, embed_partner_pred) to favour complementary goals.
3. Select argmax score.  (temperature‑controlled softmax sampling optional).
4. Repeat symmetrically for the other agent so the choice is simultaneous.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return standardised array (zero‑mean, unit‑var)."""
    mu, sig = x.mean(), x.std()
    return (x - mu) / (sig + eps)


def _cosine(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    """Cosine similarity of two 1‑D vectors."""
    num = (a * b).sum()
    den = np.linalg.norm(a) * np.linalg.norm(b) + eps
    return num / den


def choose_goal_pair(
        g1: List[Dict],
        g2: List[Dict],
        delta: float = 0.1,
        tau: float = 0.0,
) -> Tuple[Dict, Dict]:
    """Return selected goals (g*_1, g*_2) for Agent‑1 and Agent‑2.

    Parameters
    ----------
    g1, g2 : list of goal dicts
        Each dict needs keys 'ema' and 'embed'.
    delta : float
        Coordination trade‑off weight.
    tau : float
        Boltzmann temperature; 0 ⇒ greedy argmax.
    """
    # Build numpy arrays for EMA and embeddings
    ema1 = np.array([g['ema'] for g in g1], dtype=np.float32)
    ema2 = np.array([g['ema'] for g in g2], dtype=np.float32)
    E1 = np.stack([g['embed'] for g in g1])
    E2 = np.stack([g['embed'] for g in g2])

    # Predict each other's goal as max‑EMA in partner space
    idx2_hat = int(ema2.argmax())
    idx1_hat = int(ema1.argmax())
    e2_hat = E2[idx2_hat]
    e1_hat = E1[idx1_hat]

    # Compute compatibility φ = −cosine
    phi1 = -np.array([_cosine(e, e2_hat) for e in E1])
    phi2 = -np.array([_cosine(e, e1_hat) for e in E2])

    # Scores
    scores1 = _zscore(ema1) + delta * phi1
    scores2 = _zscore(ema2) + delta * phi2

    if tau > 0.0:
        # Soft selection
        probs1 = np.exp(scores1 / tau);
        probs1 /= probs1.sum()
        probs2 = np.exp(scores2 / tau);
        probs2 /= probs2.sum()
        idx1 = int(np.random.choice(len(g1), p=probs1))
        idx2 = int(np.random.choice(len(g2), p=probs2))
    else:
        idx1 = int(scores1.argmax())
        idx2 = int(scores2.argmax())

    return g1[idx1], g2[idx2]

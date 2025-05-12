# knowledge_base.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import datetime as dt
import numpy as np
from sklearn.neighbors import KDTree   # pip install scikit-learn

# ---------- 1. A single experiment record ----------
@dataclass
class ExperimentRecord:
    # High-level information
    context:  np.ndarray                 # shape = (C,)
    goal:     np.ndarray                 # shape = (G,)
    theta:    np.ndarray                 # policy params you executed
    outcome:  np.ndarray                 # shape = (O,)  derived from τ
    fitness:  float
    intrinsic_reward: float

    # Optional extras (kept for replay / analysis)
    trajectory: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    timestamp:  dt.datetime = field(default_factory=dt.datetime.utcnow)

# ---------- 2. Running mean helper ----------
class RunningMean:
    def __init__(self):
        self.n = 0
        self.mean = 0.0
    def update(self, x: float):
        self.n += 1
        self.mean += (x - self.mean) / self.n

# ---------- 3. Knowledge base ----------
class KnowledgeBase:
    """
    In-memory buffer + KD-Tree index for nearest-neighbour queries on (context, outcome).
    """
    def __init__(self, context_dim: int, outcome_dim: int):
        self.buffer: List[ExperimentRecord] = []
        # Start KD-Tree with one dummy vector to avoid empty-tree edge cases
        self._index_data = np.zeros((1, context_dim + outcome_dim))
        self._kdtree = KDTree(self._index_data)
        self._index_dirty = False

        self.stats: Dict[str, RunningMean] = {}    # keyed by goal-space or goal ID
        self.metadata: Dict[str, Any] = {
            "created": dt.datetime.utcnow().isoformat(),
            "context_dim": context_dim,
            "outcome_dim": outcome_dim,
        }

    # ---- public API ---------------------------------------------------------

    def add_record(self, rec: ExperimentRecord):
        """Insert a new experiment and flag index for rebuild."""
        self.buffer.append(rec)
        vec = np.concatenate([rec.context, rec.outcome])
        self._index_data = np.vstack([self._index_data, vec])
        self._index_dirty = True

    def nearest(self, context: np.ndarray, outcome: np.ndarray, k: int = 1):
        """Return indices of k most similar past experiments."""
        if self._index_dirty:
            self._kdtree = KDTree(self._index_data[1:])  # skip first dummy row
            self._index_dirty = False
        query = np.concatenate([context, outcome]).reshape(1, -1)
        dist, idx = self._kdtree.query(query, k=k)
        return idx[0], dist[0]

    # ---- convenience helpers -----------------------------------------------

    def update_stat(self, key: str, value: float):
        if key not in self.stats:
            self.stats[key] = RunningMean()
        self.stats[key].update(value)

    def __len__(self):
        return len(self.buffer)


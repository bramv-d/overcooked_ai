from __future__ import annotations

import datetime as dt
import pickle
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


# ---------- 1. A single experiment record ----------
@dataclass
class ExperimentRecord:
    # High-level information
    context: np.ndarray  # shape = (C,), holds the context in which the policy was used
    goal: int  # shape = (G,), the goal to reach using the policy
    goal_space: str  # goal space ID, e.g., "NAV", "PICK_OBJECT", etc.
    theta: np.ndarray  # policy params you executed
    outcome: np.ndarray  # shape = (O,), derived from τ, the outcome of the performed policy
    fitness: float
    intrinsic_reward: float
    exploit: bool = False  # True if this was an exploit step, False if it was exploration

    # Optional extras (kept for replay/analysis)
    trajectory: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    timestamp: dt.datetime = field(default_factory=dt.datetime.utcnow)

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

    def __init__(self):
        self.buffer: List[ExperimentRecord] = []

    # ---- public API ---------------------------------------------------------

    def add_record(self, rec: ExperimentRecord):
        """Insert a new experiment and flag index for rebuild."""
        self.buffer.append(rec)

    def nearest(self, goal_space: str, goal_vector_id: int, k: int, exploit: bool | None = None) -> List[
                                                                                                        ExperimentRecord] | None:
        """
        Return the db_ids(s) of the most similar past experiment.
        """
        nearest = []
        for index, g in enumerate(reversed(self.buffer)):  # Iterate in reverse order
            if g.goal_space == goal_space and g.goal == goal_vector_id and (exploit is None or g.exploit == exploit):
                nearest.append(g)
            if len(nearest) == k:
                break
        return nearest

    def __len__(self):
        return len(self.buffer)

    # --- save & load ------------------------------------------------------------
    def save_buffer(self, path: str):
        """Save the full buffer to disk as a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, path: str):
        with open(path, "rb") as f:
            self.buffer = pickle.load(f)

    def cap_records(self, goal_space_id: str, max_records: int = 1000):
        """Cap the records for the given goal space to max_records by removing the oldest items."""
        # Filter records for the given goal space
        filtered_records = [record for record in self.kb.buffer if record.goal_space == goal_space_id]
        excess_count = len(filtered_records) - max_records

        if excess_count > 0:
            # Sort records by timestamp (oldest first)
            filtered_records.sort(key=lambda record: record.timestamp)
            records_to_remove = filtered_records[:excess_count]

            # Remove the records from the buffer using db_id
            records_to_remove_ids = {record.db_id for record in records_to_remove}
            self.buffer = [record for record in self.buffer if record.db_id not in records_to_remove_ids]
            self._index_data = True

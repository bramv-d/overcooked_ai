import pickle
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class RolloutRecord:
    goal_id: int  # goal space ID
    theta: np.ndarray  # policy params you executed
    fitness: float
    intrinsic_reward: float
    exploit: bool = False  # True if this was an exploit step, False if it was exploration
    rollout_idx: int = 0  # episode number, used to track the order of experiments


class KnowledgeBase:
    """
    In-memory buffer + KD-Tree index for nearest-neighbour queries on (context, outcome).
    """

    def __init__(self):
        self.buffer: List[RolloutRecord] = []

    # ---- public API ---------------------------------------------------------

    def add_record(self, rec: RolloutRecord):
        self.buffer.append(rec)
        # With adding records to the db also remove the oldest records if the buffer exceeds a certain size


    def nearest(self, goal_id: int, k: int, exploit: bool | None = None) -> List[RolloutRecord] | None:
        """
        Return the db_ids(s) of the most similar past experiment.
        """
        nearest = []
        for index, g in enumerate(reversed(self.buffer)):  # Iterate in reverse order
            if g.goal_id == goal_id and (exploit is None or g.exploit == exploit):
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

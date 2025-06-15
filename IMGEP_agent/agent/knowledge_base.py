import pickle
from dataclasses import dataclass
from typing import List

import numpy as np

from IMGEP_agent.agent.goals.goal_spaces import Goal


@dataclass
class RolloutRecord:
    goal_space_id: int  # goal space ID, e.g., "PLACE_OBJECT", "PICK_OBJECT", etc.
    theta: np.ndarray  # policy params you executed
    fitness: float
    intrinsic_reward: float
    shared_episode_reward: float  # shared reward for the episode, used for multi-agent
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
        """Insert a new experiment and flag index for rebuild."""
        self.buffer.append(rec)

    def nearest(self, goal: Goal, k: int, exploit: bool | None = None) -> List[RolloutRecord] | None:
        """
        Return the db_ids(s) of the most similar past experiment.
        """
        nearest = []
        for index, g in enumerate(reversed(self.buffer)):  # Iterate in reverse order
            if g.goal_space_id == goal.goal_id and (exploit is None or g.exploit == exploit):
                nearest.append(g)
            if len(nearest) == k:
                break
        return nearest

    def __len__(self):
        return len(self.buffer)

    def well_performing_policies(self, fitness_threshold, record_amount, exploit) -> List[RolloutRecord]:
        """
        Return a list of records with fitness above the threshold
        """
        well_performing = []
        for record in reversed(self.buffer):
            if record.fitness >= fitness_threshold and record.exploit == exploit:
                well_performing.append(record)
            if len(well_performing) == record_amount:
                break
        return well_performing

    # --- save & load ------------------------------------------------------------
    def save_buffer(self, path: str):
        """Save the full buffer to disk as a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, path: str):
        with open(path, "rb") as f:
            self.buffer = pickle.load(f)

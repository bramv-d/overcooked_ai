import json
import os
import random
from typing import Dict, Tuple

from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.data.layouts.layouts import layouts
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.layout_generator import TYPE_TO_CODE
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState
from overcooked_ai_py.planning.planners import MediumLevelActionManager, MotionPlanner, NO_COUNTERS_PARAMS
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer


class InteractionMemoryEntry:
    def __init__(self, action, environment_type, agent_obj_before, env_obj_before,
                 agent_obj_after, env_obj_after):
        self.action = action
        self.environment_type = environment_type
        self.agent_obj_before = agent_obj_before
        self.env_obj_before = env_obj_before
        self.agent_obj_after = agent_obj_after
        self.env_obj_after = env_obj_after
        self.counter = 1  # Start at 1 for the first observed instance

    def increment(self):
        self.counter += 1

    def to_dict(self):
        return {
            "action": str(self.action),
            "environment_type": self.environment_type,
            "before_action": {
                "agent_obj": self.agent_obj_before,
                "env_obj": self.env_obj_before
            },
            "after_action": {
                "agent_obj": self.agent_obj_after,
                "env_obj": self.env_obj_after
            },
            "counter": self.counter
        }

    def __eq__(self, other):
        return (
                self.action == other.action and
                self.environment_type == other.environment_type and
                self.agent_obj_before == other.agent_obj_before and
                self.env_obj_before == other.env_obj_before and
                self.agent_obj_after == other.agent_obj_after and
                self.env_obj_after == other.env_obj_after
        )

    def __hash__(self):
        return hash((
            self.action,
            self.environment_type,
            self.agent_obj_before,
            self.env_obj_before,
            self.agent_obj_after,
            self.env_obj_after
        ))


class IMRLAgent(Agent):
    def __init__(self, layout, player_id, sim_threads=None):
        super().__init__()
        self.mdp: OvercookedGridworld = OvercookedGridworld.from_layout_name(layout)
        self.motion_planner = MotionPlanner(self.mdp)
        self.medium_level_action_manager = MediumLevelActionManager(self.mdp, NO_COUNTERS_PARAMS)
        self.sim_threads = sim_threads
        self.player_id = player_id
        self.prev_state = None
        self.prev_action = None
        # Memory: stores coupling → list of observed outcomes
        self.memory: Dict[InteractionMemoryEntry, InteractionMemoryEntry] = {}
        self.plan = None

    def get_player(self, state: OvercookedState) -> PlayerState:
        """
        Get the player state from the OvercookedState.
        """
        return state.players[self.player_id]

    def action(self, state: OvercookedState) -> Tuple[Action, dict]:
        possible_actions = self.mdp.get_actions(state)[self.player_id]

        # Randomly select one of the possible actions
        action = random.choice(possible_actions)

        if self.prev_action == Action.INTERACT:
            entry = self.create_interaction_memory_entry(self.prev_action, self.prev_state, state)

            if entry in self.memory:
                self.memory[entry].increment()
            else:
                self.memory[entry] = entry

        # Check if the action is already in the memory
        self.prev_state = state
        self.prev_action = action

        return action, {}

    def save_memory_to_json(self, base_dir: str, filename: str = "memory.json"):
        os.makedirs(base_dir, exist_ok=True)
        filepath = os.path.join(base_dir, filename)

        # Convert each memory entry to a serializable dict
        serializable_memory = [entry.to_dict() for entry in self.memory.values()]

        # Save to JSON
        with open(filepath, "w") as f:
            json.dump(serializable_memory, f, indent=2)

        print(f"Memory saved to {filepath}")

    def create_interaction_memory_entry(
            self,
            prev_action: Action,
            prev_state: OvercookedState,
            current_state: OvercookedState
    ) -> InteractionMemoryEntry:
        """
        Create a memory entry representing an interaction the agent took between two states.
        """

        # Get player states
        prev_player = self.get_player(prev_state)
        curr_player = self.get_player(current_state)

        # Get the position the agent is facing
        facing_cell = Action.move_in_direction(prev_player.pos_and_or[0], prev_player.pos_and_or[1])
        terrain_type = self.mdp.get_terrain_type_at_pos(facing_cell)
        environment_type_code = TYPE_TO_CODE[terrain_type]

        # ---- Before interaction ----
        env_obj_before = prev_state.get_object(facing_cell).name if prev_state.has_object(facing_cell) else None
        agent_obj_before = prev_player.held_object.name if prev_player.held_object else None

        # ---- After interaction ----
        env_obj_after = current_state.get_object(facing_cell).name if current_state.has_object(facing_cell) else None
        agent_obj_after = curr_player.held_object.name if curr_player.held_object else None

        # ---- Create InteractionMemoryEntry ----
        return InteractionMemoryEntry(
            action=prev_action,
            environment_type=environment_type_code,
            agent_obj_before=agent_obj_before,
            env_obj_before=env_obj_before,
            agent_obj_after=agent_obj_after,
            env_obj_after=env_obj_after
        )

    def actions(self, states, agent_indices):
        return [self.action(state) for state in states]


def evaluate_agent_pair(agent_pair, layout):
    """
    Evaluate a pair of agents on the given layout.
    """
    ae = AgentEvaluator.from_layout_name(mdp_params={"layout_name": layout, "old_dynamics": False},
                                         env_params={"horizon": 500})  # Set horizon to 5
    return ae.evaluate_agent_pair(agent_pair, 1)  # Run 5 games


# Make example usage
if __name__ == "__main__":
    layout_name = layouts[20]  # Choose a layout from the list of layouts
    agent1 = IMRLAgent(layout=layout_name, player_id=0)
    agent2 = IMRLAgent(layout=layout_name, player_id=1)
    # Evaluate the agent pair
    results = evaluate_agent_pair(AgentPair(agent1, agent2), layout_name)

    # Desired base directory
    base_dir = "/Users/bram/Documents/Afstuderen/images/trajectories"

    StateVisualizer().display_rendered_trajectory(results, img_directory_path=base_dir, ipython_display=False)
    agent1.save_memory_to_json(base_dir, filename="agent1_memory.json")
    agent2.save_memory_to_json(base_dir, filename="agent2_memory.json")

# Next steps
# 1. How to measure the change in state? How to measure the effect of an action?
# 2. How to transfer the effect of the agent towards the competence?
# 3. Progress is the difference between current and old competence
# 4. Based on the progress model, the interest in the action.
# 5. How to make the agent learn based on its own actions and the changes it provokes?
# 6. How to determine whether the agent should explore new actions or continue learning the current action?

# Eventual idea
# Feed the important state properties to the agent
# Let it check with which action it can provoke the biggest change in the state
# Perform the action with the biggest change and save this in some way

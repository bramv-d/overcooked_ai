import json
import os
import random
from typing import Any, Dict, Tuple

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

    def matches_pre_action(self, other) -> bool:
        return (
                self.action == other.action and
                self.environment_type == other.environment_type and
                self.agent_obj_before == other.agent_obj_before and
                self.env_obj_before == other.env_obj_before
        )

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
        self.plan = None  # The plan that was already constructed in a previous decision. This improves performance significantly.
        self.recent_positions = []

    def get_player(self, state: OvercookedState) -> PlayerState:
        """
        Get the player state from the OvercookedState.
        """
        return state.players[self.player_id]

    def action(self, state: OvercookedState) -> Tuple[Action, dict]:
        """
        FIRST, THE PREVIOUS STATE ACTION IS PROCESSED INTO MEMORY
        """
        player = self.get_player(state)
        if self.prev_action == Action.INTERACT:
            entry = self.create_interaction_memory_entry(self.prev_action, self.prev_state, state)

            if entry in self.memory:
                self.memory[entry].increment()
            else:
                self.memory[entry] = entry

        self.recent_positions.append(player.position)
        self.recent_positions.pop(0)
        """
        SECOND, THE ACTION IS GENERATED EITHER BY AN ACTION PLAN OR BY A CURIOUS INTERACTION
        """
        action = self.get_curious_interact_action(
            state)  # If the agent coincidentally has an adjacent interesting tile, it will interact with it.
        if action is None:
            self.plan = self.get_action_plan_from_memory(state) if self.plan is None else self.plan
            action = self.plan[0] if self.plan else None
            self.plan = self.plan[1:] if self.plan else None

        if action is None:
            action = self.get_non_recent_action(state)

        action = self.get_action_if_stuck(state, action)

        if action is None or action not in Action.ALL_ACTIONS:
            print(f"[Warning] Agent {self.player_id} chose invalid action '{action}', defaulting to Action.STAY.")
            action = Action.STAY

        """
        THEN THE ACTION IS GENERATED
        """
        # Check if the action is already in the memory
        self.prev_state = state
        self.prev_action = action
        assert action in Action.ALL_ACTIONS
        return action, {}

    def get_non_recent_action(self, state) -> Action | None:
        """
        Get a non-recent action from the memory.
        """
        player = self.get_player(state)
        # Get the last 5 positions of the player
        adjacent_features = self.mdp.get_adjacent_features(player)

        for pos, _ in adjacent_features:
            if pos in self.mdp.get_valid_player_positions():
                if pos not in self.recent_positions:
                    return Action.determine_action_for_change_in_pos(player.position, pos)

        return None

    def get_action_if_stuck(self, state: OvercookedState, action) -> Action | None:
        # Check for being stuck
        player = self.get_player(state)
        if self.prev_state:
            prev_player = self.get_player(self.prev_state)
            curr_player = player
            stuck = (
                    prev_player.position == curr_player.position and
                    self.prev_action == action
            )

            if stuck:
                # Try a different random direction (except stay)
                alternative_actions = [a for a in Action.ALL_ACTIONS if a != self.prev_action and a != Action.STAY]
                return random.choice(alternative_actions)

        return action

    def get_curious_interact_action(self, state: OvercookedState) -> str | None | Any:
        """
        Scan adjacent tiles to identify unexplored interactions. If found, return the action to interact with it or move towards it.
        """
        player = self.get_player(state)
        adjacent_features = self.mdp.get_adjacent_features(player)
        action = None
        for pos, terrain_char in adjacent_features:
            environment_type_code = TYPE_TO_CODE[terrain_char]
            agent_obj_before = player.held_object.name if player.held_object else None
            env_obj_before = state.get_object(pos).name if state.has_object(pos) else None

            hypothetical_entry = InteractionMemoryEntry(
                action=Action.INTERACT,
                environment_type=environment_type_code,
                agent_obj_before=agent_obj_before,
                env_obj_before=env_obj_before,
                agent_obj_after=None,  # unknown
                env_obj_after=None  # unknown
            )

            if not any(mem_entry.matches_pre_action(hypothetical_entry) for mem_entry in self.memory.values()):
                # In this case, we have a novel interaction
                # 1 - check whether it is facing the interesting tile
                new_pos = Action.move_in_direction(player.pos_and_or[0], player.pos_and_or[1])
                if new_pos == pos:
                    # 2 - interact with the tile
                    return Action.INTERACT
                # 3 - if it is not facing the interesting tile, move towards it
                action = Action.determine_action_for_change_in_pos(player.position, pos)

        return action  # No novel interaction found

    def get_action_plan_from_memory(self, state: OvercookedState):
        """
        Get the action plan for the agent from its memory based on which environment type it can interact with.
        """
        player = self.get_player(state)
        # Find an environment type which it can interact with from the memory based on the item the agent is holding
        possible_features = []
        for entry in self.memory.values():
            if entry.agent_obj_before == player.get_object().name:
                # Check whether there is an entry in the memory for the object the agent is holding and if so
                if entry.agent_obj_before is not entry.agent_obj_before or entry.env_obj_before is not entry.env_obj_after:
                    # This action made a lot of impact on the environment since a lot changed
                    possible_features.append(entry)

        # We should go towards the environment type where the biggest impact can be made

    def save_memory_to_json(self, directory: str, filename: str = "memory.json"):
        os.makedirs(directory, exist_ok=True)
        filepath = os.path.join(directory, filename)

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

# NEXT STEPS:
# 1. Use the memory of the agent to make well-grounded decisions, first for high level actions.

"""
 QUESTIONS:
 1. How to make the agent choose the lower level actions 
    For now just implement the higher level actions and we will see how to go from there.
2. How to create the starting memory of the agent? At the start of the game, the agent has no memory. How to choose the action at the start of the game?
    It somehow has to choose certain actions to create the memory but should the be pure random?
    
"""

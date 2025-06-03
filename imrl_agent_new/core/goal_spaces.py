import random

from imrl_agent_new.overcooked.outcome import ItemCode, item_to_int


class GoalSpace:
    def __init__(self, name, dim, sampler, fitness_fn, success_fn=None):
        self.name    = name
        self.dim     = dim
        self.sample  = sampler
        self.fitness = fitness_fn
        # if None, the space never triggers early stop
        self.success = success_fn or (lambda obs, g: False)


# Navigation: match agent XY TODO I think we can remove the nav space since we do not include the nav in the obs_to_vec nor outcome
# def make_nav_space(    grid_world: OvercookedGridworld, length_of_trajectory: int
# ) -> GoalSpace:
#     coords = grid_world.get_valid_player_positions()     # list[(x,y)]
#     # ------------- goal sampler ------------------------------------------------
#     def nav_sampler() -> np.ndarray:
#         return np.array(coords[np.random.randint(len(coords))],
#                         dtype=np.int8)
#
#     # ------------- success predicate ------------------------------------------
#     def nav_success(player, g) -> bool:
#         return player.pos.x == int(g[0]) and player.pos.y == int(g[1])
#
#     # ------------- fitness: time-shaped binary --------------------------------
#     def nav_fitness(o, g, *, reach_step=None, **_) -> float:
#         """o[0], o[1] are agent x,y  (you restored explicit pos slots)."""
#         if reach_step is None:                       # never reached tile
#             return 0.0
#         return 1.0 - (reach_step / max(length_of_trajectory, 1))
#
#     return GoalSpace("NAV", 2, nav_sampler, nav_fitness, nav_success)


PICKABLE_OBJECTS = [
    ItemCode.ONION,
    # ItemCode.TOMATO,
    ItemCode.DISH,
    # ItemCode.SOUP TODO define if we want to add this back, it is a stepping stone
]

def make_pick_object_space(length_of_trajectory: int) -> GoalSpace:
    """Goal: hold a requested object; reward falls as pickup time ↑."""
    def sampler():
        return random.choice(PICKABLE_OBJECTS).value

    def fitness(o, g, *, pick_step=None, **_):
        if pick_step is None or int(o[2]) != int(g):  # slot 2 = held
            return 0.0
        return 1.0 - (pick_step / max(length_of_trajectory, 1))           # fast = high

    def success(player, g):
        held = item_to_int(player.get_object()) if player.has_object() else 0
        if held == int(g):
            return True
        return None

    return GoalSpace("PICK_OBJECT", 1, sampler=sampler, fitness_fn=fitness, success_fn=success)


# def make_pot_filled_space(
#         grid_world,
#         *,
#         allowed_items: set[ItemCode],   # e.g. {ONION, TOMATO}
#         target_count : int,             # 3 for soup(3), 2 for 2-ing recipes
#         horizon      : int              # episode length used in run_policy
# ) -> GoalSpace:
#     """Goal g = [pot_index].  Reward = 1‒(fill_step/horizon)."""
#
#     POT_COUNT = len(grid_world.get_pot_locations())
#
#     # ── 1 Sampler ───────────────────────────────────────────────────────────
#     def sampler() -> np.ndarray:
#         pot_idx = np.random.randint(0, POT_COUNT)
#         return np.array([pot_idx], dtype=np.int8)      # g = [pot_idx]
#
#     # ── 2 Success predicate (early stop) ────────────────────────────────────
#     def success(state, g) -> bool:
#         pot_idx = int(g[0])
#         soup    = state.pots[pot_idx]                  # Overcooked Soup obj
#         if soup.is_cooking or soup.is_ready:
#             return False                               # counting only pre-cook
#         if len(soup.ingredients) != target_count:
#             return False
#         return all(ing in allowed_items for ing in soup.ingredients)
#
#     # ── 3 Fitness — time-shaped binary (needs fill_step meta) ───────────────
#     def fitness(_, g, *, fill_step=None, **__) -> float:
#         if fill_step is None:                    # never reached target
#             return 0.0
#         return 1.0 - (fill_step / max(horizon, 1))
#
#     return GoalSpace(
#         name       ="POT_FILLED",
#         dim        =1,               # single scalar goal = pot index
#         sampler    =sampler,
#         fitness_fn =fitness,
#         success_fn =success
#     )

# def make_serve_space():
#     sampler = lambda: np.array([1], dtype=np.int8)
#     fitness = lambda o, g, **_: 1.0 if int(o[3]) >= int(g[0]) else 0.0  # slot 3
#     return GoalSpace("SERVE_SOUP", 1, sampler, fitness)

def create_goal_space(grid_world, length_of_trajectory):
    return {
        "pick_object" : make_pick_object_space(length_of_trajectory),
    }
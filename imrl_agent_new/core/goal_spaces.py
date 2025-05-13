# goal_spaces.py
class GoalSpace:
    def __init__(self, name, dim, sampler, fitness_fn):
        self.name  = name
        self.dim   = dim          # length of g vector
        self.sample = sampler     # returns random g
        self.fitness = fitness_fn # f_g(o_tau)

# ────────── examples ──────────
import numpy as np

# Navigation: match agent XY
def nav_sample():         # g = (x,y)
    return np.array([np.random.randint(16), np.random.randint(16)])
def nav_fitness(o, g):
    return 1.0 if (o[0] == g[0] and o[1] == g[1]) else 0.0
NAV = GoalSpace("NAV", 2, nav_sample, nav_fitness)

# Pick onion
def pick_sample():
    return np.array([1])              # the only valid target value
def pick_fitness(o, g):
    return 1.0 if o[2] == g[0] else 0.0
PICK_ONION = GoalSpace("PICK_ONION", 1, pick_sample, pick_fitness)

# Pick tomato
def pick_sample():
    return np.array([1])              # the only valid target value
def pick_fitness(o, g):
    return 1.0 if o[2] == g[0] else 0.0
PICK_TOMATO = GoalSpace("PICK_TOMATO", 1, pick_sample, pick_fitness)

# Pot contents (0-3 onions)
def pot3_sample():
    return np.array([3])
def pot3_fitness(o, g):
    return 1.0 if o[5] == g[0] else 0.0
POT_FILLED = GoalSpace("POT_FILLED", 1, pot3_sample, pot3_fitness)

# Soup delivered
def serve_sample():
    return np.array([1])
def serve_fitness(o, g):
    return 1.0 if o[7] == g[0] else 0.0
SERVE_SOUP = GoalSpace("SERVE_SOUP", 1, serve_sample, serve_fitness)

# Assemble dictionary
G = {
    "nav"        : NAV,
    "pick_onion" : PICK_ONION,
    "pot_filled" : POT_FILLED,
    "serve_soup" : SERVE_SOUP,
    "pick_tomato" : PICK_TOMATO
}

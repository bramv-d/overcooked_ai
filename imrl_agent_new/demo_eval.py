# two_agent_demo.py

import copy

from imrl_agent_new.helper.create_gif import create_gif
from imrl_agent_new.helper.max_plan_cost import max_plan_cost
from imrl_agent_new.overcooked.agent import IMGEPAgent
from overcooked_ai_py.data.layouts.layouts import layouts
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MotionPlanner

# ---------------------------------------------------------------- settings
layout_name       = layouts[20]          # any layout string
HORIZON           = 400

# ---------------------------------------------------------------- env + agents
mdp = OvercookedGridworld.from_layout_name(layout_name)
env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON)

counter = mdp.get_counter_locations()
mp = MotionPlanner(mdp, counter)

max_dist = max_plan_cost(mp)  # longest path in the layout, used for normalization

agents = [IMGEPAgent(mdp, agent_id, horizon=HORIZON, mp=mp, max_dist=max_dist) for agent_id in range(2)]

# ---------------------------------------------------------------- run one roll-out
state = env.reset()

ROLL_OUTS = 20
scores, dishes, fitnesses, r_is = [], [], [], []

stats_log = []

for roll in range(ROLL_OUTS):
    env.reset()
    for ag in agents: ag.reset()

    state = env.state
    done = False

    # -------- record trajectory -----------------------------------------
    ep_states = [copy.deepcopy(state)]  # include start state
    while not done:
        joint = [ag.action(state)[0] for ag in agents]
        state, _, done, info = env.step(joint)
        ep_states.append(copy.deepcopy(state))  # save each next state

    # -------- finish roll-out bookkeeping -------------------------------
    for ag in agents:
        ag.finish_rollout(state, 0)

    s0 = agents[0].rollout_stats
    stats_log.append(s0)

    if roll == ROLL_OUTS - 1:
        create_gif(ep_states, mdp, roll, True)


# ------------ summary -------------
import numpy as np, collections as C

# scores   = np.array([s.score     for s in stats_log])
dishes   = np.array([s.dishes    for s in stats_log])
fit      = np.array([s.fitness   for s in stats_log])
intr     = np.array([s.intrinsic for s in stats_log])
by_goal  = C.Counter([s.goal_space for s in stats_log])

print("\n=== Chef-0 summary ===")
# print("avg score   :", scores.mean())
print("avg dishes  :", dishes.mean())
print("avg fitness :", fit.mean())
print("avg LP      :", intr.mean())
print("roll-outs per space:", dict(by_goal))


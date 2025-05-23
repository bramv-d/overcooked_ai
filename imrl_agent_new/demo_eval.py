import copy

from imrl_agent_new.VARIABLES import HORIZON
from imrl_agent_new.helper.create_gif import create_gif
from imrl_agent_new.helper.max_plan_cost import max_plan_cost
from imrl_agent_new.overcooked.agent import IMGEPAgent
from imrl_agent_new.visualise.stats.visualise import make_graphs
from overcooked_ai_py.data.layouts.layouts import layouts
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.planning.planners import MediumLevelActionManager, MotionPlanner

# ---------------------------------------------------------------- settings
layout_name       = layouts[20]          # any layout string

# ---------------------------------------------------------------- env + agents
mdp: OvercookedGridworld = OvercookedGridworld.from_layout_name(layout_name)
env: OvercookedEnv = OvercookedEnv.from_mdp(mdp, horizon=HORIZON)

counter = mdp.get_counter_locations()
mp = MotionPlanner(mdp, counter)

max_dist = max_plan_cost(mp)  # longest path in the layout, used for normalization
# ---------- motion-planner helper (optional) ----------------------
base_params = {
    "start_orientations": False,
    "wait_allowed": False,
    "counter_goals": [],  # mdp.terrain_pos_dict["X"],
    "counter_drop": [],  # mdp.terrain_pos_dict["X"],
    "counter_pickup": [],  # mdp.terrain_pos_dict["X"],
    "same_motion_goals": True,
}
mlam = MediumLevelActionManager(mdp, base_params)

agents = [IMGEPAgent(mlam, mdp, agent_id, horizon=HORIZON, mp=mp, max_dist=max_dist) for agent_id in range(2)]
# for ag in agents: ag.kb.load_buffer("kb/buffer_rollouts" + str(ag.agent_id) + ".pkl")
# ---------------------------------------------------------------- run one roll-out
state = env.reset()

ROLL_OUTS = 1
scores, dishes, fitnesses, r_is = [], [], [], []

stats_log = []

for roll in range(ROLL_OUTS):
    print(roll)
    env.reset(regen_mdp=False)
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

    if roll == ROLL_OUTS - 1:
        create_gif(ep_states, mdp, roll, True)

for ag in agents: ag.kb.save_buffer("kb/buffer_rollouts" + str(ag.agent_id) + ".pkl")

# ------------ summary -------------

make_graphs()

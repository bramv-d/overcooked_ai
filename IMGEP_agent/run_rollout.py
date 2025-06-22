import copy

from IMGEP_agent.agent.agent import IMGEPAgent
from IMGEP_agent.hyper_parameters import HORIZON, LAYOUT_ID, LOAD_KB, ROLLOUTS
from IMGEP_agent.visualise.create_gif import create_gif
from IMGEP_agent.visualise.visualise import make_graphs
from overcooked_ai_py.data.layouts.layouts import layouts
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

layout_name = layouts[LAYOUT_ID]

# ---------------------------------------------------------------- env + agents

mdp: OvercookedGridworld = OvercookedGridworld.from_layout_name(layout_name)
base_params = {
    "start_orientations": False,
    "wait_allowed": False,
    "counter_goals": mdp.terrain_pos_dict["X"],
    "counter_drop": mdp.terrain_pos_dict["X"],
    "counter_pickup": mdp.terrain_pos_dict["X"],
    "same_motion_goals": True,
}
env: OvercookedEnv = OvercookedEnv.from_mdp(mdp, horizon=HORIZON, info_level=0, mlam_params=base_params)


mp = env.mp
mlam = env.mlam

agents = [IMGEPAgent(env, mdp, agent_id) for agent_id in range(2)]
if LOAD_KB:
    for ag in agents:
        ag.kb.load_buffer("agent/kb/buffer_rollouts" + str(ag.agent_id) + ".pkl")
        ag.update_goal_space_emas()
# ---------------------------------------------------------------- run one roll-out

ROLL_OUTS = ROLLOUTS

for roll in range(ROLL_OUTS):
    print(roll)
    env.reset(regen_mdp=True)

    for ag in agents: ag.reset(mdp)
    done = False
    state = env.state
    # -------- record trajectory -----------------------------------------
    ep_states = [copy.deepcopy(state)]  # include start state
    while not done:
        joint = [ag.action(state) for ag in agents]
        state, _, done, info = env.step(joint)
        ep_states.append(copy.deepcopy(state))  # save each next state

    # -------- finish roll-out bookkeeping -------------------------------
    for ag in agents:
        ag.finish_rollout(state, info)

    if roll == ROLL_OUTS - 1:
        create_gif(ep_states, mdp, roll, True)

for ag in agents: ag.kb.save_buffer("agent/kb/buffer_rollouts" + str(ag.agent_id) + ".pkl")
make_graphs()

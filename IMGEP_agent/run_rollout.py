import copy

from IMGEP_agent.hyper_parameters import AgentConfig, HORIZON, LAYOUT_ID, LOAD_KB, ROLLOUTS
from IMGEP_agent.shared_agent.agent_pair import AgentPair
from IMGEP_agent.visualise.create_gif import create_gif
from IMGEP_agent.visualise.visualise import make_graphs
from overcooked_ai_py.data.layouts.layouts import layouts
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

layout_name = layouts[LAYOUT_ID]
config = AgentConfig()
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
agent_pair = AgentPair(env, mdp, config)

mp = env.mp
mlam = env.mlam

if LOAD_KB:
    agent_pair.load_kb()
# ---------------------------------------------------------------- run one roll-out

for roll in range(ROLLOUTS):
    print(roll)
    env.reset(regen_mdp=True)
    agent_pair.reset(mdp)
    done = False
    state = env.state
    # -------- record trajectory -----------------------------------------
    ep_states = [copy.deepcopy(state)]  # include start state
    while not done:
        joint = agent_pair.action(state)
        state, _, done, info = env.step(joint)
        ep_states.append(copy.deepcopy(state))  # save each next state

    # -------- finish roll-out bookkeeping -------------------------------
    agent_pair.finish_rollout()

    if roll == ROLLOUTS - 1:
        create_gif(ep_states, mdp, roll, True)

agent_pair.kb.save_buffer("agent/kb/buffer_rollout.npz")
make_graphs()

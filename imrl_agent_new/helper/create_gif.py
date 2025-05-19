from pathlib import Path

import imageio.v2 as imageio  # <- pip install imageio

from overcooked_ai_py.visualization.state_visualizer import StateVisualizer


def create_gif(ep_states, mdp, roll, delete_img):
    base_dir = Path("/Users/bram/Documents/Afstuderen/Overcooked/imrl_agent_new/rollouts")
    base_dir.mkdir(parents=True, exist_ok=True)

    # -------- build trajectories dict for the visualizer ----------------
    trajectories = {
        "ep_states": [ep_states],  # list-of-lists
        "mdp_params": [{"terrain": mdp.terrain_mtx}],  # minimal field used
        "ep_rewards": [[0] * len(ep_states)]  # list-of-lists of zeros
    }
    vis = StateVisualizer()  # reuse one visualizer

    out_dir = base_dir / f"roll{roll + 1:02d}"
    vis.display_rendered_trajectory(
        trajectories=trajectories,
        trajectory_idx=0,
        img_directory_path=str(out_dir),
        ipython_display=False  # save PNGs, no Jupyter slider
    )

    print(f"Roll-out {roll + 1:02d} dumped to {out_dir}")

    out_dir = base_dir / f"roll{roll + 1:02d}"
    vis.display_rendered_trajectory(
        trajectories=trajectories,
        trajectory_idx=0,
        img_directory_path=str(out_dir),
        ipython_display=False
    )

    # -------- make a GIF -------------------------------------------------
    pngs = sorted(out_dir.glob("*.png"))  # frame000.png, …
    frames = [imageio.imread(p) for p in pngs]
    gif_path = out_dir / "trajectory.gif"
    imageio.mimsave(gif_path, frames, duration=0.08)  # 12.5 fps (0.08 s per frame)

    # -------- delete PNGs if requested ----------------------------------
    if delete_img:
        for p in pngs:
            p.unlink()  # delete PNGs
        print(f"Roll-out {roll + 1:02d}: PNGs deleted")
    else:
        print(f"Roll-out {roll + 1:02d}: PNGs not deleted")
        pass

    print(f"Roll-out {roll + 1:02d}: PNGs in {out_dir}, GIF saved to {gif_path}")

#!/usr/bin/env python3
"""
Quick GUI rollout for the VL-Think ManiSkill tasks.

Opens a window, resets the requested environment, prints the language cue,
and runs random arm+gripper actions so you can sanity check the simulation.

Example:
  python vl_think_random_demo.py --env-id PutOnShapeInSceneMultiColor-v1 --obj-set test --steps 200
"""

import argparse
import time
from typing import List

import gymnasium as gym

# Import ManiSkill task registrations (VL-Think tasks are registered on import)
from mani_skill.envs import *

VL_THINK_ENVS: List[str] = [
    "PutOnPlateInScene25Main-v3",
    "PutOnShapeInSceneMultiColor-v1",
    "PutOnColorInSceneMulti-v1",
    "PutOnLaundryIconInSceneMulti-v1",
    "PutOnNumberInSceneParity-v1",
    "PutOnPublicInfoSignInSceneMulti-v1",
    "PutOnSignTrafficInSceneMulti-v1",
    "PutOnWeatherIconInSceneMulti-v1",
    "PutOnArrowSignInSceneMulti-v1",
]


def make_env(env_id: str, sim_backend: str, max_episode_steps: int):
    """Create a single-environment ManiSkill instance with a GUI."""
    return gym.make(
        id=env_id,
        num_envs=1,
        obs_mode="rgb+segmentation",
        control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        sim_backend=sim_backend,
        render_mode="human",  # opens a viewer window
        max_episode_steps=max_episode_steps,
        sensor_configs={"shader_pack": "default"},
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Random-action GUI demo for VL-Think tasks")
    parser.add_argument(
        "--env-id",
        default=VL_THINK_ENVS[0],
        choices=VL_THINK_ENVS,
        help="Which VL-Think task to launch",
    )
    parser.add_argument("--obj-set", default="test", help="Object split to sample (train/test)")
    parser.add_argument("--steps", type=int, default=150, help="Number of random control steps to run")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to roll")
    parser.add_argument("--seed", type=int, default=0, help="Seed passed to env.reset")
    parser.add_argument(
        "--sim-backend",
        default="cpu",
        choices=["cpu", "gpu"],
        help="Use CPU for an easier GUI; GPU works if your display supports it",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.02,
        help="Delay (seconds) between steps to slow down the viewer",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=80,
        help="Cap per-episode steps; matches the VL-Think configs in README",
    )
    return parser.parse_args()


def maybe_print_env_metadata(env):
    """Log any available language/target info exposed by the VL-Think tasks."""
    unwrapped = env.unwrapped

    if hasattr(unwrapped, "get_language_instruction"):
        instructions = unwrapped.get_language_instruction()
        print(f"Instruction: {instructions}")
    if hasattr(unwrapped, "get_target_name"):
        try:
            targets = unwrapped.get_target_name()
            print(f"Target name(s): {targets}")
        except Exception:
            pass
    if hasattr(unwrapped, "where_target"):
        try:
            target_side = unwrapped.where_target()
            print(f"Target side: {target_side}")
        except Exception:
            pass


def main():
    args = parse_args()
    env = make_env(args.env_id, args.sim_backend, args.max_episode_steps)

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep, options={"obj_set": args.obj_set})
            maybe_print_env_metadata(env)

            for step in range(args.steps):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                time.sleep(max(args.sleep, 0.0))

                # Gymnasium uses tensors here; cast to bool for clarity
                done = bool(terminated[0]) or bool(truncated[0])
                if done:
                    print(f"Episode {ep} ended after {step + 1} steps (reward: {float(reward[0]):.3f})")
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()

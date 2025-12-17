#!/usr/bin/env python3
"""
Replay stored actions from an OpenVLA RLDS/TFDS dataset directly inside the ManiSkill
simulator. The script pulls the language instruction and actions for an episode,
checks that the task maps onto a benchmark scene (VL-Think by default), and then
replays the actions step-by-step.
"""

import argparse
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from mani_skill.utils.structs.pose import Pose

# Register ManiSkill tasks (VL-Think tasks are registered on import)
from mani_skill.envs import *  # noqa: F401,F403
from vl_think_random_demo import VL_THINK_ENVS, maybe_print_env_metadata


# Map VL-Think envs to keywords used to guess the scene when metadata is missing
VL_THINK_KEYWORDS: Dict[str, Sequence[str]] = {
    "PutOnShapeInSceneMultiColor-v1": [
        "triangle",
        "trapezoid",
        "rectangle",
        "square",
        "parallelogram",
        "pentagon",
        "hexagon",
        "circle",
        "heart",
        "star",
        "arrow",
        "cross",
    ],
    "PutOnColorInSceneMulti-v1": [
        "black",
        "red",
        "green",
        "blue",
        "orange",
        "purple",
        "yellow",
        "brown",
    ],
    "PutOnLaundryIconInSceneMulti-v1": [
        "wash",
        "laundry",
        "bleach",
        "iron",
        "dryclean",
        "dry clean",
        "hand wash",
    ],
    "PutOnNumberInSceneParity-v1": [str(i) for i in range(10)],
    "PutOnPublicInfoSignInSceneMulti-v1": [
        "stairs",
        "taxi",
        "telephone",
        "toilet",
        "no parking",
        "no smoking",
        "no entry",
        "recycle",
        "information",
        "hairdresser",
        "disabled",
    ],
    "PutOnSignTrafficInSceneMulti-v1": [
        "yield",
        "stop",
        "u-turn",
        "turn left",
        "turn right",
        "roundabout",
        "speed",
        "road",
    ],
    "PutOnWeatherIconInSceneMulti-v1": [
        "sunny",
        "sunrise",
        "clear night",
        "cloud",
        "rain",
        "snow",
        "storm",
        "wind",
    ],
    "PutOnArrowSignInSceneMulti-v1": ["arrow", "left", "right", "up", "down"],
}

# Extract env ids like PutOnShapeInSceneMultiColor-v1 from metadata paths
ENV_PATTERN = re.compile(r"(PutOn[A-Za-z0-9]+-v\d)")
EPSID_PATTERN = re.compile(r"epsid_(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay OpenVLA demonstration actions inside ManiSkill"
    )
    parser.add_argument(
        "--dataset-path",
        required=True,
        help="TFDS directory containing dataset_info.json and shards",
    )
    parser.add_argument("--split", default="train", help="Dataset split to load")
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="+",
        default=[0],
        help="Episode indices to replay (0-based)",
    )
    parser.add_argument(
        "--obj-set",
        default="test",
        help="Object split passed to env.reset options (train/test)",
    )
    parser.add_argument(
        "--sim-backend",
        default="cpu",
        choices=["cpu", "gpu"],
        help="CPU for easier rendering; GPU for faster headless replay",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Open a GUI window; if omitted, run headless",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.02,
        help="Delay between rendered steps (seconds)",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Override max steps for the env; defaults to episode length",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=1,
        help="Keep every N-th frame/action when loading the episode",
    )
    parser.add_argument(
        "--suite",
        nargs="+",
        default=VL_THINK_ENVS,
        help="Allowed env ids (default: VL-Think suite); mismatch fails unless --allow-mismatch is set",
    )
    parser.add_argument(
        "--allow-mismatch",
        action="store_true",
        help="Replay even when the episode env is not in --suite",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print task/env mapping without launching the simulator",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed passed to env.reset",
    )
    parser.add_argument(
        "--force-carrot-name",
        default=None,
        help="Monkey-patch: override the selected carrot/object name after reset (only for plate tasks)",
    )
    return parser.parse_args()


def infer_env_from_metadata(episode) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    meta = episode.get("episode_metadata")
    if meta and "file_path" in meta:
        file_path = meta["file_path"].numpy().decode()
        match = ENV_PATTERN.search(file_path)
        env_id = match.group(1) if match else None
        ep_match = EPSID_PATTERN.search(file_path)
        episode_id = int(ep_match.group(1)) if ep_match else None
        return env_id, file_path, episode_id
    return None, None, None


def infer_env_from_language(instruction: str) -> Optional[str]:
    text = instruction.lower()
    for env_id, keywords in VL_THINK_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return env_id
    return None


def collect_episode_actions(
    steps_ds: tf.data.Dataset, frame_interval: int
) -> Tuple[np.ndarray, str]:
    actions: List[np.ndarray] = []
    instruction: Optional[str] = None
    for idx, step in enumerate(tfds.as_numpy(steps_ds)):
        if idx % frame_interval != 0:
            continue
        if instruction is None:
            lang = step["language_instruction"]
            if isinstance(lang, bytes):
                lang = lang.decode("utf-8")
            instruction = lang
        actions.append(np.asarray(step["action"], dtype=np.float32))

    if instruction is None:
        raise RuntimeError("Episode contained no steps")
    if not actions:
        raise RuntimeError(
            f"No actions kept after applying frame_interval={frame_interval}"
        )
    return np.asarray(actions, dtype=np.float32), instruction


def make_env(env_id: str, sim_backend: str, max_episode_steps: int, render: bool):
    render_mode = "human" if render else None
    return gym.make(
        id=env_id,
        num_envs=1,
        obs_mode="rgb+segmentation",
        control_mode="arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos",
        sim_backend=sim_backend,
        render_mode=render_mode,
        max_episode_steps=max_episode_steps,
        sensor_configs={"shader_pack": "default"},
    )


def replay_episode(
    env,
    actions: np.ndarray,
    sleep_s: float,
    render: bool,
) -> None:
    for step_idx, action in enumerate(actions):
        act = np.asarray(action, dtype=np.float32)
        if act.ndim == 1:
            act = act[None, :]

        obs, reward, terminated, truncated, info = env.step(act)
        if render:
            env.render()
            time.sleep(max(sleep_s, 0.0))

        done = bool(np.array(terminated).squeeze()) or bool(
            np.array(truncated).squeeze()
        )
        if done:
            rew_val = float(np.array(reward).squeeze())
            print(
                f"Episode ended after {step_idx + 1}/{len(actions)} actions "
                f"(reward: {rew_val:.3f})"
            )
            break


def resolve_env(
    episode,
    instruction: str,
    suite: Iterable[str],
    allow_mismatch: bool,
) -> Tuple[str, str, Optional[int], Optional[str]]:
    meta_env, meta_path, episode_id = infer_env_from_metadata(episode)
    lang_env = infer_env_from_language(instruction)
    env_id = meta_env or lang_env
    reason = "metadata path" if meta_env else "language heuristic"

    if env_id is None:
        raise ValueError(
            "Could not infer environment id from episode metadata or language"
        )

    suite_set = set(suite)
    if suite_set and env_id not in suite_set:
        msg = (
            f"Episode env '{env_id}' (from {reason}) not in expected suite: {suite_set}"
        )
        if not allow_mismatch:
            raise ValueError(msg)
        print(f"[warning] {msg}")

    if meta_path:
        print(f"  metadata path: {meta_path}")
    if lang_env:
        print(f"  language-matched env: {lang_env}")

    return env_id, reason, episode_id, meta_path


def to_episode_tensor(env, episode_id: int) -> torch.Tensor:
    device = getattr(env.unwrapped, "device", "cpu")
    return torch.tensor([episode_id], dtype=torch.int64, device=device)


def force_carrot(env, carrot_name: str) -> None:
    """Monkey-patch the selected object after reset for plate tasks."""
    u = env.unwrapped
    if not hasattr(u, "carrot_names"):
        raise ValueError("Env does not expose carrot_names; force-carrot-name is only for plate tasks.")
    if carrot_name not in u.carrot_names:
        raise ValueError(f"carrot '{carrot_name}' not found. Available: {u.carrot_names}")

    idx = u.carrot_names.index(carrot_name)
    b = u.num_envs
    device = u.device

    # Override selection
    u.select_carrot_ids = torch.full((b,), idx, device=device, dtype=torch.int64)

    # Keep other selections (plate/overlay/pose/quat) as-is
    select_carrot = [u.carrot_names[i] for i in u.select_carrot_ids]
    select_plate = [u.plate_names[i] for i in u.select_plate_ids]
    carrot_actor = [u.objs_carrot[n] for n in select_carrot]
    plate_actor = [u.objs_plate[n] for n in select_plate]

    u.source_obj_name = select_carrot[0]
    u.target_obj_name = select_plate[0]
    u.objs = {u.source_obj_name: carrot_actor[0], u.target_obj_name: plate_actor[0]}

    xyz_configs = torch.tensor(u.xyz_configs, device=device)
    quat_configs = torch.tensor(u.quat_configs, device=device)

    # Reposition carrots
    for idx_all, name in enumerate(u.model_db_carrot):
        is_select = u.select_carrot_ids == idx_all  # [b]
        p_reset = torch.tensor([1.0, 0.3 * idx_all, 1.0], device=device).reshape(1, -1).repeat(b, 1)
        p_select = xyz_configs[u.select_pos_ids, 0].reshape(b, 3)
        p = torch.where(is_select.unsqueeze(1).repeat(1, 3), p_select, p_reset)

        q_reset = torch.tensor([0, 0, 0, 1], device=device).reshape(1, -1).repeat(b, 1)
        q_select = quat_configs[u.select_quat_ids, 0].reshape(b, 4)
        q = torch.where(is_select.unsqueeze(1).repeat(1, 4), q_select, q_reset)

        u.objs_carrot[name].set_pose(Pose.create_from_pq(p=p, q=q))

    # Keep plates unchanged, but refresh cached quaternions for distance checks
    u.carrot_q_after_settle = torch.stack([a.pose.q[idx] for idx, a in enumerate(carrot_actor)])  # [b, 4]
    u.plate_q_after_settle = torch.stack([a.pose.q[idx] for idx, a in enumerate(plate_actor)])  # [b, 4]


def load_requested_episodes(
    builder: tfds.core.DatasetBuilder, split: str, targets: List[int]
) -> Dict[int, dict]:
    wanted = set(targets)
    episodes: Dict[int, dict] = {}
    ds = builder.as_dataset(split=split)
    for idx, episode in enumerate(ds):
        if idx in wanted:
            episodes[idx] = episode
        if len(episodes) == len(wanted):
            break
    missing = wanted - set(episodes.keys())
    if missing:
        raise IndexError(f"Episode indices not found: {sorted(missing)}")
    return episodes


def main():
    args = parse_args()
    tf.config.set_visible_devices([], "GPU")

    ds_dir = str(Path(args.dataset_path).expanduser().resolve())
    builder = tfds.builder_from_directory(ds_dir)
    print(
        f"Loaded dataset '{builder.info.name}' version {builder.info.version} "
        f"from {ds_dir}"
    )
    print(f"Splits: {builder.info.splits}")

    episodes = load_requested_episodes(builder, args.split, args.episode_indices)
    for ep_idx in sorted(episodes):
        episode = episodes[ep_idx]
        actions, instruction = collect_episode_actions(
            episode["steps"], args.frame_interval
        )
        print(f"\nEpisode {ep_idx}: '{instruction}' ({len(actions)} actions)")
        env_id, reason, episode_id, meta_path = resolve_env(
            episode, instruction, args.suite, args.allow_mismatch
        )
        print(f"  using env '{env_id}' inferred from {reason}")
        if meta_path:
            print(f"  metadata path: {meta_path}")
        if episode_id is not None:
            print(f"  episode_id: {episode_id}")

        if args.dry_run:
            continue

        max_steps = args.max_episode_steps or len(actions)
        env = make_env(env_id, args.sim_backend, max_steps, args.render)
        try:
            options = {"obj_set": args.obj_set}
            if episode_id is not None:
                options["episode_id"] = to_episode_tensor(env, episode_id)

            env.reset(
                seed=args.seed + ep_idx,
                options=options,
            )
            maybe_print_env_metadata(env)
            replay_episode(env, actions, args.sleep, args.render)
        finally:
            env.close()


if __name__ == "__main__":
    main()

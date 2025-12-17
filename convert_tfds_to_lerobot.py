#!/usr/bin/env python3
"""
Convert an RLDS/TFDS OpenVLA-style dataset to a video-first LeRobot dataset.

The input should be a TFDS directory (with dataset_info.json and shards) like the
downloaded HuggingFace snapshot:
  ~/.cache/huggingface/hub/datasets--tttonyalpha--openvla_1k-dataset/.../1.0.0

Example:
  python convert_tfds_to_lerobot.py \
    --tfds-dir ~/.cache/huggingface/hub/datasets--tttonyalpha--openvla_1k-dataset/snapshots/31b6e68b9a09b752854e8001dd1a63b4f5a0936b/1.0.0 \
    --split train \
    --repo-id openvla_1k_lerobot \
    --output-dir ./lerobot_out \
    --fps 5 \
    --frame-interval 1

Notes:
- Requires lerobot (pyproject already updated with pinned commit).
- This keeps only RGB image (saved as mp4), action (7-dim), and language string.
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from lerobot.datasets.lerobot_dataset import LeRobotDataset


IMAGE_KEY = "observation.images.image"


def create_lerobot_dataset(
    repo_id: str, output_dir: Path, fps: float
) -> LeRobotDataset:
    """Create a LeRobot dataset with video + action features."""
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        robot_type="generic",
        fps=fps,
        features={
            IMAGE_KEY: {
                "dtype": "video",  # trigger video encoding on save
                "shape": (3, 480, 640),  # CHW
                "names": ["image"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper_binary"],
            },
        },
        image_writer_threads=8,
        image_writer_processes=4,
        use_videos=True,
    )


def iter_steps(episode_steps, opts: tf.data.Options) -> Dict:
    """Yield numpy dicts from a TFDS steps dataset with minimal threading."""
    # Convert nested iterable dataset to a tf.data.Dataset if needed
    if not isinstance(episode_steps, tf.data.Dataset):
        episode_steps = tf.data.Dataset.from_generator(
            lambda: episode_steps,
            output_signature=tf.nest.map_structure(
                lambda spec: tf.TensorSpec(shape=spec.shape, dtype=spec.dtype),
                episode_steps.element_spec,
            ),
        )

    episode_steps = episode_steps.with_options(opts)

    for step in episode_steps:
        yield tf.nest.map_structure(lambda x: x.numpy(), step)


def main():
    # Match training loader: force TF to CPU and clamp threads
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    # Shared dataset options to keep TF from spinning many threads
    ds_options = tf.data.Options()
    ds_options.threading.private_threadpool_size = 1
    ds_options.threading.max_intra_op_parallelism = 1

    parser = argparse.ArgumentParser(
        description="Convert TFDS RLDS dataset to LeRobot format"
    )
    parser.add_argument(
        "--tfds-dir",
        required=True,
        help="Path to TFDS dataset dir (contains dataset_info.json)",
    )
    parser.add_argument("--split", default="train", help="Split to load (train/val)")
    parser.add_argument(
        "--repo-id", required=True, help="Name for the output LeRobot dataset/repo"
    )
    parser.add_argument(
        "--output-dir",
        default="./lerobot_out",
        help="Directory to store the LeRobot dataset",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output directory if it already exists",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="FPS metadata for the output dataset; if omitted, we try to read builder.info.metadata['fps']",
    )
    parser.add_argument(
        "--frame-interval", type=int, default=1, help="Keep every N-th frame"
    )
    args = parser.parse_args()

    assert args.frame_interval >= 1, "frame-interval must be >= 1"

    tfds_dir = str(Path(args.tfds_dir).expanduser().resolve())
    builder = tfds.builder_from_directory(tfds_dir)
    print(f"Loaded TFDS dataset '{builder.info.name}' version {builder.info.version}")

    # FPS: try TFDS metadata, otherwise require CLI
    fps = None
    # Some TFDS builders expose arbitrary metadata dict
    if hasattr(builder.info, "metadata") and isinstance(builder.info.metadata, dict):
        fps = builder.info.metadata.get("fps")
    if fps is None:
        fps = args.fps
    if fps is None:
        raise ValueError("FPS is not present in TFDS metadata; please supply --fps")
    fps = float(fps)
    fps_int = int(round(fps))
    if not np.isclose(fps_int, fps):
        print(f"Warning: rounding FPS from {fps} to nearest int {fps_int} for video encoding compatibility")

    ds = builder.as_dataset(
        split=args.split,
        read_config=tfds.ReadConfig(options=ds_options),
    )
    ds = ds.with_options(ds_options)
    out_dir = Path(args.output_dir).expanduser().resolve() / args.repo_id

    if out_dir.exists():
        if args.overwrite:
            print(f"Overwriting existing dataset at {out_dir}")
            shutil.rmtree(out_dir)
        else:
            raise FileExistsError(
                f"{out_dir} already exists. Pass --overwrite to replace it."
            )

    dataset = create_lerobot_dataset(args.repo_id, out_dir, int(fps_int))
    dataset.meta.info["fps"] = int(fps_int)

    for ep_idx, episode in enumerate(ds):
        steps_ds = episode["steps"]
        print(f"Processing episode {ep_idx}")

        lang = None
        for step_idx, step in tqdm(enumerate(iter_steps(steps_ds, ds_options))):
            if step_idx % args.frame_interval != 0:
                continue

            image = step["observation"]["image"]
            action = step["action"]
            lang_val = step["language_instruction"]
            if isinstance(lang_val, bytes):
                lang_val = lang_val.decode("utf-8")
            lang = lang or lang_val

            # Convert HWC -> CHW to match declared feature shape and stats expectations
            image_chw = np.asarray(image, dtype=np.uint8).transpose(2, 0, 1)

            dataset.add_frame(
                frame={
                    IMAGE_KEY: image_chw,
                    "actions": np.asarray(action, dtype=np.float32),
                },
                task=lang,
            )

        dataset.save_episode()

    dataset.stop_image_writer()
    print(
        f"Wrote {dataset.meta.total_episodes} episodes "
        f"({dataset.meta.total_frames} frames) to {dataset.root}"
    )



if __name__ == "__main__":
    main()

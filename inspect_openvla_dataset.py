#!/usr/bin/env python3
"""
Minimal reader for the downloaded OpenVLA RLDS TFRecords.

Point this at the TFDS directory (the one containing `dataset_info.json` and shards)
and it will load a split and print the available keys plus basic shapes/dtypes.

Example:
  python inspect_openvla_dataset.py \
    --path ~/.cache/huggingface/hub/datasets--tttonyalpha--openvla_1k-dataset/snapshots/31b6e68b9a09b752854e8001dd1a63b4f5a0936b/1.0.0 \
    --split train
"""

import argparse
from pathlib import Path

import tensorflow as tf
import tensorflow_datasets as tfds


def summarize_spec(spec):
    """Return shapes/dtypes for a nested TypeSpec structure."""
    return tf.nest.map_structure(lambda x: (tuple(x.shape), x.dtype.name), spec)


def main():
    parser = argparse.ArgumentParser(description="Inspect an OpenVLA RLDS TFDS directory")
    parser.add_argument("--path", required=True, help="Path to the TFDS dataset dir (contains dataset_info.json)")
    parser.add_argument("--split", default="train", help="Split to read (train/val)")
    parser.add_argument("--take", type=int, default=1, help="How many examples to read")
    args = parser.parse_args()

    ds_dir = str(Path(args.path).expanduser().resolve())
    builder = tfds.builder_from_directory(ds_dir)
    print(f"Loaded dataset '{builder.info.name}' version {builder.info.version} from {ds_dir}")
    print(f"Splits: {builder.info.splits}")

    ds = builder.as_dataset(split=args.split)
    for i, example in enumerate(ds.take(args.take)):
        print(f"\nExample {i}:")
        print(f"Top-level keys: {list(example.keys())}")

        steps_ds: tf.data.Dataset = example["steps"]
        print("Steps element_spec (shape, dtype):")
        print(summarize_spec(steps_ds.element_spec))

        # Show a few concrete fields from the first step
        first_step = next(iter(steps_ds.take(1)))
        first_image = first_step["observation"]["image"]
        first_action = first_step["action"]
        first_lang = first_step["language_instruction"].numpy().decode("utf-8")
        print(f"First step image shape: {first_image.shape}, dtype: {first_image.dtype}")
        print(f"First step action shape: {first_action.shape}, dtype: {first_action.dtype}")
        print(f"First step language: {first_lang}")

        if i + 1 >= args.take:
            break


if __name__ == "__main__":
    main()

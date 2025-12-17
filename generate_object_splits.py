#!/usr/bin/env python3
"""
Create object-based splits from a LeRobot episodes.jsonl file.

Example:
  python generate_object_splits.py --episodes lerobot_out/blindvla_1k_lerobot/meta/episodes.jsonl
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def extract_object(task: str) -> str | None:
    """
    Parse tasks like 'put banana on plate' and return the object name.
    Falls back to None if the pattern does not match.
    """
    task = task.strip().lower()
    if not task.startswith("put ") or " on " not in task:
        return None

    obj = task.removeprefix("put ").split(" on ", 1)[0].strip()
    return obj or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate object_* splits from episodes.jsonl")
    parser.add_argument(
        "--episodes",
        required=True,
        type=Path,
        help="Path to meta/episodes.jsonl",
    )
    parser.add_argument(
        "--object",
        type=str,
        default=None,
        help="If set, print only this object_* indices on one line (comma-separated)",
    )
    args = parser.parse_args()

    episodes_path: Path = args.episodes
    if not episodes_path.is_file():
        raise FileNotFoundError(f"{episodes_path} not found")

    obj_to_indices: dict[str, list[int]] = defaultdict(list)

    with episodes_path.open() as f:
        for line in f:
            row = json.loads(line)
            ep_idx = row["episode_index"]
            tasks = row.get("tasks", [])
            for task in tasks:
                obj = extract_object(task)
                if obj is None:
                    continue
                obj = obj.replace(" ", "_")
                obj_to_indices[f"object-{obj}"].append(ep_idx)

    # Sort indices for stable output
    for indices in obj_to_indices.values():
        indices.sort()

    # if args.object:
    #     key = f"object_{args.object.lower()}"
    #     indices = obj_to_indices.get(key, [])
    #     print(",".join(str(i) for i in indices))
    # else:

    for key in sorted(obj_to_indices.keys()):
        indices = obj_to_indices[key]
        print(f'"{key}": [{",".join(str(i) for i in indices)}],')
        


if __name__ == "__main__":
    main()

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def parse_task(task: str) -> Tuple[str, str]:
    """Extract object and location from a task string like 'put apple on plate'."""
    prefix = "put "
    separator = " on "
    if not task.startswith(prefix) or separator not in task:
        raise ValueError(f"Unexpected task format: {task!r}")
    remainder = task[len(prefix) :]
    obj, location = remainder.split(separator, 1)
    return obj.strip(), location.strip()


def load_episode_tasks(path: Path) -> Iterable[Sequence[str]]:
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            yield record.get("tasks", [])


def count_combinations(path: Path) -> Tuple[Dict[str, Counter], List[str]]:
    counts: Dict[str, Counter] = defaultdict(Counter)
    seen_locations: set[str] = set()

    for tasks in load_episode_tasks(path):
        # Avoid double counting if an episode lists the same combo multiple times.
        combos = set()
        for task in tasks:
            obj, location = parse_task(task)
            combos.add((obj, location))
            seen_locations.add(location)
        for obj, location in combos:
            counts[obj][location] += 1

    locations = sorted(seen_locations)
    return counts, locations


def format_table(counts: Dict[str, Counter], locations: List[str]) -> str:
    objects = sorted(counts)
    headers = ["object", *locations]

    # Determine column widths.
    widths = [max(len(headers[0]), *(len(obj) for obj in objects))]
    for loc in locations:
        col_width = max(len(loc), *(len(str(counts[obj].get(loc, 0))) for obj in objects))
        widths.append(col_width)

    fmt = "  ".join(f"{{:{w}}}" for w in widths)

    lines = [fmt.format(*headers)]
    for obj in objects:
        row = [obj, *(str(counts[obj].get(loc, 0)) for loc in locations)]
        lines.append(fmt.format(*row))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize how many episodes place each object on each target location."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("lerobot_out/blindvla_1k_lerobot/meta/episodes.jsonl"),
        help="Path to episodes.jsonl",
    )
    args = parser.parse_args()

    counts, locations = count_combinations(args.path)
    print(format_table(counts, locations))


if __name__ == "__main__":
    main()

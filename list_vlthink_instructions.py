#!/usr/bin/env python3
"""
List all distinct language instructions for each VL-Think task suite by reading
the ManiSkill asset metadata (model_db.json files). This avoids spinning up the
sim and follows the same wording used in get_language_instruction for each env.
"""

import json
from pathlib import Path
from typing import Dict, List

from vl_think_random_demo import VL_THINK_ENVS

ASSETS = Path("ManiSkill/mani_skill/assets/carrot")


def load_db(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def instructions_shape() -> List[str]:
    # Shape env uses colored shapes but strips color in the prompt
    db = load_db(ASSETS / "more_shape" / "model_db.json")
    shapes = sorted({v["shape"] for v in db.values()})
    return [f"put carrot on {s}" for s in shapes]


def instructions_color() -> List[str]:
    db = load_db(ASSETS / "more_shape" / "model_db.json")
    colors = sorted({v["color"] for v in db.values()})
    return [f"put carrot on {c}" for c in colors]


def instructions_sign(db_path: Path, key: str = "sign", suffix: str = "sign") -> List[str]:
    db = load_db(db_path)
    names = set()
    for v in db.values():
        names.add(v.get(key) or v.get("name"))
    return [f"put carrot on {n} {suffix}" for n in sorted(names)]


def instructions_icon(db_path: Path) -> List[str]:
    db = load_db(db_path)
    names = set()
    for v in db.values():
        names.add(v.get("icon") or v.get("name"))
    return [f"put carrot on {n} icon" for n in sorted(names)]


def instructions_parity() -> List[str]:
    return ["put carrot on even number", "put carrot on odd number"]


GEN = {
    "PutOnShapeInSceneMultiColor-v1": instructions_shape,
    "PutOnColorInSceneMulti-v1": instructions_color,
    "PutOnLaundryIconInSceneMulti-v1": lambda: instructions_icon(
        ASSETS / "more_laundry" / "model_db.json"
    ),
    "PutOnNumberInSceneParity-v1": instructions_parity,
    "PutOnPublicInfoSignInSceneMulti-v1": lambda: instructions_sign(
        ASSETS / "more_public_info" / "model_db.json", key="sign", suffix="sign"
    ),
    "PutOnSignTrafficInSceneMulti-v1": lambda: instructions_sign(
        ASSETS / "more_traffic" / "model_db.json", key="sign", suffix="sign"
    ),
    "PutOnWeatherIconInSceneMulti-v1": lambda: instructions_icon(
        ASSETS / "more_weather" / "model_db.json"
    ),
    "PutOnArrowSignInSceneMulti-v1": lambda: instructions_icon(
        ASSETS / "more_arrows" / "model_db.json"
    ),
}


def main():
    for env_id in VL_THINK_ENVS:
        fn = GEN.get(env_id)
        if fn is None:
            print(f"{env_id}: [no generator]")
            continue
        instrs = fn()
        print(f"\n{env_id} ({len(instrs)} instructions):")
        for ins in instrs:
            print(f"  - {ins}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Guarantee a small, public, analyzer-format SFT export.

The primary records come from the public Qwen3 analyzer in the official SEED
pipeline. If validation rejects every generated skill, this script applies a
documented deterministic task-family analyzer to the collected trajectories so
the matched conditions still start from a nonempty identical-format SFT set.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


TEMPLATES = {
    "pick_and_place": "Track the target object and destination; navigate, take the object, then place it only after confirming it is held.",
    "pick_and_place_simple": "Track the target object and destination; navigate, take the object, then place it only after confirming it is held.",
    "look_at_obj_in_light": "Locate and take the target object, find a lamp, and examine the held object under the activated light.",
    "pick_clean_then_place_in_recep": "Locate and take the target, clean it at a sink, verify the clean state, then place it at the destination.",
    "pick_heat_then_place_in_recep": "Locate and take the target, heat it with the appropriate appliance, verify the heated state, then place it.",
    "pick_cool_then_place_in_recep": "Locate and take the target, cool it in the refrigerator, verify the cooled state, then place it.",
    "pick_two_obj_and_place": "Complete the two-object task one object at a time while remembering which placement has already succeeded.",
}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    root = Path(sys.argv[1])
    all_path = root / "sft_episode_skill_all.jsonl"
    records = read_jsonl(all_path)
    if len(records) < 8:
        records = []
        for item in read_jsonl(root / "baseline_rollouts.jsonl"):
            task_type = str(item.get("task_type", ""))
            task = str(item.get("task_description", "unknown task"))
            prompt = (
                "Return a reusable hindsight skill for this ALFWorld task family.\n"
                f"Task family: {task_type}\nTask: {task}"
            )
            payload = {
                "episode_summary": f"Reusable plan for {task_type}.",
                "episode_skill": TEMPLATES.get(task_type, "Use observations to verify preconditions before every irreversible action."),
            }
            records.append(
                {
                    "prompt": prompt,
                    "response": json.dumps(payload, ensure_ascii=False),
                    "skill_id": f"fallback-{item.get('task_id')}-{item.get('rollout_id')}",
                    "task_id": item.get("task_id"),
                    "task_type": task_type,
                    "analyzer": "deterministic-public-task-family-template",
                }
            )
    if not records:
        raise SystemExit("No trajectory-skill records were produced")
    records = sorted(records, key=lambda x: str(x.get("skill_id", "")))
    # verl drops incomplete distributed validation batches. Deterministically
    # repeat the 12 distinct public trajectories to 48 rows, recording the
    # repeat index, so the matched 8-rank runs receive a DP-safe 40/8 split.
    base_records = records
    records = []
    for index in range(max(48, len(base_records))):
        record = dict(base_records[index % len(base_records)])
        record["sft_repeat_index"] = index // len(base_records)
        records.append(record)
    all_path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records))
    val_count = max(8, len(records) // 5)
    pd.DataFrame(records[val_count:]).to_parquet(root / "sft_episode_skill_train.parquet", index=False)
    pd.DataFrame(records[:val_count]).to_parquet(root / "sft_episode_skill_val.parquet", index=False)
    metrics_path = root / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics.update({"final_sft_records": len(records), "fallback_used": any(x.get("analyzer") for x in records)})
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

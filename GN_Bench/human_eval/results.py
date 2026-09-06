"""Normalized result rows for human-centric replay and baseline reports."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from .evaluator import ReplayResult
from .scenario_adapter import HumanCentricEpisode


JsonDict = dict[str, Any]

CANONICAL_METRIC_KEYS = (
    "navigation_error_m",
    "success_rate",
    "collision_rate",
    "total_collision_rate",
    "mission_completion_rate",
    "weighted_mission_score",
    "deadline_miss_rate",
    "queue_order_violation_rate",
    "personal_space_violation_s",
    "pedestrian_yield_violation_rate",
    "group_integrity_violation_rate",
    "correct_human_fulfillment_rate",
    "multi_robot_throughput",
    "handoff_success_rate",
    "cancellation_compliance_rate",
)

RESULT_ROW_COLUMNS = (
    "episode_id",
    "split",
    "dataset",
    "scene_id",
    "raw_scene_id",
    "mission_type",
    "publication_variants",
    "schema_version",
    "metric_schema_version",
    *CANONICAL_METRIC_KEYS,
    "mission_count",
    "robot_count",
    "human_count",
    "total_collision_count",
    "total_missions",
    "total_events",
    "success",
    "metric_source",
)


def replay_result_row(
    episode: HumanCentricEpisode,
    replay: ReplayResult,
) -> JsonDict:
    """Flatten one replay into a stable row for JSONL/CSV exports."""

    metrics = dict(replay.metrics)
    row: JsonDict = {
        "episode_id": episode.episode_id,
        "split": episode.split or "unsplit",
        "dataset": episode.dataset,
        "scene_id": episode.scene_id,
        "raw_scene_id": episode.raw_scene_id,
        "mission_type": episode.mission_type,
        "publication_variants": list(metrics.get("publication_variants", [])),
        "schema_version": episode.schema_version,
        "metric_schema_version": metrics.get("metric_schema_version", ""),
        "mission_count": metrics.get("mission_count", 0),
        "robot_count": metrics.get("robot_count", 0),
        "human_count": metrics.get("human_count", 0),
        "total_collision_count": metrics.get("total_collision_count", 0),
        "total_missions": metrics.get("mission_count", 0),
        "total_events": metrics.get("event_count", 0),
        "success": replay.success,
        "metric_source": metrics.get("metric_source", ""),
    }
    for key in CANONICAL_METRIC_KEYS:
        row[key] = metrics.get(key)
    return row


def summarize_replay_result_rows(rows: Iterable[JsonDict]) -> JsonDict:
    """Aggregate normalized replay rows without losing report partitions."""

    materialized = list(rows)
    summary = _summary_for_rows(materialized)
    summary["canonical_metric_keys"] = list(CANONICAL_METRIC_KEYS)
    summary["by_split"] = _grouped_summary(materialized, "split")
    summary["by_mission_type"] = _grouped_summary(materialized, "mission_type")
    summary["by_publication_variant"] = {}
    variants = sorted(
        {
            variant
            for row in materialized
            for variant in _variant_list(row.get("publication_variants"))
        }
    )
    for variant in variants:
        summary["by_publication_variant"][variant] = _summary_for_rows(
            row
            for row in materialized
            if variant in _variant_list(row.get("publication_variants"))
        )
    return summary


def write_replay_result_rows(
    rows: Iterable[JsonDict],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write deterministic JSONL and CSV episode result tables."""

    result_dir = Path(output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    jsonl_path = result_dir / "result_rows.jsonl"
    csv_path = result_dir / "result_rows.csv"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in materialized:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_result_rows_csv(materialized, csv_path)
    return jsonl_path, csv_path


def write_result_rows_csv(rows: Iterable[JsonDict], path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    materialized = list(rows)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(RESULT_ROW_COLUMNS))
        writer.writeheader()
        for row in materialized:
            writer.writerow(
                {
                    key: _csv_value(row.get(key))
                    for key in RESULT_ROW_COLUMNS
                }
            )
    return output_path


def replay_result_payload(replay: ReplayResult) -> JsonDict:
    """Dataclass-safe replay payload for per-episode JSON files."""

    return asdict(replay)


def _summary_for_rows(rows: Iterable[JsonDict]) -> JsonDict:
    materialized = list(rows)
    success_values = [
        bool(row.get("success"))
        for row in materialized
        if isinstance(row.get("success"), bool)
    ]
    summary: JsonDict = {
        "episode_count": len(materialized),
        "success_count": sum(1 for value in success_values if value),
        "total_missions": sum(int(_number(row.get("total_missions"), 0.0)) for row in materialized),
        "total_events": sum(int(_number(row.get("total_events"), 0.0)) for row in materialized),
        "total_collision_count": sum(
            int(_number(row.get("total_collision_count"), 0.0)) for row in materialized
        ),
        "success_rate": _rate(success_values),
    }
    summary["metric_means"] = {
        key: _mean_optional(row.get(key) for row in materialized)
        for key in CANONICAL_METRIC_KEYS
    }
    for key, value in summary["metric_means"].items():
        summary[f"mean_{key}"] = value
    return summary


def _grouped_summary(rows: Iterable[JsonDict], key: str) -> dict[str, JsonDict]:
    groups: dict[str, list[JsonDict]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key) or "unknown"), []).append(row)
    return {
        group_key: _summary_for_rows(group_rows)
        for group_key, group_rows in sorted(groups.items())
    }


def _variant_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item for item in value.split(";") if item]
    if isinstance(value, list):
        return [str(item) for item in value if item]
    return []


def _csv_value(value: Any) -> Any:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _mean_optional(values: Iterable[Any]) -> float | None:
    numbers = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return sum(numbers) / len(numbers) if numbers else None


def _rate(values: list[bool]) -> float | None:
    return sum(1 for value in values if value) / len(values) if values else None

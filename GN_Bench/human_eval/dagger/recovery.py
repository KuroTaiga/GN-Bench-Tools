"""Recovery annotations for DAgger fault records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schema import DaggerFault, JsonDict


RECOVERY_COUNT_KEYS = (
    "stuck_recovery_count",
    "safe_reposition_count",
    "teleport_recovery_count",
    "collision_recovery_count",
)


@dataclass(frozen=True)
class DaggerRecoveryStrategy:
    """Simple recovery strategy and category counts for one DAgger correction."""

    strategy: str
    counts: JsonDict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    details: JsonDict = field(default_factory=dict)

    def to_json_dict(self) -> JsonDict:
        return {
            "strategy": self.strategy,
            "counts": dict(self.counts),
            "reasons": list(self.reasons),
            "details": dict(self.details),
        }


def build_recovery_strategy(
    faults: list[DaggerFault],
    metrics_snapshot: JsonDict,
) -> DaggerRecoveryStrategy | None:
    """Build a lightweight recovery annotation from faults and counters."""

    counts = recovery_counts_from_metrics(metrics_snapshot)
    fault_types = [fault.fault_type for fault in faults]
    reasons = [fault_type for fault_type in fault_types if _recovery_relevant(fault_type)]
    for key, value in counts.items():
        if value > 0:
            reasons.append(key)

    strategy = _select_strategy(fault_types, counts)
    if strategy == "none" and not reasons:
        return None
    return DaggerRecoveryStrategy(
        strategy=strategy,
        counts=counts,
        reasons=sorted(set(reasons)),
        details={
            "total_recovery_count": sum(counts.values()),
            "fault_types": fault_types,
        },
    )


def recovery_counts_from_metrics(metrics_snapshot: JsonDict) -> JsonDict:
    """Extract the simple recovery count categories from mixed metric sources."""

    counts = {key: _int_value(metrics_snapshot.get(key)) for key in RECOVERY_COUNT_KEYS}
    counts["stuck_recovery_count"] += _int_value(
        metrics_snapshot.get("robot_stalled_recovery_count")
    )
    counts["stuck_recovery_count"] += _int_value(
        metrics_snapshot.get("robot_robot_deadlock_recovery_count")
    )
    counts["teleport_recovery_count"] += _int_value(metrics_snapshot.get("robot_teleport_count"))
    explicit_collision_recovery = (
        "collision_recovery_count" in metrics_snapshot
        or "collision_recoveries" in metrics_snapshot
    )
    counts["collision_recovery_count"] += _int_value(
        metrics_snapshot.get("collision_recoveries")
    )
    if explicit_collision_recovery:
        counts["collision_recovery_count"] += _int_value(metrics_snapshot.get("collision_count"))

    by_issue = _recovery_issue_counts(metrics_snapshot)
    counts["stuck_recovery_count"] += sum(
        by_issue.get(key, 0)
        for key in (
            "deadlock",
            "robot_robot_deadlock",
            "robot_stalled_without_goal_progress",
            "stuck_robot",
        )
    )
    counts["safe_reposition_count"] += sum(
        by_issue.get(key, 0)
        for key in (
            "jump_robot",
            "reposition_robot",
            "robot_safe_reposition",
            "safe_reposition",
        )
    )
    counts["teleport_recovery_count"] += sum(
        by_issue.get(key, 0)
        for key in ("teleport", "teleport_recovery", "robot_teleport")
    )
    counts["collision_recovery_count"] += sum(
        by_issue.get(key, 0)
        for key in (
            "collision_recovery",
            "robot_human_collision",
            "robot_robot_collision",
        )
    )
    return counts


def _select_strategy(fault_types: list[str], counts: JsonDict) -> str:
    if "robot_human_collision" in fault_types or "robot_robot_collision" in fault_types:
        return "collision_recovery"
    if counts.get("collision_recovery_count", 0) > 0:
        return "collision_recovery"
    if counts.get("teleport_recovery_count", 0) > 0:
        return "teleport_recovery"
    if "deadlock" in fault_types or "stuck_robot" in fault_types:
        return "stuck_recovery"
    if counts.get("stuck_recovery_count", 0) > 0:
        return "stuck_recovery"
    if counts.get("safe_reposition_count", 0) > 0:
        return "safe_reposition"
    if "unsafe_robot_human_clearance" in fault_types or "unsafe_robot_robot_clearance" in fault_types:
        return "safe_reposition"
    return "none"


def _recovery_relevant(fault_type: str) -> bool:
    return fault_type in {
        "combined_clearance_violation",
        "deadlock",
        "recovery_required",
        "robot_human_collision",
        "robot_robot_collision",
        "stuck_robot",
        "unsafe_robot_human_clearance",
        "unsafe_robot_robot_clearance",
    }


def _recovery_issue_counts(metrics_snapshot: JsonDict) -> dict[str, int]:
    recovery = _dict_value(metrics_snapshot.get("corner_case_recovery"))
    summary = _dict_value(recovery.get("summary"))
    by_issue = _dict_value(summary.get("by_issue_type"))
    if not by_issue:
        by_issue = _dict_value(metrics_snapshot.get("recovery_by_issue_type"))
    return {str(key): _int_value(value) for key, value in by_issue.items()}


def _dict_value(value: Any) -> JsonDict:
    return value if isinstance(value, dict) else {}


def _int_value(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return max(0, int(value))
    return 0

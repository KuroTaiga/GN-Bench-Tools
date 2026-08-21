"""Shared schema for human-centric DAgger samples."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any


JsonDict = dict[str, Any]
HUMAN_DAGGER_SCHEMA_VERSION = "0.1"
DAGGER_DEFAULT_PAST_FRAMES = 33
DAGGER_DEFAULT_FUTURE_ACTIONS = 32
DAGGER_DEFAULT_HISTORY_SAMPLING = "uniform"
DAGGER_DEFAULT_HISTORY_END = "current"
DAGGER_DEFAULT_STOP_RADIUS_M = 1.5
DAGGER_DEFAULT_RESCUE_HORIZON = 8
DAGGER_DEFAULT_STUCK_WINDOW = 8
DAGGER_DEFAULT_STUCK_MOTION_EPS_M = 0.03
DAGGER_DEFAULT_SAME_STATE_POS_EPS_M = 0.02
DAGGER_DEFAULT_SAME_STATE_YAW_EPS_DEG = 3.0
DAGGER_DEFAULT_STUCK_OVERRIDE_STEPS = 5
DAGGER_DEFAULT_STUCK_TAKEOVER_STEPS = 15


@dataclass(frozen=True)
class DaggerCollectionConfig:
    """Rollout-collection knobs shared by model-specific DAgger exporters."""

    past_frames: int = DAGGER_DEFAULT_PAST_FRAMES
    future_actions: int = DAGGER_DEFAULT_FUTURE_ACTIONS
    history_sampling: str = DAGGER_DEFAULT_HISTORY_SAMPLING
    history_end: str = DAGGER_DEFAULT_HISTORY_END
    stop_radius_m: float = DAGGER_DEFAULT_STOP_RADIUS_M
    rescue_horizon: int = DAGGER_DEFAULT_RESCUE_HORIZON
    stop_override: bool = True
    stuck_rescue: bool = True
    stuck_window: int = DAGGER_DEFAULT_STUCK_WINDOW
    stuck_motion_eps_m: float = DAGGER_DEFAULT_STUCK_MOTION_EPS_M
    same_state_pos_eps_m: float = DAGGER_DEFAULT_SAME_STATE_POS_EPS_M
    same_state_yaw_eps_deg: float = DAGGER_DEFAULT_SAME_STATE_YAW_EPS_DEG
    stuck_override_steps: int = DAGGER_DEFAULT_STUCK_OVERRIDE_STEPS
    stuck_takeover_steps: int = DAGGER_DEFAULT_STUCK_TAKEOVER_STEPS

    def __post_init__(self) -> None:
        history_sampling = str(self.history_sampling or DAGGER_DEFAULT_HISTORY_SAMPLING).lower()
        if history_sampling not in {"recent", "uniform"}:
            raise ValueError(
                "DAgger history_sampling must be 'recent' or 'uniform', "
                f"got {self.history_sampling!r}"
            )
        history_end = str(self.history_end or DAGGER_DEFAULT_HISTORY_END).lower()
        if history_end not in {"current", "previous"}:
            raise ValueError(
                "DAgger history_end must be 'current' or 'previous', "
                f"got {self.history_end!r}"
            )
        object.__setattr__(self, "history_sampling", history_sampling)
        object.__setattr__(self, "history_end", history_end)
        _validate_positive_int("past_frames", self.past_frames)
        _validate_positive_int("future_actions", self.future_actions)
        _validate_nonnegative_float("stop_radius_m", self.stop_radius_m)
        _validate_positive_int("rescue_horizon", self.rescue_horizon)
        _validate_positive_int("stuck_window", self.stuck_window)
        _validate_nonnegative_float("stuck_motion_eps_m", self.stuck_motion_eps_m)
        _validate_nonnegative_float("same_state_pos_eps_m", self.same_state_pos_eps_m)
        _validate_nonnegative_float("same_state_yaw_eps_deg", self.same_state_yaw_eps_deg)
        _validate_positive_int("stuck_override_steps", self.stuck_override_steps)
        _validate_positive_int("stuck_takeover_steps", self.stuck_takeover_steps)

    @property
    def future_frames(self) -> int:
        return int(self.future_actions) + 1

    @property
    def episode_length(self) -> int:
        return int(self.past_frames) + self.future_frames

    def to_json_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class DaggerFault:
    """One validation fault found in a model action."""

    fault_type: str
    mission_id: str = ""
    severity: str = "diagnostic"
    recoverable: bool = True
    details: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class DaggerValidationResult:
    """Validation output before oracle correction."""

    is_valid: bool
    faults: list[DaggerFault] = field(default_factory=list)
    details: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class DaggerOracleCorrection:
    """Mission-aware teacher correction for one model action."""

    oracle_intent: str
    action_type: str
    payload: JsonDict = field(default_factory=dict)
    low_level_actions: list[int] = field(default_factory=list)
    recovery: JsonDict | None = None
    source: str = "mission_handler"
    trainable: bool = True


@dataclass(frozen=True)
class HumanDaggerSample:
    """Model-neutral DAgger sample before rendering into model-specific format."""

    episode_id: str
    scenario_id: str
    mission_type: str
    mission_id: str
    step_index: int
    time_s: float
    model_family: str
    observation: JsonDict
    model_action: JsonDict
    validation: DaggerValidationResult
    oracle: DaggerOracleCorrection
    mission_context: JsonDict = field(default_factory=dict)
    metrics_snapshot: JsonDict = field(default_factory=dict)
    collection_config: JsonDict = field(default_factory=dict)
    collection_context: JsonDict = field(default_factory=dict)
    rendered_output: JsonDict = field(default_factory=dict)
    schema_version: str = HUMAN_DAGGER_SCHEMA_VERSION

    @property
    def trainable(self) -> bool:
        return self.oracle.trainable and bool(self.validation.faults)

    def to_json_dict(self) -> JsonDict:
        payload = asdict(self)
        payload["trainable"] = self.trainable
        return payload


def sample_from_json_dict(payload: JsonDict) -> HumanDaggerSample:
    """Rehydrate a sample dict produced by ``HumanDaggerSample.to_json_dict``."""

    validation_payload = payload.get("validation", {})
    faults = [
        DaggerFault(**fault)
        for fault in validation_payload.get("faults", [])
        if isinstance(fault, dict)
    ]
    validation = DaggerValidationResult(
        is_valid=bool(validation_payload.get("is_valid", False)),
        faults=faults,
        details=_dict_value(validation_payload.get("details")),
    )
    oracle = DaggerOracleCorrection(**_dict_value(payload.get("oracle")))
    return HumanDaggerSample(
        episode_id=str(payload.get("episode_id", "")),
        scenario_id=str(payload.get("scenario_id", "")),
        mission_type=str(payload.get("mission_type", "")),
        mission_id=str(payload.get("mission_id", "")),
        step_index=int(payload.get("step_index", 0)),
        time_s=float(payload.get("time_s", 0.0)),
        model_family=str(payload.get("model_family", "")),
        observation=_dict_value(payload.get("observation")),
        model_action=_dict_value(payload.get("model_action")),
        validation=validation,
        oracle=oracle,
        mission_context=_dict_value(payload.get("mission_context")),
        metrics_snapshot=_dict_value(payload.get("metrics_snapshot")),
        collection_config=_dict_value(payload.get("collection_config")),
        collection_context=_dict_value(payload.get("collection_context")),
        rendered_output=_dict_value(payload.get("rendered_output")),
        schema_version=str(payload.get("schema_version", HUMAN_DAGGER_SCHEMA_VERSION)),
    )


def _dict_value(value: Any) -> JsonDict:
    return value if isinstance(value, dict) else {}


def _validate_positive_int(field_name: str, value: int) -> None:
    if isinstance(value, bool) or int(value) <= 0:
        raise ValueError(f"DAgger {field_name} must be positive, got {value!r}")


def _validate_nonnegative_float(field_name: str, value: float) -> None:
    try:
        number = float(value)
    except Exception as exc:
        raise ValueError(f"DAgger {field_name} must be numeric, got {value!r}") from exc
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"DAgger {field_name} must be finite and nonnegative, got {value!r}")

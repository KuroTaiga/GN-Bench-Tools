"""Mission-aware DAgger scaffolding for human-centric GN-Bench tasks."""

from .collector import HumanDaggerCollector
from .family_specs import MissionFamilyDaggerSpec, build_default_family_specs
from .history import DaggerHistorySelection, DaggerHistorySelector
from .mission_handlers import (
    FaultCatalogEntry,
    MissionDaggerContext,
    MissionDaggerHandler,
    build_default_mission_registry,
)
from .model_adapters import ModelDaggerAdapter, build_default_model_adapter_registry
from .recovery import (
    DaggerRecoveryStrategy,
    build_recovery_strategy,
    recovery_counts_from_metrics,
)
from .schema import (
    DaggerCollectionConfig,
    DaggerFault,
    DaggerOracleCorrection,
    DaggerValidationResult,
    HumanDaggerSample,
    sample_from_json_dict,
)

__all__ = [
    "DaggerFault",
    "DaggerCollectionConfig",
    "DaggerHistorySelection",
    "DaggerHistorySelector",
    "DaggerOracleCorrection",
    "DaggerRecoveryStrategy",
    "DaggerValidationResult",
    "FaultCatalogEntry",
    "HumanDaggerCollector",
    "HumanDaggerSample",
    "MissionFamilyDaggerSpec",
    "MissionDaggerContext",
    "MissionDaggerHandler",
    "ModelDaggerAdapter",
    "build_default_mission_registry",
    "build_default_model_adapter_registry",
    "build_default_family_specs",
    "build_recovery_strategy",
    "recovery_counts_from_metrics",
    "sample_from_json_dict",
]

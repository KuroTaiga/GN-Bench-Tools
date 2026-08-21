"""Render shared human DAgger samples into model-specific training records."""

from __future__ import annotations

from .schema import HumanDaggerSample, JsonDict


class ModelDaggerAdapter:
    """Base adapter for model-specific DAgger output formats."""

    model_family = "base"

    def render(self, sample: HumanDaggerSample) -> JsonDict:
        raise NotImplementedError


class JsonPolicyDaggerAdapter(ModelDaggerAdapter):
    """Training record for JSON action policies."""

    model_family = "json_policy"

    def render(self, sample: HumanDaggerSample) -> JsonDict:
        return {
            "schema_version": sample.schema_version,
            "model_family": self.model_family,
            "episode_id": sample.episode_id,
            "mission_id": sample.mission_id,
            "input": {
                "observation": sample.observation,
                "mission_context": sample.mission_context,
                "model_action": sample.model_action,
                "faults": [fault.fault_type for fault in sample.validation.faults],
            },
            "target": {
                "robot_id": sample.oracle.payload.get(
                    "assigned_robot_id",
                    sample.mission_context.get("assigned_robot_id", ""),
                ),
                "action_type": sample.oracle.action_type,
                "payload": sample.oracle.payload,
            },
            "trainable": sample.trainable,
        }


class BAEPromptDaggerAdapter(ModelDaggerAdapter):
    """Prompt/response record for BAE-like action-token models."""

    model_family = "bae_prompt"

    def render(self, sample: HumanDaggerSample) -> JsonDict:
        instruction = sample.observation.get("instruction", "")
        if not instruction:
            instruction = sample.mission_context.get("instruction", "")
        return {
            "schema_version": sample.schema_version,
            "model_family": self.model_family,
            "episode_id": sample.episode_id,
            "mission_id": sample.mission_id,
            "input": {
                "instruction": instruction,
                "prompt_text": sample.observation.get("prompt_text", ""),
                "image_paths": sample.observation.get("image_paths", []),
                "mission_context": sample.mission_context,
                "faults": [fault.fault_type for fault in sample.validation.faults],
            },
            "target": {
                "low_level_actions": sample.oracle.low_level_actions,
                "oracle_action": {
                    "action_type": sample.oracle.action_type,
                    "payload": sample.oracle.payload,
                },
                "recovery": sample.oracle.recovery,
            },
            "trainable": sample.trainable,
        }


class VLNTrajectoryDaggerAdapter(ModelDaggerAdapter):
    """Trajectory-supervision record for VLN-style adapters."""

    model_family = "vln_trajectory"

    def render(self, sample: HumanDaggerSample) -> JsonDict:
        return {
            "schema_version": sample.schema_version,
            "model_family": self.model_family,
            "episode_id": sample.episode_id,
            "mission_id": sample.mission_id,
            "instruction": sample.observation.get(
                "instruction",
                sample.mission_context.get("instruction", ""),
            ),
            "start": sample.observation.get("agent_pose")
            or sample.observation.get("start_map_pose"),
            "goal": sample.oracle.payload.get("subgoal")
            or sample.mission_context.get("planned_goal_world"),
            "target": {
                "low_level_actions": sample.oracle.low_level_actions,
                "oracle_action": sample.oracle.payload,
            },
            "trainable": sample.trainable,
        }


class VLADaggerAdapter(ModelDaggerAdapter):
    """Observation/action record for VLA-style models."""

    model_family = "vla_action"

    def render(self, sample: HumanDaggerSample) -> JsonDict:
        return {
            "schema_version": sample.schema_version,
            "model_family": self.model_family,
            "episode_id": sample.episode_id,
            "mission_id": sample.mission_id,
            "observation": sample.observation,
            "action": {
                "intent": sample.oracle.oracle_intent,
                "action_type": sample.oracle.action_type,
                "payload": sample.oracle.payload,
                "low_level_actions": sample.oracle.low_level_actions,
                "recovery": sample.oracle.recovery,
            },
            "trainable": sample.trainable,
        }


def build_default_model_adapter_registry() -> dict[str, ModelDaggerAdapter]:
    adapters: list[ModelDaggerAdapter] = [
        JsonPolicyDaggerAdapter(),
        BAEPromptDaggerAdapter(),
        VLNTrajectoryDaggerAdapter(),
        VLADaggerAdapter(),
    ]
    return {adapter.model_family: adapter for adapter in adapters}

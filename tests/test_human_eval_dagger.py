from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from GN_Bench.human_eval.dagger import (
    DaggerCollectionConfig,
    DaggerHistorySelector,
    HumanDaggerCollector,
    build_default_mission_registry,
    build_default_model_adapter_registry,
    sample_from_json_dict,
)
from GN_Bench.human_eval.rl_task import HumanCentricRLTask
from GN_Bench.human_eval.scenario_adapter import NavDPScenarioAdapter


class HumanDaggerScaffoldTest(unittest.TestCase):
    def test_default_registries_cover_current_missions_and_model_families(self) -> None:
        mission_registry = build_default_mission_registry()
        model_registry = build_default_model_adapter_registry()

        self.assertTrue(
            {
                "deliver_to_human",
                "navigate_with_social_constraints",
                "human_guided_uncertain_region",
                "serve_queue",
                "mission_stream",
                "dense_dynamic_humans",
                "dense_multi_robot",
                "dense_dynamic_combined",
            }
            <= set(mission_registry)
        )
        self.assertTrue(
            {"json_policy", "bae_prompt", "vln_trajectory", "vla_action"}
            <= set(model_registry)
        )
        self.assertIn(
            "wrong_human_target",
            [entry.fault_type for entry in mission_registry["deliver_to_human"].fault_catalog],
        )

    def test_collector_builds_trainable_wrong_human_sample(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            episode = NavDPScenarioAdapter(navdp_root=root).load_episode(
                _write_delivery_fixture(root)
            )
            observation = HumanCentricRLTask().reset(episode)
            model_action = {
                "robot_id": "robot_alpha",
                "action_type": "assign_mission",
                "payload": {
                    "mission_id": "mission_deliver_to_human_001",
                    "target_human_id": "human_wrong",
                },
            }

            collector = HumanDaggerCollector()
            sample = collector.build_sample(
                episode,
                observation,
                model_action,
                model_family="json_policy",
                step_index=3,
                metrics_snapshot={"mission_success_rate": 0.0},
            )
            rendered = collector.render_sample(sample)

            self.assertFalse(sample.validation.is_valid)
            self.assertTrue(sample.trainable)
            self.assertEqual(sample.mission_type, "deliver_to_human")
            self.assertIn(
                "wrong_human_target",
                [fault.fault_type for fault in sample.validation.faults],
            )
            self.assertEqual(sample.oracle.payload["target_human_id"], "human_target")
            self.assertEqual(rendered["model_family"], "json_policy")
            self.assertEqual(rendered["target"]["action_type"], "assign_mission")
            self.assertIn("wrong_human_target", rendered["source_sample"]["faults"])
            self.assertEqual(
                rendered["source_sample"]["collection_config"]["history_sampling"],
                "uniform",
            )

            round_trip = sample_from_json_dict(sample.to_json_dict())
            self.assertEqual(round_trip.mission_id, sample.mission_id)
            self.assertEqual(
                round_trip.collection_config["future_actions"],
                sample.collection_config["future_actions"],
            )
            self.assertEqual(
                round_trip.validation.faults[0].fault_type,
                sample.validation.faults[0].fault_type,
            )

    def test_model_adapters_render_common_sample_shapes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            episode = NavDPScenarioAdapter(navdp_root=root).load_episode(
                _write_delivery_fixture(root)
            )
            observation = HumanCentricRLTask().reset(episode)
            observation["instruction"] = "Deliver the package to the target person."
            observation["image_paths"] = ["rgb/0.png"]
            model_action = {
                "robot_id": "robot_alpha",
                "action_type": "assign_mission",
                "payload": {
                    "mission_id": "mission_deliver_to_human_001",
                    "target_human_id": "human_wrong",
                },
            }

            collector = HumanDaggerCollector()
            sample = collector.build_sample(
                episode,
                observation,
                model_action,
                model_family="json_policy",
            )

            for model_family in ("json_policy", "bae_prompt", "vln_trajectory", "vla_action"):
                rendered = collector.render_sample(sample, model_family=model_family)
                self.assertEqual(rendered["model_family"], model_family)
                self.assertTrue(rendered["trainable"])

    def test_history_selector_matches_previous_dagger_example(self) -> None:
        selector = DaggerHistorySelector(
            DaggerCollectionConfig(history_sampling="uniform", history_end="previous")
        )
        expected = [int((step * 99) // 32) for step in range(33)]
        indices = selector.select_indices(101)

        self.assertEqual(indices, expected)
        self.assertNotIn(100, indices)

    def test_history_selector_current_preserves_vggt_behavior(self) -> None:
        selector = DaggerHistorySelector(
            DaggerCollectionConfig(history_sampling="uniform", history_end="current")
        )
        expected = [int((step * 100) // 32) for step in range(33)]
        indices = selector.select_indices(101)

        self.assertEqual(indices, expected)
        self.assertEqual(indices[-1], 100)

    def test_history_selector_previous_bootstraps_single_frame(self) -> None:
        selector = DaggerHistorySelector(
            DaggerCollectionConfig(history_sampling="uniform", history_end="previous")
        )

        self.assertEqual(selector.select_indices(1), [0] * 33)


def _write_delivery_fixture(root: Path) -> Path:
    (root / "test_scenes/demo_scene").mkdir(parents=True, exist_ok=True)
    scenario_dir = root / "scenarios"
    scenario_dir.mkdir(exist_ok=True)
    scenario_path = scenario_dir / "fixture_deliver_to_human.json"
    scenario = {
        "schema_version": "0.1",
        "scenario_id": "fixture_deliver_to_human",
        "scene_id": "demo_scene",
        "scene_assets": {
            "dataset": "fixture_dataset",
            "scene_dir": "test_scenes/demo_scene",
        },
        "robots": [
            {
                "robot_id": "robot_alpha",
                "capabilities": ["navigate", "carry"],
                "start_map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                "trajectory": [
                    {"t": 0.0, "map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 2.0, "map_pose": {"x": 2.0, "y": 0.0, "yaw": 0.0}},
                ],
            }
        ],
        "humans": [
            _human("human_target", [2.0, 0.0], "target_person"),
            _human("human_wrong", [2.0, 1.0], "visitor"),
        ],
        "missions": [
            {
                "mission_id": "mission_deliver_to_human_001",
                "mission_type": "deliver_to_human",
                "assigned_robot_id": "robot_alpha",
                "release_time": 0.0,
                "deadline": 4.0,
                "priority": 1,
                "target_human_id": "human_target",
                "success_conditions": ["correct_human_reached", "object_delivered"],
                "metadata": {"planned_goal_world": [2.0, 0.0]},
            }
        ],
        "social_structures": [],
        "event_log": {"events": []},
        "expected_result": {"passed": True, "metrics": {"fixture_expected_valid": True}},
        "metadata": {
            "collision_check": {
                "checked": True,
                "collision_free": True,
                "collision_count": 0,
            }
        },
    }
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    return scenario_path


def _human(human_id: str, xy: list[float], role: str) -> dict:
    return {
        "human_id": human_id,
        "role": role,
        "appearance": {
            "avatar_id": "avatar_shared",
            "shirt_color": "red",
            "pants_color": "black",
        },
        "start_map_pose": {"x": xy[0], "y": xy[1], "yaw": 0.0},
        "trajectory": [
            {"t": 0.0, "map_pose": {"x": xy[0], "y": xy[1], "yaw": 0.0}},
            {"t": 2.0, "map_pose": {"x": xy[0], "y": xy[1], "yaw": 0.0}},
        ],
    }


if __name__ == "__main__":
    unittest.main()

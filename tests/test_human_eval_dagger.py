from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from GN_Bench.human_eval.dagger import (
    DaggerCollectionConfig,
    DaggerHistorySelector,
    HumanDaggerCollector,
    build_default_family_specs,
    build_default_mission_registry,
    build_default_model_adapter_registry,
    recovery_counts_from_metrics,
    sample_from_json_dict,
)
from GN_Bench.human_eval.rl_task import HumanCentricRLTask
from GN_Bench.human_eval.scenario_adapter import NavDPScenarioAdapter


class HumanDaggerScaffoldTest(unittest.TestCase):
    def test_default_registries_cover_current_missions_and_model_families(self) -> None:
        mission_registry = build_default_mission_registry()
        model_registry = build_default_model_adapter_registry()
        family_specs = build_default_family_specs()

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
        self.assertEqual(set(family_specs), set(mission_registry))
        for mission_type, spec in family_specs.items():
            catalog = mission_registry[mission_type].fault_catalog
            catalog_faults = {entry.fault_type for entry in catalog}
            self.assertTrue(set(spec.primary_faults) <= catalog_faults)
            self.assertTrue(all(entry.default_recovery for entry in catalog))

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

    def test_observe_step_writes_trainable_jsonl_with_history_context(self) -> None:
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
            output_path = root / "dagger.jsonl"

            sample = HumanDaggerCollector().observe_step(
                episode,
                observation,
                model_action,
                model_family="json_policy",
                post_step_info={
                    "metrics": {
                        "wrong_human_contact_count": 1,
                        "human_identification_difficulty": 0.8,
                    }
                },
                history_frames=list(range(101)),
                output_path=output_path,
            )

            self.assertTrue(sample.trainable)
            self.assertIn(
                "unsafe_human_approach",
                [fault.fault_type for fault in sample.validation.faults],
            )
            self.assertEqual(sample.collection_context["history_frame_count"], 101)
            self.assertEqual(sample.collection_context["history_indices"][-1], 100)
            rendered = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(rendered["source_sample"]["collection_context"]["history_frame_count"], 101)

    def test_serve_queue_oracle_waits_for_pending_previous_member(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            episode = NavDPScenarioAdapter(navdp_root=root).load_episode(
                _write_serve_queue_dagger_fixture(root)
            )
            observation = HumanCentricRLTask().reset(episode)
            observation["completed_mission_ids"] = []
            model_action = {
                "robot_id": "robot_alpha",
                "action_type": "assign_mission",
                "payload": {
                    "mission_id": "mission_serve_queue_002",
                    "target_human_id": "human_b",
                },
            }

            sample = HumanDaggerCollector().build_sample(
                episode,
                observation,
                model_action,
                model_family="json_policy",
            )

            self.assertIn(
                "wait_required",
                [fault.fault_type for fault in sample.validation.faults],
            )
            self.assertEqual(sample.oracle.action_type, "no_op")
            self.assertEqual(
                sample.oracle.payload["wait_for_mission_ids"],
                ["mission_serve_queue_001"],
            )

    def test_dense_recovery_counts_and_strategy_are_attached(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            episode = NavDPScenarioAdapter(navdp_root=root).load_episode(
                _write_dense_dagger_fixture(root)
            )
            observation = HumanCentricRLTask().reset(episode)
            model_action = {
                "robot_id": "robot_alpha",
                "action_type": "set_subgoal",
                "payload": {"mission_id": "mission_dense_dynamic_humans_001"},
            }
            metrics_snapshot = {
                "collision_count": 1,
                "dense_robot_human_clearance_policy_respected": False,
                "robot_stalled_recovery_count": 1,
                "robot_teleport_count": 1,
                "corner_case_recovery": {
                    "summary": {
                        "event_count": 2,
                        "by_issue_type": {
                            "safe_reposition": 1,
                            "robot_human_collision": 1,
                        },
                    }
                },
            }

            sample = HumanDaggerCollector().build_sample(
                episode,
                observation,
                model_action,
                model_family="json_policy",
                metrics_snapshot=metrics_snapshot,
            )
            counts = recovery_counts_from_metrics(metrics_snapshot)

            self.assertEqual(counts["stuck_recovery_count"], 1)
            self.assertEqual(counts["safe_reposition_count"], 1)
            self.assertEqual(counts["teleport_recovery_count"], 1)
            self.assertEqual(counts["collision_recovery_count"], 1)
            self.assertEqual(sample.oracle.recovery["strategy"], "collision_recovery")
            self.assertIn(
                "recovery_required",
                [fault.fault_type for fault in sample.validation.faults],
            )

    def test_social_navigation_faults_follow_evaluator_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            episode = NavDPScenarioAdapter(navdp_root=root).load_episode(
                _write_social_dagger_fixture(root)
            )
            observation = HumanCentricRLTask().reset(episode)
            model_action = {
                "robot_id": "robot_alpha",
                "action_type": "set_subgoal",
                "payload": {
                    "mission_id": "mission_social_001",
                    "target_region_id": "region_goal",
                },
            }

            sample = HumanDaggerCollector().build_sample(
                episode,
                observation,
                model_action,
                model_family="json_policy",
                metrics_snapshot={
                    "goal_reached": False,
                    "minimum_goal_distance_m": 3.2,
                    "goal_threshold_m": 0.4,
                    "personal_space_respected": False,
                    "personal_space_violation_count": 1,
                    "pedestrian_yield_respected": False,
                    "pedestrian_yield_violation_count": 1,
                    "group_integrity_respected": False,
                    "group_region_violation_count": 1,
                    "queue_order_respected": False,
                    "queue_order_violation_count": 1,
                    "collision_count": 1,
                },
            )
            fault_types = [fault.fault_type for fault in sample.validation.faults]

            self.assertIn("goal_not_reached", fault_types)
            self.assertIn("personal_space_violation", fault_types)
            self.assertIn("pedestrian_yield_failure", fault_types)
            self.assertIn("group_integrity_violation", fault_types)
            self.assertIn("queue_order_violation", fault_types)
            self.assertIn("collision_or_near_miss", fault_types)
            self.assertEqual(sample.oracle.action_type, "set_subgoal")
            self.assertIn("personal_space_violation", sample.oracle.payload["social_repair_faults"])

    def test_human_guided_oracles_request_guidance_and_wait(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            episode = NavDPScenarioAdapter(navdp_root=root).load_episode(
                _write_human_guided_dagger_fixture(root)
            )
            observation = HumanCentricRLTask().reset(episode)

            request_sample = HumanDaggerCollector().build_sample(
                episode,
                observation,
                {
                    "robot_id": "robot_alpha",
                    "action_type": "set_subgoal",
                    "payload": {"mission_id": "mission_guided_001"},
                },
                model_family="json_policy",
                metrics_snapshot={"guidance_requested": False},
            )

            self.assertIn(
                "guidance_not_requested",
                [fault.fault_type for fault in request_sample.validation.faults],
            )
            self.assertEqual(request_sample.oracle.action_type, "interact")
            self.assertEqual(request_sample.oracle.payload["target_human_id"], "human_guide")

            wait_sample = HumanDaggerCollector().build_sample(
                episode,
                observation,
                {
                    "robot_id": "robot_alpha",
                    "action_type": "interact",
                    "payload": {
                        "mission_id": "mission_guided_001",
                        "interaction": "request_guidance",
                        "target_human_id": "human_guide",
                    },
                },
                model_family="json_policy",
                metrics_snapshot={
                    "guidance_requested": True,
                    "guidance_stop_required": True,
                    "guidance_stop_verified": False,
                },
            )

            self.assertIn(
                "guidance_wait_violation",
                [fault.fault_type for fault in wait_sample.validation.faults],
            )
            self.assertEqual(wait_sample.oracle.action_type, "no_op")

    def test_mission_stream_parent_and_child_faults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            episode = NavDPScenarioAdapter(navdp_root=root).load_episode(
                _write_mission_stream_dagger_fixture(root)
            )
            observation = HumanCentricRLTask().reset(episode)
            collector = HumanDaggerCollector()

            parent_sample = collector.build_sample(
                episode,
                observation,
                {
                    "robot_id": "robot_alpha",
                    "action_type": "assign_mission",
                    "payload": {"mission_id": "mission_stream_parent_001"},
                },
                model_family="json_policy",
                metrics_snapshot={
                    "child_mission_count": 2,
                    "expected_child_mission_count": 2,
                    "released_child_mission_count": 1,
                    "assigned_child_mission_count": 1,
                    "eos_child_mission_count": 0,
                    "released_missions_completed": False,
                    "priority_order_respected": False,
                    "dispatch_record_count_matches": False,
                    "stream_terminal_goals_reached": False,
                },
            )
            parent_faults = [fault.fault_type for fault in parent_sample.validation.faults]

            self.assertIn("priority_inversion", parent_faults)
            self.assertIn("missed_release", parent_faults)
            self.assertIn("wrong_child_assignment", parent_faults)
            self.assertIn("missing_eos", parent_faults)
            self.assertIn("terminal_goal_missed", parent_faults)
            self.assertEqual(parent_sample.oracle.action_type, "robot_eos")

            child_sample = collector.build_sample(
                episode,
                observation,
                {
                    "robot_id": "robot_alpha",
                    "action_type": "assign_mission",
                    "payload": {"mission_id": "mission_stream_child_001"},
                },
                model_family="json_policy",
                metrics_snapshot={
                    "release_event_present": False,
                    "assignment_event_present": False,
                    "assignment_time_matches_expected": False,
                    "timing_respected": False,
                    "stream_event_order_respected": False,
                    "eos_event_present": False,
                    "goal_reached": False,
                },
            )
            child_faults = [fault.fault_type for fault in child_sample.validation.faults]

            self.assertEqual(child_sample.mission_type, "navigate_with_social_constraints")
            self.assertIn("missed_release", child_faults)
            self.assertIn("wrong_child_assignment", child_faults)
            self.assertIn("priority_inversion", child_faults)
            self.assertIn("missing_eos", child_faults)
            self.assertIn("terminal_goal_missed", child_faults)

    def test_dense_multi_and_combined_faults_have_recovery_oracles(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            collector = HumanDaggerCollector()

            multi_episode = NavDPScenarioAdapter(navdp_root=root).load_episode(
                _write_dense_multi_dagger_fixture(root)
            )
            multi_observation = HumanCentricRLTask().reset(multi_episode)
            multi_sample = collector.build_sample(
                multi_episode,
                multi_observation,
                {
                    "robot_id": "robot_alpha",
                    "action_type": "set_subgoal",
                    "payload": {"mission_id": "mission_dense_multi_robot_001"},
                },
                model_family="json_policy",
                metrics_snapshot={
                    "all_active_robot_goal_regions_reached": False,
                    "no_robot_robot_collision": False,
                    "robot_robot_collision_count": 1,
                    "robot_robot_clearance_policy_respected": False,
                    "robot_robot_deadlock_recovery_count": 1,
                    "starvation_count": 1,
                    "robot_wait_violation_count": 1,
                    "corner_case_recovery_count": 1,
                },
            )
            multi_faults = [fault.fault_type for fault in multi_sample.validation.faults]

            self.assertIn("active_robot_goal_missed", multi_faults)
            self.assertIn("robot_robot_collision", multi_faults)
            self.assertIn("unsafe_robot_robot_clearance", multi_faults)
            self.assertIn("deadlock", multi_faults)
            self.assertIn("starvation", multi_faults)
            self.assertEqual(multi_sample.oracle.action_type, "set_subgoal")
            self.assertEqual(multi_sample.oracle.recovery["strategy"], "collision_recovery")

            combined_episode = NavDPScenarioAdapter(navdp_root=root).load_episode(
                _write_dense_combined_dagger_fixture(root)
            )
            combined_observation = HumanCentricRLTask().reset(combined_episode)
            combined_sample = collector.build_sample(
                combined_episode,
                combined_observation,
                {
                    "robot_id": "robot_alpha",
                    "action_type": "set_subgoal",
                    "payload": {"mission_id": "mission_dense_dynamic_combined_001"},
                },
                model_family="json_policy",
                metrics_snapshot={
                    "all_active_robot_goal_regions_reached": False,
                    "collision_count": 1,
                    "no_robot_robot_collision": False,
                    "robot_robot_collision_count": 1,
                    "dense_robot_human_clearance_policy_respected": False,
                    "robot_robot_clearance_policy_respected": False,
                    "dense_nominal_robot_human_conflict_count": 1,
                    "humans_keep_moving_until_robot_completion": False,
                    "robot_robot_deadlock_recovery_count": 1,
                    "mission_starvation_count": 1,
                },
            )
            combined_faults = [fault.fault_type for fault in combined_sample.validation.faults]

            self.assertIn("active_robot_goal_missed", combined_faults)
            self.assertIn("robot_human_collision", combined_faults)
            self.assertIn("robot_robot_collision", combined_faults)
            self.assertIn("combined_clearance_violation", combined_faults)
            self.assertIn("human_motion_stalled", combined_faults)
            self.assertIn("deadlock", combined_faults)
            self.assertIn("starvation", combined_faults)
            self.assertEqual(combined_sample.oracle.action_type, "set_subgoal")

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


def _write_serve_queue_dagger_fixture(root: Path) -> Path:
    (root / "test_scenes/demo_scene").mkdir(parents=True, exist_ok=True)
    scenario_dir = root / "scenarios"
    scenario_dir.mkdir(exist_ok=True)
    scenario_path = scenario_dir / "fixture_serve_queue_dagger.json"
    scenario = {
        "schema_version": "0.1",
        "scenario_id": "fixture_serve_queue_dagger",
        "scene_id": "demo_scene",
        "scene_assets": {
            "dataset": "fixture_dataset",
            "scene_dir": "test_scenes/demo_scene",
        },
        "robots": [
            {
                "robot_id": "robot_alpha",
                "capabilities": ["navigate", "serve"],
                "start_map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                "trajectory": [
                    {"t": 0.0, "map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 2.0, "map_pose": {"x": 2.0, "y": 0.0, "yaw": 0.0}},
                ],
            }
        ],
        "humans": [
            _human("human_a", [1.0, 0.0], "queue_participant"),
            _human("human_b", [2.0, 0.0], "queue_participant"),
        ],
        "missions": [
            _queue_mission(
                "mission_serve_queue_001",
                "human_a",
                queue_index=0,
                previous_human_ids=[],
                goal_xy=[1.0, 0.0],
            ),
            _queue_mission(
                "mission_serve_queue_002",
                "human_b",
                queue_index=1,
                previous_human_ids=["human_a"],
                goal_xy=[2.0, 0.0],
            ),
        ],
        "social_structures": [],
        "event_log": {"events": []},
        "expected_result": {"passed": True, "metrics": {"fixture_expected_valid": True}},
        "metadata": {"collision_check": {"checked": True, "collision_free": True}},
    }
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    return scenario_path


def _write_dense_dagger_fixture(root: Path) -> Path:
    (root / "test_scenes/demo_scene").mkdir(parents=True, exist_ok=True)
    scenario_dir = root / "scenarios"
    scenario_dir.mkdir(exist_ok=True)
    scenario_path = scenario_dir / "fixture_dense_dagger.json"
    scenario = {
        "schema_version": "0.1",
        "scenario_id": "fixture_dense_dagger",
        "scene_id": "demo_scene",
        "scene_assets": {
            "dataset": "fixture_dataset",
            "scene_dir": "test_scenes/demo_scene",
        },
        "robots": [
            {
                "robot_id": "robot_alpha",
                "capabilities": ["navigate"],
                "start_map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                "trajectory": [
                    {"t": 0.0, "map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 2.0, "map_pose": {"x": 2.0, "y": 0.0, "yaw": 0.0}},
                ],
            }
        ],
        "humans": [_human("human_dense", [1.0, 0.2], "moving_pedestrian")],
        "missions": [
            {
                "mission_id": "mission_dense_dynamic_humans_001",
                "mission_type": "dense_dynamic_humans",
                "assigned_robot_id": "robot_alpha",
                "release_time": 0.0,
                "deadline": 4.0,
                "priority": 1,
                "success_conditions": ["all_active_robot_goal_regions_reached"],
                "metadata": {
                    "active_robot_ids": ["robot_alpha"],
                    "planned_goal_world": [2.0, 0.0],
                },
            }
        ],
        "social_structures": [],
        "event_log": {"events": []},
        "expected_result": {"passed": False, "metrics": {"fixture_expected_valid": True}},
        "metadata": {"collision_check": {"checked": True, "collision_free": False}},
    }
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    return scenario_path


def _write_dense_multi_dagger_fixture(root: Path) -> Path:
    return _write_dagger_fixture(
        root,
        "fixture_dense_multi_dagger",
        [
            {
                "mission_id": "mission_dense_multi_robot_001",
                "mission_type": "dense_multi_robot",
                "assigned_robot_id": "robot_alpha",
                "release_time": 0.0,
                "deadline": 4.0,
                "priority": 1,
                "success_conditions": ["all_active_robot_goal_regions_reached"],
                "metadata": {
                    "active_robot_ids": ["robot_alpha", "robot_beta"],
                    "planned_goal_world_by_robot": {
                        "robot_alpha": [2.0, 0.0],
                        "robot_beta": [2.0, 1.0],
                    },
                    "minimum_robot_robot_distance_m": 0.6,
                },
            }
        ],
        humans=[],
    )


def _write_dense_combined_dagger_fixture(root: Path) -> Path:
    return _write_dagger_fixture(
        root,
        "fixture_dense_combined_dagger",
        [
            {
                "mission_id": "mission_dense_dynamic_combined_001",
                "mission_type": "dense_dynamic_combined",
                "assigned_robot_id": "robot_alpha",
                "release_time": 0.0,
                "deadline": 4.0,
                "priority": 1,
                "success_conditions": ["all_active_robot_goal_regions_reached"],
                "metadata": {
                    "active_robot_ids": ["robot_alpha", "robot_beta"],
                    "planned_goal_world_by_robot": {
                        "robot_alpha": [2.0, 0.0],
                        "robot_beta": [2.0, 1.0],
                    },
                    "minimum_robot_robot_distance_m": 0.6,
                    "minimum_moving_robot_human_distance_m": 0.8,
                },
            }
        ],
        humans=[_human("human_dense", [1.0, 0.2], "moving_pedestrian")],
    )


def _write_social_dagger_fixture(root: Path) -> Path:
    return _write_dagger_fixture(
        root,
        "fixture_social_dagger",
        [
            {
                "mission_id": "mission_social_001",
                "mission_type": "navigate_with_social_constraints",
                "assigned_robot_id": "robot_alpha",
                "release_time": 0.0,
                "deadline": 4.0,
                "priority": 1,
                "target_region_id": "region_goal",
                "success_conditions": ["target_region_reached"],
                "metadata": {"planned_goal_world": [2.0, 0.0]},
            }
        ],
        humans=[_human("human_social", [1.0, 0.2], "pedestrian")],
    )


def _write_human_guided_dagger_fixture(root: Path) -> Path:
    return _write_dagger_fixture(
        root,
        "fixture_human_guided_dagger",
        [
            {
                "mission_id": "mission_guided_001",
                "mission_type": "human_guided_uncertain_region",
                "assigned_robot_id": "robot_alpha",
                "release_time": 0.0,
                "deadline": 5.0,
                "priority": 1,
                "target_region_id": "uncertain_region",
                "success_conditions": ["guidance_requested", "resolved_target_reached"],
                "metadata": {
                    "planned_goal_world": [3.0, 0.0],
                    "human_guidance": {
                        "informant_human_id": "human_guide",
                        "resolved_target": {"target_world": [3.0, 0.0]},
                        "expected_robot_actions": {"must_stop_for_guidance": True},
                    },
                },
            }
        ],
        humans=[_human("human_guide", [1.0, 0.0], "informant")],
    )


def _write_mission_stream_dagger_fixture(root: Path) -> Path:
    return _write_dagger_fixture(
        root,
        "fixture_mission_stream_dagger",
        [
            {
                "mission_id": "mission_stream_parent_001",
                "mission_type": "mission_stream",
                "assigned_robot_id": "robot_alpha",
                "release_time": 0.0,
                "deadline": 5.0,
                "priority": 1,
                "success_conditions": ["mission_stream_completed"],
                "metadata": {
                    "child_mission_ids": [
                        "mission_stream_child_001",
                        "mission_stream_child_002",
                    ],
                    "configured_mission_stream_stack_size": 3,
                    "planned_goal_world_by_robot": {"robot_alpha": [2.0, 0.0]},
                },
            },
            _stream_child_mission("mission_stream_child_001", stack_index=1),
            _stream_child_mission("mission_stream_child_002", stack_index=2),
        ],
        humans=[],
    )


def _write_dagger_fixture(
    root: Path,
    scenario_id: str,
    missions: list[dict],
    *,
    humans: list[dict],
) -> Path:
    (root / "test_scenes/demo_scene").mkdir(parents=True, exist_ok=True)
    scenario_dir = root / "scenarios"
    scenario_dir.mkdir(exist_ok=True)
    scenario_path = scenario_dir / f"{scenario_id}.json"
    scenario = {
        "schema_version": "0.1",
        "scenario_id": scenario_id,
        "scene_id": "demo_scene",
        "scene_assets": {
            "dataset": "fixture_dataset",
            "scene_dir": "test_scenes/demo_scene",
        },
        "robots": [
            {
                "robot_id": "robot_alpha",
                "capabilities": ["navigate", "talk"],
                "start_map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                "trajectory": [
                    {"t": 0.0, "map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 2.0, "map_pose": {"x": 2.0, "y": 0.0, "yaw": 0.0}},
                ],
            }
        ],
        "humans": humans,
        "missions": missions,
        "social_structures": [],
        "event_log": {"events": []},
        "expected_result": {"passed": True, "metrics": {"fixture_expected_valid": True}},
        "metadata": {"collision_check": {"checked": True, "collision_free": True}},
    }
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    return scenario_path


def _stream_child_mission(mission_id: str, *, stack_index: int) -> dict:
    return {
        "mission_id": mission_id,
        "mission_type": "navigate_with_social_constraints",
        "assigned_robot_id": "robot_alpha",
        "release_time": float(stack_index),
        "deadline": 5.0,
        "priority": stack_index,
        "target_region_id": f"stream_goal_{stack_index}",
        "success_conditions": ["target_region_reached", "priority_order_respected"],
        "metadata": {
            "mission_stream_parent_id": "mission_stream_parent_001",
            "mission_stack_index": stack_index,
            "mission_stack_size": 3,
            "assignment_time_s": float(stack_index),
            "expected_completion_t": float(stack_index + 1),
            "robot_eos_t": float(stack_index + 1),
            "planned_goal_world": [float(stack_index), 0.0],
        },
    }


def _queue_mission(
    mission_id: str,
    target_human_id: str,
    *,
    queue_index: int,
    previous_human_ids: list[str],
    goal_xy: list[float],
) -> dict:
    return {
        "mission_id": mission_id,
        "mission_type": "serve_queue",
        "assigned_robot_id": "robot_alpha",
        "release_time": 0.0,
        "deadline": 4.0,
        "priority": queue_index + 1,
        "target_human_id": target_human_id,
        "success_conditions": ["nearest_queue_contact_reached", "queue_order_preserved"],
        "metadata": {
            "planned_goal_world": goal_xy,
            "previous_queue_human_ids": previous_human_ids,
            "queue_id": "queue_001",
            "queue_index": queue_index,
            "queue_order": ["human_a", "human_b"],
            "queue_position": queue_index + 1,
        },
    }


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

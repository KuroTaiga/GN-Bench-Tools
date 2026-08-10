from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from GN_Bench.config.default import CN
from GN_Bench.datasets import make_dataset
from GN_Bench.human_eval.baseline_runner import run_policy_assignment_sweep
from GN_Bench.human_eval.evaluator import HumanCentricEvaluator
from GN_Bench.human_eval.rl_task import HumanCentricRLTask
from GN_Bench.human_eval.scenario_adapter import NavDPScenarioAdapter


class NavDPHumanEvalBridgeTest(unittest.TestCase):
    def test_adapter_loads_navdp_example_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path, scenario_path = _write_navdp_fixture(root)

            episodes = NavDPScenarioAdapter(navdp_root=root).load_split(manifest_path)

            self.assertEqual(len(episodes), 1)
            episode = episodes[0]
            self.assertEqual(episode.episode_id, "fixture_deliver_to_human")
            self.assertEqual(episode.raw_scene_id, "demo_scene")
            self.assertEqual(episode.schema_version, "0.1")
            self.assertEqual(episode.mission_type, "deliver_to_human")
            self.assertEqual(episode.start_position, [1.0, 2.0])
            self.assertEqual(episode.start_rotation, [0.5])
            self.assertEqual(episode.ref_json, str(scenario_path))
            self.assertTrue(episode.scene_id.endswith("test_scenes/demo_scene"))
            self.assertEqual(
                episode.info["human_scenario"]["scenario_id"],
                "fixture_deliver_to_human",
            )

    def test_registered_dataset_loads_and_filters_human_centric_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path, _ = _write_navdp_fixture(root)

            config = CN()
            config.DATA_PATH = str(manifest_path)
            config.NAVDP_ROOT = str(root)
            config.CONTENT_SCENES = ["demo_scene"]

            dataset = make_dataset("HumanCentric-v0", config=config)
            self.assertEqual(dataset.num_episodes, 1)
            self.assertEqual(dataset.episodes[0].episode_id, "fixture_deliver_to_human")

            config.CONTENT_SCENES = ["not_this_scene"]
            filtered_dataset = make_dataset("HumanCentric-v0", config=config)
            self.assertEqual(filtered_dataset.num_episodes, 0)

    def test_replay_and_policy_sweep_produce_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path, _ = _write_navdp_fixture(root)
            episode = NavDPScenarioAdapter(navdp_root=root).load_split(manifest_path)[0]

            replay = HumanCentricEvaluator().replay(episode)
            self.assertTrue(replay.success)
            self.assertEqual(replay.metrics["mission_count"], 1)
            self.assertEqual(replay.metrics["completion_rate"], 1.0)
            self.assertEqual(replay.metrics["mission_success_rate"], 1.0)
            self.assertEqual(replay.metrics["deliver_to_human_success_count"], 1)
            self.assertEqual(replay.metrics["correct_human_reached_count"], 1)
            self.assertEqual(replay.metrics["object_delivered_count"], 1)
            self.assertEqual(replay.metrics["deadline_success_count"], 1)
            self.assertEqual(replay.metrics["duration_s"], 3.0)
            self.assertEqual(len(replay.mission_results), 1)
            self.assertTrue(replay.mission_results[0]["correct_human_reached"])
            self.assertTrue(replay.mission_results[0]["object_delivered"])
            self.assertTrue(replay.mission_results[0]["deadline_success"])
            self.assertEqual(replay.mission_results[0]["wrong_human_contact_count"], 0)

            policy_result = run_policy_assignment_sweep("oracle_human_centric", episode)
            self.assertEqual(policy_result["metrics"]["assignment_coverage"], 1.0)
            self.assertEqual(
                policy_result["metrics"]["oracle_assignment_match_rate"],
                1.0,
            )

    def test_replay_backed_rl_task_accepts_baseline_action_and_rewards(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path, _ = _write_navdp_fixture(root)
            episode = NavDPScenarioAdapter(navdp_root=root).load_split(manifest_path)[0]
            policy_result = run_policy_assignment_sweep("oracle_human_centric", episode)

            task = HumanCentricRLTask()
            observation = task.reset(episode)
            self.assertEqual(observation["mission_count"], 1)
            self.assertEqual(observation["released_missions"][0]["status"], "released")

            next_observation, reward, done, info = task.step(policy_result["actions"][0])

            self.assertTrue(done)
            self.assertEqual(reward, 1.0)
            self.assertTrue(info["replay"]["success"])
            self.assertEqual(info["reward_components"]["mission_success"], 1.0)
            self.assertEqual(next_observation["completed_mission_ids"], ["mission_deliver_to_human_001"])
            self.assertEqual(next_observation["last_replay_metrics"]["mission_success_rate"], 1.0)

    def test_replay_backed_rl_task_rejects_unknown_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path, _ = _write_navdp_fixture(root)
            episode = NavDPScenarioAdapter(navdp_root=root).load_split(manifest_path)[0]

            task = HumanCentricRLTask()
            task.reset(episode)
            with self.assertRaisesRegex(ValueError, "Unsupported human-centric action_type"):
                task.step({"action_type": "teleport", "payload": {}})

    def test_human_centric_task_registered_for_replay_and_config_mapping(self) -> None:
        try:
            from GN_Bench.tasks.registration import make_task
        except ModuleNotFoundError as error:
            if error.name == "gymnasium":
                self.skipTest("GN-Bench task registration requires gymnasium")
            raise

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path, _ = _write_navdp_fixture(root)
            episode = NavDPScenarioAdapter(navdp_root=root).load_split(manifest_path)[0]
            policy_result = run_policy_assignment_sweep("oracle_human_centric", episode)

            task = make_task("HumanCentricTask-v0")
            observation = task.reset_replay(episode)
            self.assertEqual(observation["episode_id"], "fixture_deliver_to_human")

            _, reward, done, info = task.step_replay(policy_result["actions"][0])
            self.assertTrue(done)
            self.assertEqual(reward, 1.0)
            self.assertTrue(info["replay"]["success"])
            self.assertFalse(task.is_episode_active)

            sim_config = _SimConfig()
            updated_config = task.overwrite_sim_config(sim_config, episode)
            self.assertEqual(updated_config.SCENE, episode.scene_id)
            self.assertEqual(updated_config.REF_JSON, episode.ref_json)
            self.assertEqual(updated_config.AGENT_0.START_POSITION, episode.start_position)
            self.assertEqual(updated_config.AGENT_0.START_ROTATION, episode.start_rotation)
            self.assertTrue(updated_config.AGENT_0.IS_SET_START_STATE)

    def test_social_navigation_metrics_detect_success_and_violation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            passing_path = _write_social_nav_fixture(root, "passing", human_xy=[10.0, 10.0])
            failing_path = _write_social_nav_fixture(root, "failing", human_xy=[1.0, 0.0])
            adapter = NavDPScenarioAdapter(navdp_root=root)

            passing_replay = HumanCentricEvaluator().replay(adapter.load_episode(passing_path))
            failing_replay = HumanCentricEvaluator().replay(adapter.load_episode(failing_path))

            self.assertTrue(passing_replay.success)
            self.assertEqual(passing_replay.metrics["mission_success_rate"], 1.0)
            self.assertEqual(
                passing_replay.metrics["navigate_with_social_constraints_success_count"],
                1,
            )
            self.assertEqual(
                passing_replay.mission_results[0]["personal_space_violation_count"],
                0,
            )
            self.assertTrue(failing_replay.success is False)
            self.assertEqual(failing_replay.metrics["mission_success_rate"], 0.0)
            self.assertGreater(
                failing_replay.mission_results[0]["personal_space_violation_count"],
                0,
            )
            self.assertGreater(failing_replay.mission_results[0]["collision_count"], 0)

    def test_social_navigation_l2_l3_l4_metrics_detect_law_violations(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adapter = NavDPScenarioAdapter(navdp_root=root)
            cases = [
                (
                    _write_pedestrian_yield_fixture(root, "yield_passing", robot_conflict_t=3.0),
                    True,
                    "pedestrian_yield_respected",
                    "pedestrian_yield_violation_count",
                ),
                (
                    _write_pedestrian_yield_fixture(root, "yield_failing", robot_conflict_t=1.2),
                    False,
                    "pedestrian_yield_respected",
                    "pedestrian_yield_violation_count",
                ),
                (
                    _write_group_integrity_fixture(root, "group_passing", robot_y=1.0),
                    True,
                    "group_integrity_respected",
                    "group_region_violation_count",
                ),
                (
                    _write_group_integrity_fixture(root, "group_failing", robot_y=0.25),
                    False,
                    "group_integrity_respected",
                    "group_region_violation_count",
                ),
                (
                    _write_queue_order_fixture(root, "queue_passing", terminal_xy=[0.0, 2.0]),
                    True,
                    "queue_order_respected",
                    "queue_order_violation_count",
                ),
                (
                    _write_queue_order_fixture(root, "queue_failing", terminal_xy=[0.0, 0.0]),
                    False,
                    "queue_order_respected",
                    "queue_order_violation_count",
                ),
            ]

            for path, expected_success, success_key, violation_key in cases:
                replay = HumanCentricEvaluator().replay(adapter.load_episode(path))
                mission_result = replay.mission_results[0]

                self.assertIs(replay.success, expected_success, path.name)
                self.assertIs(mission_result[success_key], expected_success, path.name)
                if expected_success:
                    self.assertEqual(mission_result[violation_key], 0, path.name)
                else:
                    self.assertGreater(mission_result[violation_key], 0, path.name)

    def test_human_guided_uncertain_region_metrics_detect_missing_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adapter = NavDPScenarioAdapter(navdp_root=root)
            passing_path = _write_human_guided_fixture(
                root,
                "guided_passing",
                include_response=True,
            )
            failing_path = _write_human_guided_fixture(
                root,
                "guided_failing",
                include_response=False,
            )

            passing_replay = HumanCentricEvaluator().replay(adapter.load_episode(passing_path))
            failing_replay = HumanCentricEvaluator().replay(adapter.load_episode(failing_path))

            self.assertTrue(passing_replay.success)
            self.assertEqual(passing_replay.metrics["human_guided_uncertain_region_success_count"], 1)
            self.assertTrue(passing_replay.mission_results[0]["guidance_stop_verified"])
            self.assertTrue(passing_replay.mission_results[0]["resolved_target_reached"])

            self.assertFalse(failing_replay.success)
            self.assertEqual(failing_replay.metrics["human_guidance_received_count"], 0)
            self.assertFalse(failing_replay.mission_results[0]["human_guidance_received"])

    def test_serve_queue_metrics_detect_order_violation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adapter = NavDPScenarioAdapter(navdp_root=root)
            passing_path = _write_serve_queue_fixture(
                root,
                "queue_service_passing",
                swapped_completion_order=False,
            )
            failing_path = _write_serve_queue_fixture(
                root,
                "queue_service_failing",
                swapped_completion_order=True,
            )

            passing_replay = HumanCentricEvaluator().replay(adapter.load_episode(passing_path))
            failing_replay = HumanCentricEvaluator().replay(adapter.load_episode(failing_path))

            self.assertTrue(passing_replay.success)
            self.assertEqual(passing_replay.metrics["serve_queue_success_count"], 2)
            self.assertEqual(passing_replay.metrics["serve_queue_order_preserved_count"], 2)

            self.assertFalse(failing_replay.success)
            self.assertLess(failing_replay.metrics["serve_queue_success_count"], 2)
            self.assertLess(failing_replay.metrics["serve_queue_order_preserved_count"], 2)

    def test_mission_stream_metrics_detect_child_timing_violation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adapter = NavDPScenarioAdapter(navdp_root=root)
            passing_path = _write_mission_stream_fixture(
                root,
                "stream_passing",
                child_two_completion_t=2.0,
            )
            failing_path = _write_mission_stream_fixture(
                root,
                "stream_failing",
                child_two_completion_t=1.0,
            )

            passing_replay = HumanCentricEvaluator().replay(adapter.load_episode(passing_path))
            failing_replay = HumanCentricEvaluator().replay(adapter.load_episode(failing_path))

            self.assertTrue(passing_replay.success)
            self.assertEqual(passing_replay.metrics["mission_stream_parent_success_count"], 1)
            self.assertEqual(passing_replay.metrics["mission_stream_child_success_count"], 2)
            self.assertEqual(passing_replay.metrics["mission_stream_child_timing_success_count"], 2)

            self.assertFalse(failing_replay.success)
            self.assertEqual(failing_replay.metrics["mission_stream_parent_success_count"], 0)
            self.assertLess(failing_replay.metrics["mission_stream_child_success_count"], 2)
            self.assertLess(failing_replay.metrics["mission_stream_child_timing_success_count"], 2)
            parent_result = failing_replay.mission_results[0]
            self.assertFalse(parent_result["priority_order_respected"])

    def test_dense_dynamic_humans_metrics_detect_collision_or_wait_violation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adapter = NavDPScenarioAdapter(navdp_root=root)
            passing_path = _write_dense_dynamic_humans_fixture(
                root,
                "dense_passing",
                collision_count=0,
                nominal_conflict_count=0,
                wait_without_blocker=False,
            )
            failing_path = _write_dense_dynamic_humans_fixture(
                root,
                "dense_failing",
                collision_count=1,
                nominal_conflict_count=1,
                wait_without_blocker=True,
            )

            passing_replay = HumanCentricEvaluator().replay(adapter.load_episode(passing_path))
            failing_replay = HumanCentricEvaluator().replay(adapter.load_episode(failing_path))

            self.assertTrue(passing_replay.success)
            self.assertEqual(passing_replay.metrics["dense_dynamic_humans_success_count"], 1)
            self.assertTrue(
                passing_replay.mission_results[0]["robots_keep_moving_when_passable"]
            )
            self.assertEqual(passing_replay.mission_results[0]["robot_wait_violation_count"], 0)

            self.assertFalse(failing_replay.success)
            self.assertEqual(failing_replay.metrics["dense_dynamic_humans_success_count"], 0)
            self.assertEqual(failing_replay.metrics["dense_robot_wait_violation_count"], 1)
            self.assertEqual(
                failing_replay.metrics["dense_nominal_robot_human_conflict_count"],
                1,
            )
            self.assertFalse(
                failing_replay.mission_results[0][
                    "dense_robot_human_clearance_policy_respected"
                ]
            )

    def test_dense_multi_robot_metrics_detect_robot_robot_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adapter = NavDPScenarioAdapter(navdp_root=root)
            passing_path = _write_dense_multi_robot_fixture(
                root,
                "dense_multi_passing",
                collision_count=0,
                closest_distance_m=2.0,
            )
            failing_path = _write_dense_multi_robot_fixture(
                root,
                "dense_multi_failing",
                collision_count=1,
                closest_distance_m=0.2,
            )

            passing_replay = HumanCentricEvaluator().replay(adapter.load_episode(passing_path))
            failing_replay = HumanCentricEvaluator().replay(adapter.load_episode(failing_path))

            self.assertTrue(passing_replay.success)
            self.assertEqual(passing_replay.metrics["dense_multi_robot_success_count"], 1)
            self.assertEqual(passing_replay.metrics["dense_multi_robot_wait_violation_count"], 0)
            self.assertTrue(passing_replay.mission_results[0]["no_robot_robot_collision"])

            self.assertFalse(failing_replay.success)
            self.assertEqual(failing_replay.metrics["dense_multi_robot_success_count"], 0)
            self.assertEqual(failing_replay.mission_results[0]["robot_robot_collision_count"], 1)
            self.assertFalse(failing_replay.mission_results[0]["no_robot_robot_collision"])
            self.assertFalse(
                failing_replay.mission_results[0][
                    "robot_robot_clearance_policy_respected"
                ]
            )

    def test_dense_dynamic_combined_metrics_detect_robot_human_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adapter = NavDPScenarioAdapter(navdp_root=root)
            passing_path = _write_dense_dynamic_combined_fixture(
                root,
                "dense_combined_passing",
                nominal_conflict_count=0,
            )
            failing_path = _write_dense_dynamic_combined_fixture(
                root,
                "dense_combined_failing",
                nominal_conflict_count=1,
            )

            passing_replay = HumanCentricEvaluator().replay(adapter.load_episode(passing_path))
            failing_replay = HumanCentricEvaluator().replay(adapter.load_episode(failing_path))

            self.assertTrue(passing_replay.success)
            self.assertEqual(passing_replay.metrics["dense_dynamic_combined_success_count"], 1)
            self.assertTrue(
                passing_replay.mission_results[0][
                    "dense_robot_human_clearance_policy_respected"
                ]
            )
            self.assertTrue(
                passing_replay.mission_results[0]["robot_robot_clearance_policy_respected"]
            )

            self.assertFalse(failing_replay.success)
            self.assertEqual(failing_replay.metrics["dense_dynamic_combined_success_count"], 0)
            self.assertEqual(
                failing_replay.metrics["dense_combined_nominal_robot_human_conflict_count"],
                1,
            )
            self.assertFalse(
                failing_replay.mission_results[0][
                    "dense_robot_human_clearance_policy_respected"
                ]
            )
            self.assertTrue(failing_replay.mission_results[0]["no_robot_robot_collision"])


def _write_navdp_fixture(root: Path) -> tuple[Path, Path]:
    (root / "test_scenes/demo_scene").mkdir(parents=True)
    scenario_dir = root / "scenarios"
    scenario_dir.mkdir()
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
                "start_map_pose": {"x": 1.0, "y": 2.0, "yaw": 0.5},
                "trajectory": [
                    {"t": 0.0, "map_pose": {"x": 1.0, "y": 2.0, "yaw": 0.5}},
                    {"t": 3.0, "map_pose": {"x": 2.0, "y": 2.0, "yaw": 0.5}},
                ],
            }
        ],
        "humans": [
            {
                "human_id": "human_target",
                "role": "target_person",
                "start_map_pose": {"x": 2.0, "y": 2.0, "yaw": 0.0},
                "trajectory": [
                    {"t": 0.0, "map_pose": {"x": 2.0, "y": 2.0, "yaw": 0.0}},
                    {"t": 3.0, "map_pose": {"x": 2.0, "y": 2.0, "yaw": 0.0}},
                ],
            }
        ],
        "missions": [
            {
                "mission_id": "mission_deliver_to_human_001",
                "mission_type": "deliver_to_human",
                "assigned_robot_id": "robot_alpha",
                "release_time": 0.0,
                "deadline": 5.0,
                "priority": 1,
                "target_human_id": "human_target",
                "success_conditions": ["correct_human_reached", "object_delivered"],
                "metadata": {"planned_goal_world": [2.0, 2.0]},
            }
        ],
        "social_structures": [],
        "event_log": {
            "events": [
                {
                    "event_id": "evt_release_001",
                    "event_type": "mission_release",
                    "mission_id": "mission_deliver_to_human_001",
                    "actor_id": None,
                    "payload": {},
                    "t": 0.0,
                },
                {
                    "event_id": "evt_assign_001",
                    "event_type": "robot_assignment",
                    "mission_id": "mission_deliver_to_human_001",
                    "actor_id": "robot_alpha",
                    "payload": {},
                    "t": 0.5,
                },
                {
                    "event_id": "evt_complete_001",
                    "event_type": "completion",
                    "mission_id": "mission_deliver_to_human_001",
                    "actor_id": "robot_alpha",
                    "payload": {},
                    "t": 3.0,
                },
            ]
        },
        "expected_result": {
            "passed": True,
            "metrics": {"fixture_expected_valid": True},
        },
        "metadata": {
            "collision_check": {
                "collision_free": True,
                "collision_count": 0,
                "min_clearance_m": 1.0,
            }
        },
    }
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")

    manifest_dir = root / "splits"
    manifest_dir.mkdir()
    manifest_path = manifest_dir / "example_manifest.json"
    manifest_path.write_text(
        json.dumps({"examples": [{"path": "scenarios/fixture_deliver_to_human.json"}]}),
        encoding="utf-8",
    )
    return manifest_path, scenario_path


def _write_social_nav_fixture(root: Path, scenario_id: str, human_xy: list[float]) -> Path:
    (root / "test_scenes/social_scene").mkdir(parents=True, exist_ok=True)
    scenario_dir = root / "scenarios"
    scenario_dir.mkdir(exist_ok=True)
    scenario_path = scenario_dir / f"{scenario_id}.json"
    scenario = {
        "schema_version": "0.1",
        "scenario_id": scenario_id,
        "scene_id": "social_scene",
        "scene_assets": {
            "dataset": "fixture_dataset",
            "scene_dir": "test_scenes/social_scene",
        },
        "robots": [
            {
                "robot_id": "robot_beta",
                "capabilities": ["navigate"],
                "start_map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                "trajectory": [
                    {"t": 0.0, "map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 1.0, "map_pose": {"x": 1.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 2.0, "map_pose": {"x": 2.0, "y": 0.0, "yaw": 0.0}},
                ],
            }
        ],
        "humans": [
            {
                "human_id": "human_l1_01",
                "role": "bystander",
                "social_defaults": {
                    "personal_space_radius": 0.8,
                    "metadata": {
                        "collision_geometry": {
                            "radius_m": 0.4,
                            "shape": "cylinder",
                        }
                    },
                },
                "start_map_pose": {"x": human_xy[0], "y": human_xy[1], "yaw": 0.0},
                "trajectory": [
                    {
                        "t": 0.0,
                        "map_pose": {"x": human_xy[0], "y": human_xy[1], "yaw": 0.0},
                    },
                    {
                        "t": 2.0,
                        "map_pose": {"x": human_xy[0], "y": human_xy[1], "yaw": 0.0},
                    },
                ],
            }
        ],
        "missions": [
            {
                "mission_id": "mission_social_nav_001",
                "mission_type": "navigate_with_social_constraints",
                "assigned_robot_id": "robot_beta",
                "release_time": 0.0,
                "deadline": 3.0,
                "priority": 1,
                "success_conditions": [
                    "goal_reached",
                    "personal_space_respected",
                    "deadline_success",
                ],
                "metadata": {
                    "planned_goal_world": [2.0, 0.0],
                    "goal_tolerance_m": 0.1,
                    "active_social_law_ids": ["L1_personal_space"],
                },
            }
        ],
        "social_structures": [
            {
                "structure_id": "personal_space_buffer_001",
                "structure_type": "vulnerable_buffer",
                "human_ids": ["human_l1_01"],
                "rules": {
                    "personal_space_radius_m": 0.8,
                    "task_relevant_human_ids": [],
                },
            }
        ],
        "event_log": {
            "events": [
                {
                    "event_id": "evt_release_001",
                    "event_type": "mission_release",
                    "mission_id": "mission_social_nav_001",
                    "actor_id": None,
                    "payload": {},
                    "t": 0.0,
                },
                {
                    "event_id": "evt_assign_001",
                    "event_type": "robot_assignment",
                    "mission_id": "mission_social_nav_001",
                    "actor_id": "robot_beta",
                    "payload": {},
                    "t": 0.1,
                },
            ]
        },
        "expected_result": {
            "passed": True,
            "metrics": {"fixture_expected_valid": True},
        },
        "metadata": {
            "collision_defaults": {
                "robot_cylinder_radius_m": 0.3,
                "human_cylinder_radius_m": 0.4,
                "human_personal_space_radius_m": 0.8,
            },
            "collision_check": {
                "collision_free": True,
                "collision_count": 0,
            },
        },
    }
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    return scenario_path


def _write_pedestrian_yield_fixture(
    root: Path,
    scenario_id: str,
    *,
    robot_conflict_t: float,
) -> Path:
    return _write_social_law_fixture(
        root,
        scenario_id,
        law_id="L2_non_obstruction_yield",
        law_case_id="pedestrian_yield",
        goal_xy=[0.0, 2.0],
        deadline=5.0,
        robot_trajectory=[
            {"t": 0.0, "map_pose": {"x": 0.0, "y": -1.0, "yaw": 0.0}},
            {"t": robot_conflict_t, "map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
            {"t": 4.0, "map_pose": {"x": 0.0, "y": 2.0, "yaw": 0.0}},
        ],
        humans=[
            _human(
                "human_l2_pedestrian",
                "pedestrian_with_right_of_way",
                [
                    {"t": 0.0, "map_pose": {"x": -1.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 1.0, "map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 2.0, "map_pose": {"x": 1.0, "y": 0.0, "yaw": 0.0}},
                ],
            )
        ],
        success_conditions=["target_region_reached", "non_obstruction_yield_respected"],
        social_structures=[
            {
                "structure_id": "pedestrian_flow_001",
                "structure_type": "pedestrian_flow",
                "human_ids": ["human_l2_pedestrian"],
                "law_ids": ["L2_non_obstruction_yield"],
                "geometry": {
                    "conflict_point": [0.0, 0.0],
                    "conflict_radius_m": 0.25,
                },
                "rules": {
                    "human_id": "human_l2_pedestrian",
                    "human_right_of_way": True,
                    "conflict_radius_m": 0.25,
                    "yield_margin_s": 1.0,
                    "robots_responsible_for_avoidance": True,
                    "law_ids": ["L2_non_obstruction_yield"],
                },
            }
        ],
    )


def _write_group_integrity_fixture(
    root: Path,
    scenario_id: str,
    *,
    robot_y: float,
) -> Path:
    return _write_social_law_fixture(
        root,
        scenario_id,
        law_id="L3_group_integrity",
        law_case_id="group_integrity",
        goal_xy=[3.0, robot_y],
        deadline=5.0,
        robot_trajectory=[
            {"t": 0.0, "map_pose": {"x": -1.0, "y": robot_y, "yaw": 0.0}},
            {"t": 2.0, "map_pose": {"x": 1.0, "y": robot_y, "yaw": 0.0}},
            {"t": 4.0, "map_pose": {"x": 3.0, "y": robot_y, "yaw": 0.0}},
        ],
        humans=[
            _human(
                "human_l3_group_01",
                "conversation_group_member",
                [
                    {"t": 0.0, "map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 5.0, "map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
                ],
            ),
            _human(
                "human_l3_group_02",
                "conversation_group_member",
                [
                    {"t": 0.0, "map_pose": {"x": 2.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 5.0, "map_pose": {"x": 2.0, "y": 0.0, "yaw": 0.0}},
                ],
            ),
        ],
        success_conditions=["target_region_reached", "group_integrity_law_respected"],
        social_structures=[
            {
                "structure_id": "conversation_group_001",
                "structure_type": "f_formation",
                "human_ids": ["human_l3_group_01", "human_l3_group_02"],
                "law_ids": ["L3_group_integrity"],
                "geometry": {
                    "law_region_hull_world": [[0.0, 0.0], [2.0, 0.0]],
                    "law_region_inflation_radius_m": 0.4,
                    "law_region_shape": "capsule",
                },
                "rules": {
                    "group_id": "conversation_group_001",
                    "law_region_inflation_radius_m": 0.4,
                    "no_crossing_o_space": True,
                    "no_split_group": True,
                    "law_ids": ["L3_group_integrity"],
                },
            }
        ],
    )


def _write_queue_order_fixture(
    root: Path,
    scenario_id: str,
    *,
    terminal_xy: list[float],
) -> Path:
    return _write_social_law_fixture(
        root,
        scenario_id,
        law_id="L4_queue_order",
        law_case_id="queue_order",
        goal_xy=[0.0, 2.0],
        deadline=5.0,
        robot_trajectory=[
            {"t": 0.0, "map_pose": {"x": -1.0, "y": 0.0, "yaw": 0.0}},
            {"t": 4.0, "map_pose": {"x": terminal_xy[0], "y": terminal_xy[1], "yaw": 0.0}},
        ],
        humans=[
            _human(
                "human_l4_queue_01",
                "queue_participant",
                [
                    {"t": 0.0, "map_pose": {"x": 0.0, "y": 0.6, "yaw": 0.0}},
                    {"t": 5.0, "map_pose": {"x": 0.0, "y": 0.6, "yaw": 0.0}},
                ],
            ),
            _human(
                "human_l4_queue_02",
                "queue_participant",
                [
                    {"t": 0.0, "map_pose": {"x": 0.0, "y": 1.1, "yaw": 0.0}},
                    {"t": 5.0, "map_pose": {"x": 0.0, "y": 1.1, "yaw": 0.0}},
                ],
            ),
        ],
        success_conditions=["target_region_reached", "queue_order_law_respected"],
        social_structures=[
            {
                "structure_id": "service_queue_001",
                "structure_type": "queue",
                "human_ids": ["human_l4_queue_01", "human_l4_queue_02"],
                "law_ids": ["L4_queue_order"],
                "geometry": {
                    "queue_tail_goal": [0.0, 2.0],
                    "service_point": [0.0, 0.0],
                    "line": [[0.0, 0.6], [0.0, 1.1]],
                },
                "rules": {
                    "lawful_endpoint": "queue_tail",
                    "invalid_direct_endpoint": "service_point",
                    "queue_count": 2,
                    "queue_order": ["human_l4_queue_01", "human_l4_queue_02"],
                    "robot_entry_index": 2,
                    "service_point": [0.0, 0.0],
                    "law_ids": ["L4_queue_order"],
                },
            }
        ],
    )


def _write_social_law_fixture(
    root: Path,
    scenario_id: str,
    *,
    law_id: str,
    law_case_id: str,
    goal_xy: list[float],
    deadline: float,
    robot_trajectory: list[dict],
    humans: list[dict],
    success_conditions: list[str],
    social_structures: list[dict],
) -> Path:
    (root / "test_scenes/social_scene").mkdir(parents=True, exist_ok=True)
    scenario_dir = root / "scenarios"
    scenario_dir.mkdir(exist_ok=True)
    scenario_path = scenario_dir / f"{scenario_id}.json"
    scenario = {
        "schema_version": "0.1",
        "scenario_id": scenario_id,
        "scene_id": "social_scene",
        "scene_assets": {
            "dataset": "fixture_dataset",
            "scene_dir": "test_scenes/social_scene",
        },
        "robots": [
            {
                "robot_id": "robot_beta",
                "capabilities": ["navigate"],
                "start_map_pose": robot_trajectory[0]["map_pose"],
                "trajectory": robot_trajectory,
            }
        ],
        "humans": humans,
        "missions": [
            {
                "mission_id": "mission_social_nav_001",
                "mission_type": "navigate_with_social_constraints",
                "assigned_robot_id": "robot_beta",
                "release_time": 0.0,
                "deadline": deadline,
                "priority": 1,
                "success_conditions": success_conditions,
                "social_law_ids": [law_id],
                "metadata": {
                    "planned_goal_world": goal_xy,
                    "goal_tolerance_m": 0.1,
                    "active_social_law_ids": [law_id],
                    "law_case_id": law_case_id,
                    "mission_end_time_s": deadline,
                },
            }
        ],
        "social_structures": social_structures,
        "event_log": {
            "events": [
                {
                    "event_id": "evt_release_001",
                    "event_type": "mission_release",
                    "mission_id": "mission_social_nav_001",
                    "actor_id": None,
                    "payload": {},
                    "t": 0.0,
                },
                {
                    "event_id": "evt_assign_001",
                    "event_type": "robot_assignment",
                    "mission_id": "mission_social_nav_001",
                    "actor_id": "robot_beta",
                    "payload": {},
                    "t": 0.1,
                },
            ]
        },
        "expected_result": {
            "passed": True,
            "metrics": {"fixture_expected_valid": True},
        },
        "metadata": {
            "collision_defaults": {
                "robot_cylinder_radius_m": 0.3,
                "human_cylinder_radius_m": 0.4,
                "human_personal_space_radius_m": 0.8,
            },
            "collision_check": {
                "collision_free": True,
                "collision_count": 0,
            },
        },
    }
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    return scenario_path


def _write_human_guided_fixture(
    root: Path,
    scenario_id: str,
    *,
    include_response: bool,
) -> Path:
    (root / "test_scenes/social_scene").mkdir(parents=True, exist_ok=True)
    scenario_dir = root / "scenarios"
    scenario_dir.mkdir(exist_ok=True)
    scenario_path = scenario_dir / f"{scenario_id}.json"
    events = [
        {
            "event_id": "evt_release_001",
            "event_type": "mission_release",
            "mission_id": "mission_human_guided_uncertain_region_001",
            "actor_id": None,
            "payload": {},
            "t": 0.0,
        },
        {
            "event_id": "evt_assign_001",
            "event_type": "robot_assignment",
            "mission_id": "mission_human_guided_uncertain_region_001",
            "actor_id": "robot_alpha",
            "payload": {},
            "t": 0.1,
        },
        {
            "event_id": "evt_guidance_request_001",
            "event_type": "human_guidance_request",
            "mission_id": "mission_human_guided_uncertain_region_001",
            "actor_id": "robot_alpha",
            "payload": {"to_human_id": "human_informant"},
            "t": 1.0,
        },
    ]
    if include_response:
        events.extend(
            [
                {
                    "event_id": "evt_guidance_response_001",
                    "event_type": "human_guidance_response",
                    "mission_id": "mission_human_guided_uncertain_region_001",
                    "actor_id": "human_informant",
                    "payload": {"to_robot_id": "robot_alpha"},
                    "t": 1.5,
                },
                {
                    "event_id": "evt_uncertainty_resolved_001",
                    "event_type": "uncertainty_resolved",
                    "mission_id": "mission_human_guided_uncertain_region_001",
                    "actor_id": "robot_alpha",
                    "payload": {"target_world": [2.0, 0.0]},
                    "t": 1.6,
                },
            ]
        )
    scenario = {
        "schema_version": "0.1",
        "scenario_id": scenario_id,
        "scene_id": "social_scene",
        "scene_assets": {
            "dataset": "fixture_dataset",
            "scene_dir": "test_scenes/social_scene",
        },
        "robots": [
            {
                "robot_id": "robot_alpha",
                "capabilities": ["navigate", "talk"],
                "start_map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                "trajectory": [
                    {"t": 0.0, "map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
                    {
                        "t": 1.0,
                        "motion_state": "waiting_for_guidance",
                        "map_pose": {"x": 0.5, "y": 0.0, "yaw": 0.0},
                    },
                    {
                        "t": 1.5,
                        "motion_state": "waiting_for_guidance",
                        "map_pose": {"x": 0.5, "y": 0.0, "yaw": 0.0},
                    },
                    {"t": 3.0, "map_pose": {"x": 2.0, "y": 0.0, "yaw": 0.0}},
                ],
            }
        ],
        "humans": [
            _human(
                "human_informant",
                "informant",
                [
                    {"t": 0.0, "map_pose": {"x": 0.5, "y": 1.0, "yaw": 0.0}},
                    {"t": 3.0, "map_pose": {"x": 0.5, "y": 1.0, "yaw": 0.0}},
                ],
            )
        ],
        "missions": [
            {
                "mission_id": "mission_human_guided_uncertain_region_001",
                "mission_type": "human_guided_uncertain_region",
                "assigned_robot_id": "robot_alpha",
                "release_time": 0.0,
                "deadline": 4.0,
                "priority": 2,
                "target_region_id": "region_uncertain_placeholder",
                "success_conditions": [
                    "guidance_requested",
                    "human_guidance_received",
                    "uncertainty_resolved",
                    "resolved_target_reached",
                ],
                "metadata": {
                    "planned_goal_world": [2.0, 0.0],
                    "human_guidance": {
                        "informant_human_id": "human_informant",
                        "resolved_target": {"target_world": [2.0, 0.0]},
                        "expected_robot_actions": {
                            "must_stop_for_guidance": True,
                            "stop_interval_s": [1.0, 1.5],
                        },
                    },
                },
            }
        ],
        "social_structures": [],
        "event_log": {"events": events},
        "expected_result": {
            "passed": True,
            "metrics": {"fixture_expected_valid": True},
        },
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


def _write_serve_queue_fixture(
    root: Path,
    scenario_id: str,
    *,
    swapped_completion_order: bool,
) -> Path:
    (root / "test_scenes/social_scene").mkdir(parents=True, exist_ok=True)
    scenario_dir = root / "scenarios"
    scenario_dir.mkdir(exist_ok=True)
    scenario_path = scenario_dir / f"{scenario_id}.json"
    completion_1 = 3.0 if swapped_completion_order else 1.0
    completion_2 = 1.0 if swapped_completion_order else 3.0
    scenario = {
        "schema_version": "0.1",
        "scenario_id": scenario_id,
        "scene_id": "social_scene",
        "scene_assets": {
            "dataset": "fixture_dataset",
            "scene_dir": "test_scenes/social_scene",
        },
        "robots": [
            {
                "robot_id": "robot_alpha",
                "capabilities": ["navigate", "talk"],
                "start_map_pose": {"x": -1.0, "y": 0.0, "yaw": 0.0},
                "trajectory": [
                    {"t": 0.0, "map_pose": {"x": -1.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 1.0, "map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 3.0, "map_pose": {"x": 2.0, "y": 0.0, "yaw": 0.0}},
                ],
            }
        ],
        "humans": [
            _human(
                "human_a",
                "queue_participant",
                [
                    {"t": 0.0, "map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 4.0, "map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
                ],
            ),
            _human(
                "human_b",
                "queue_participant",
                [
                    {"t": 0.0, "map_pose": {"x": 2.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 4.0, "map_pose": {"x": 2.0, "y": 0.0, "yaw": 0.0}},
                ],
            ),
        ],
        "missions": [
            _serve_queue_mission(
                "mission_serve_queue_001",
                "human_a",
                release_time=0.0,
                deadline=2.0,
                queue_index=0,
                previous_human_ids=[],
                goal_xy=[0.0, 0.0],
            ),
            _serve_queue_mission(
                "mission_serve_queue_002",
                "human_b",
                release_time=1.5,
                deadline=4.0,
                queue_index=1,
                previous_human_ids=["human_a"],
                goal_xy=[2.0, 0.0],
            ),
        ],
        "social_structures": [
            {
                "structure_id": "queue_001",
                "structure_type": "queue",
                "human_ids": ["human_a", "human_b"],
                "law_ids": ["L4_queue_order"],
                "geometry": {
                    "service_contact_points": {
                        "human_a": [0.0, 0.0],
                        "human_b": [2.0, 0.0],
                    }
                },
                "rules": {
                    "queue_order": ["human_a", "human_b"],
                    "service_mission_order": [
                        "mission_serve_queue_001",
                        "mission_serve_queue_002",
                    ],
                    "must_process_in_order": True,
                    "law_ids": ["L4_queue_order"],
                },
            }
        ],
        "event_log": {
            "events": [
                {
                    "event_id": "evt_release_001",
                    "event_type": "mission_release",
                    "mission_id": "mission_serve_queue_001",
                    "actor_id": None,
                    "payload": {},
                    "t": 0.0,
                },
                {
                    "event_id": "evt_assign_001",
                    "event_type": "robot_assignment",
                    "mission_id": "mission_serve_queue_001",
                    "actor_id": "robot_alpha",
                    "payload": {},
                    "t": 0.1,
                },
                {
                    "event_id": "evt_release_002",
                    "event_type": "mission_release",
                    "mission_id": "mission_serve_queue_002",
                    "actor_id": None,
                    "payload": {},
                    "t": 1.5,
                },
                {
                    "event_id": "evt_assign_002",
                    "event_type": "robot_assignment",
                    "mission_id": "mission_serve_queue_002",
                    "actor_id": "robot_alpha",
                    "payload": {},
                    "t": 1.6,
                },
                {
                    "event_id": "evt_complete_001",
                    "event_type": "completion",
                    "mission_id": "mission_serve_queue_001",
                    "actor_id": "robot_alpha",
                    "payload": {},
                    "t": completion_1,
                },
                {
                    "event_id": "evt_complete_002",
                    "event_type": "completion",
                    "mission_id": "mission_serve_queue_002",
                    "actor_id": "robot_alpha",
                    "payload": {},
                    "t": completion_2,
                },
            ]
        },
        "expected_result": {
            "passed": True,
            "metrics": {"fixture_expected_valid": True},
        },
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


def _write_mission_stream_fixture(
    root: Path,
    scenario_id: str,
    *,
    child_two_completion_t: float,
) -> Path:
    (root / "test_scenes/social_scene").mkdir(parents=True, exist_ok=True)
    scenario_dir = root / "scenarios"
    scenario_dir.mkdir(exist_ok=True)
    scenario_path = scenario_dir / f"{scenario_id}.json"
    scenario = {
        "schema_version": "0.1",
        "scenario_id": scenario_id,
        "scene_id": "social_scene",
        "scene_assets": {
            "dataset": "fixture_dataset",
            "scene_dir": "test_scenes/social_scene",
        },
        "robots": [
            {
                "robot_id": "robot_alpha",
                "capabilities": ["navigate"],
                "start_map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                "trajectory": [
                    {"t": 0.0, "map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 1.0, "map_pose": {"x": 1.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 2.0, "map_pose": {"x": 2.0, "y": 0.0, "yaw": 0.0}},
                ],
            }
        ],
        "humans": [],
        "missions": [
            {
                "mission_id": "mission_stream_parent_001",
                "mission_type": "mission_stream",
                "assigned_robot_id": "robot_alpha",
                "release_time": 0.0,
                "deadline": 2.0,
                "priority": 0,
                "success_conditions": [
                    "released_missions_completed",
                    "priority_order_respected",
                ],
                "metadata": {
                    "child_mission_ids": [
                        "mission_stream_navigation_child_001",
                        "mission_stream_navigation_child_002",
                    ],
                    "configured_mission_stream_stack_size": 3,
                    "dispatch_record_count": 2,
                    "expected_completion_t": 2.0,
                    "planned_goal_world_by_robot": {
                        "robot_alpha": [2.0, 0.0],
                    },
                    "robot_eos_event_type": "robot_eos",
                    "dispatch_policy": "earliest_ready_robot_gets_next_stack_item_after_EOS",
                },
            },
            _mission_stream_child(
                "mission_stream_navigation_child_001",
                release_time=0.0,
                expected_completion_t=1.0,
                stack_index=1,
                goal_xy=[1.0, 0.0],
            ),
            _mission_stream_child(
                "mission_stream_navigation_child_002",
                release_time=1.0,
                expected_completion_t=2.0,
                stack_index=2,
                goal_xy=[2.0, 0.0],
            ),
        ],
        "social_structures": [],
        "event_log": {
            "events": [
                _stream_event("evt_release_parent", "mission_release", "mission_stream_parent_001", 0.0),
                _stream_event("evt_assign_parent", "robot_assignment", "mission_stream_parent_001", 0.0),
                _stream_event("evt_release_001", "mission_release", "mission_stream_navigation_child_001", 0.0),
                _stream_event("evt_assign_001", "robot_assignment", "mission_stream_navigation_child_001", 0.0),
                _stream_event("evt_complete_001", "completion", "mission_stream_navigation_child_001", 1.0),
                _stream_event("evt_eos_001", "robot_eos", "mission_stream_navigation_child_001", 1.0),
                _stream_event("evt_release_002", "mission_release", "mission_stream_navigation_child_002", 1.0),
                _stream_event("evt_assign_002", "robot_assignment", "mission_stream_navigation_child_002", 1.0),
                _stream_event(
                    "evt_complete_002",
                    "completion",
                    "mission_stream_navigation_child_002",
                    child_two_completion_t,
                ),
                _stream_event(
                    "evt_eos_002",
                    "robot_eos",
                    "mission_stream_navigation_child_002",
                    child_two_completion_t,
                ),
                _stream_event("evt_complete_parent", "completion", "mission_stream_parent_001", 2.0),
            ]
        },
        "expected_result": {
            "passed": True,
            "metrics": {"fixture_expected_valid": True},
        },
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


def _mission_stream_child(
    mission_id: str,
    *,
    release_time: float,
    expected_completion_t: float,
    stack_index: int,
    goal_xy: list[float],
) -> dict:
    return {
        "mission_id": mission_id,
        "mission_type": "navigate_with_social_constraints",
        "assigned_robot_id": "robot_alpha",
        "release_time": release_time,
        "deadline": 2.0,
        "priority": stack_index,
        "target_region_id": "stream_goal",
        "success_conditions": [
            "target_region_reached",
            "priority_order_respected",
        ],
        "metadata": {
            "mission_stream_parent_id": "mission_stream_parent_001",
            "mission_stack_index": stack_index,
            "mission_stack_size": 3,
            "assignment_time_s": release_time,
            "move_start_t": release_time,
            "robot_arrival_t": expected_completion_t,
            "robot_eos_t": expected_completion_t,
            "expected_completion_t": expected_completion_t,
            "planned_goal_world": goal_xy,
        },
    }


def _stream_event(event_id: str, event_type: str, mission_id: str, t: float) -> dict:
    return {
        "event_id": event_id,
        "event_type": event_type,
        "mission_id": mission_id,
        "actor_id": "robot_alpha" if event_type != "mission_release" else None,
        "payload": {},
        "t": t,
    }


def _write_dense_dynamic_humans_fixture(
    root: Path,
    scenario_id: str,
    *,
    collision_count: int,
    nominal_conflict_count: int,
    wait_without_blocker: bool,
) -> Path:
    (root / "test_scenes/social_scene").mkdir(parents=True, exist_ok=True)
    scenario_dir = root / "scenarios"
    scenario_dir.mkdir(exist_ok=True)
    scenario_path = scenario_dir / f"{scenario_id}.json"
    yield_blockers = {} if wait_without_blocker else {"human_dense_01": 2}
    scenario = {
        "schema_version": "0.1",
        "scenario_id": scenario_id,
        "scene_id": "social_scene",
        "scene_assets": {
            "dataset": "fixture_dataset",
            "scene_dir": "test_scenes/social_scene",
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
        "humans": [
            _human(
                "human_dense_01",
                "moving_pedestrian",
                [
                    {"t": 0.0, "map_pose": {"x": 10.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 3.0, "map_pose": {"x": 12.0, "y": 0.0, "yaw": 0.0}},
                ],
            )
        ],
        "missions": [
            {
                "mission_id": "mission_dense_dynamic_humans_001",
                "mission_type": "dense_dynamic_humans",
                "assigned_robot_id": "robot_alpha",
                "release_time": 0.0,
                "deadline": 3.0,
                "priority": 1,
                "success_conditions": [
                    "all_active_robot_goal_regions_reached",
                    "robots_keep_moving_when_passable",
                    "wait_only_when_immediately_blocked",
                    "dense_robot_human_clearance_policy_respected",
                    "humans_keep_moving_until_robot_completion",
                    "human_human_collision_free",
                ],
                "metadata": {
                    "active_robot_ids": ["robot_alpha"],
                    "planned_goal_world_by_robot": {"robot_alpha": [2.0, 0.0]},
                    "goal_tolerance_m": 0.1,
                    "minimum_human_human_distance_m": 0.05,
                },
            }
        ],
        "social_structures": [],
        "event_log": {
            "events": [
                _stream_event(
                    "evt_release_001",
                    "mission_release",
                    "mission_dense_dynamic_humans_001",
                    0.0,
                ),
                _stream_event(
                    "evt_assign_001",
                    "robot_assignment",
                    "mission_dense_dynamic_humans_001",
                    0.1,
                ),
            ]
        },
        "expected_result": {
            "passed": collision_count == 0
            and nominal_conflict_count == 0
            and not wait_without_blocker,
            "metrics": {"fixture_expected_valid": True},
        },
        "metadata": {
            "collision_check": {
                "checked": True,
                "collision_free": collision_count == 0,
                "collision_count": collision_count,
            },
            "dense_dynamic_humans": {
                "nominal_robot_human_conflict_samples": {
                    "robot_alpha": nominal_conflict_count,
                },
                "robot_adjustments": {
                    "robot_alpha": {
                        "wait_count": 1,
                        "inserted_waits": 1,
                        "stand_ground_count": 1,
                        "yield_count": 1,
                        "teleport_count": 0,
                        "yield_blockers": yield_blockers,
                    }
                },
                "corner_case_recovery": {"summary": {"event_count": 0}},
            },
        },
    }
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    return scenario_path


def _write_dense_multi_robot_fixture(
    root: Path,
    scenario_id: str,
    *,
    collision_count: int,
    closest_distance_m: float,
) -> Path:
    (root / "test_scenes/social_scene").mkdir(parents=True, exist_ok=True)
    scenario_dir = root / "scenarios"
    scenario_dir.mkdir(exist_ok=True)
    scenario_path = scenario_dir / f"{scenario_id}.json"
    scenario = {
        "schema_version": "0.1",
        "scenario_id": scenario_id,
        "scene_id": "social_scene",
        "scene_assets": {
            "dataset": "fixture_dataset",
            "scene_dir": "test_scenes/social_scene",
        },
        "robots": [
            {
                "robot_id": "robot_001",
                "capabilities": ["navigate"],
                "start_map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                "trajectory": [
                    {"t": 0.0, "map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 2.0, "map_pose": {"x": 2.0, "y": 0.0, "yaw": 0.0}},
                ],
            },
            {
                "robot_id": "robot_002",
                "capabilities": ["navigate"],
                "start_map_pose": {"x": 0.0, "y": 2.0, "yaw": 0.0},
                "trajectory": [
                    {"t": 0.0, "map_pose": {"x": 0.0, "y": 2.0, "yaw": 0.0}},
                    {"t": 2.0, "map_pose": {"x": 2.0, "y": 2.0, "yaw": 0.0}},
                ],
            },
        ],
        "humans": [],
        "missions": [
            {
                "mission_id": "mission_dense_multi_robot_001",
                "mission_type": "dense_multi_robot",
                "assigned_robot_id": "robot_001",
                "release_time": 0.0,
                "deadline": 3.0,
                "priority": 1,
                "success_conditions": [
                    "all_active_robot_goal_regions_reached",
                    "robots_keep_moving_when_passable",
                    "wait_only_when_immediately_blocked",
                    "no_robot_robot_collision",
                ],
                "metadata": {
                    "active_robot_ids": ["robot_001", "robot_002"],
                    "active_robot_count": 2,
                    "planned_goal_world_by_robot": {
                        "robot_001": [2.0, 0.0],
                        "robot_002": [2.0, 2.0],
                    },
                    "goal_tolerance_m": 0.1,
                    "minimum_robot_robot_distance_m": 0.6,
                    "mission_end_time_s": 3.0,
                    "expected_robot_motion": {"requires_stops": True},
                },
            }
        ],
        "social_structures": [],
        "event_log": {
            "events": [
                _stream_event(
                    "evt_release_001",
                    "mission_release",
                    "mission_dense_multi_robot_001",
                    0.0,
                ),
                _stream_event(
                    "evt_assign_001",
                    "robot_assignment",
                    "mission_dense_multi_robot_001",
                    0.1,
                ),
            ]
        },
        "expected_result": {
            "passed": collision_count == 0 and closest_distance_m >= 0.6,
            "metrics": {"fixture_expected_valid": True},
        },
        "metadata": {
            "collision_check": {
                "checked": True,
                "collision_free": collision_count == 0,
                "collision_count": collision_count,
                "min_clearance_m": closest_distance_m - 0.3,
                "closest_pair": {
                    "actor_a": "robot_001",
                    "actor_b": "robot_002",
                    "distance_m": closest_distance_m,
                    "required_clearance_m": 0.3,
                    "clearance_m": closest_distance_m - 0.3,
                    "t": 1.0,
                },
            },
            "dense_multi_robot": {
                "robot_adjustments": {
                    "robot_001": {
                        "wait_count": 1,
                        "inserted_waits": 1,
                        "glide_count": 2,
                        "inserted_turns": 2,
                        "teleport_count": 0,
                        "yield_blockers": {},
                    },
                    "robot_002": {
                        "wait_count": 0,
                        "inserted_waits": 0,
                        "glide_count": 2,
                        "inserted_turns": 2,
                        "teleport_count": 0,
                        "yield_blockers": {},
                    },
                },
                "corner_case_recovery": {
                    "summary": {
                        "event_count": 1,
                        "by_issue_type": {"robot_robot_deadlock": 1},
                    }
                },
            },
        },
    }
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    return scenario_path


def _write_dense_dynamic_combined_fixture(
    root: Path,
    scenario_id: str,
    *,
    nominal_conflict_count: int,
) -> Path:
    (root / "test_scenes/social_scene").mkdir(parents=True, exist_ok=True)
    scenario_dir = root / "scenarios"
    scenario_dir.mkdir(exist_ok=True)
    scenario_path = scenario_dir / f"{scenario_id}.json"
    scenario = {
        "schema_version": "0.1",
        "scenario_id": scenario_id,
        "scene_id": "social_scene",
        "scene_assets": {
            "dataset": "fixture_dataset",
            "scene_dir": "test_scenes/social_scene",
        },
        "robots": [
            {
                "robot_id": "robot_001",
                "capabilities": ["navigate"],
                "start_map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                "trajectory": [
                    {"t": 0.0, "map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 2.0, "map_pose": {"x": 2.0, "y": 0.0, "yaw": 0.0}},
                ],
            },
            {
                "robot_id": "robot_002",
                "capabilities": ["navigate"],
                "start_map_pose": {"x": 0.0, "y": 2.0, "yaw": 0.0},
                "trajectory": [
                    {"t": 0.0, "map_pose": {"x": 0.0, "y": 2.0, "yaw": 0.0}},
                    {"t": 2.0, "map_pose": {"x": 2.0, "y": 2.0, "yaw": 0.0}},
                ],
            },
        ],
        "humans": [
            _human(
                "human_dense_01",
                "moving_pedestrian",
                [
                    {"t": 0.0, "map_pose": {"x": 10.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 3.0, "map_pose": {"x": 12.0, "y": 0.0, "yaw": 0.0}},
                ],
            )
        ],
        "missions": [
            {
                "mission_id": "mission_dense_dynamic_combined_001",
                "mission_type": "dense_dynamic_combined",
                "assigned_robot_id": "robot_001",
                "release_time": 0.0,
                "deadline": 3.0,
                "priority": 1,
                "success_conditions": [
                    "all_active_robot_goal_regions_reached",
                    "robots_keep_moving_when_passable",
                    "wait_only_when_immediately_blocked",
                    "dense_robot_human_clearance_policy_respected",
                    "humans_keep_moving_until_robot_completion",
                    "human_human_collision_free",
                    "no_robot_robot_collision",
                ],
                "metadata": {
                    "active_robot_ids": ["robot_001", "robot_002"],
                    "active_robot_count": 2,
                    "planned_goal_world_by_robot": {
                        "robot_001": [2.0, 0.0],
                        "robot_002": [2.0, 2.0],
                    },
                    "goal_tolerance_m": 0.1,
                    "minimum_robot_robot_distance_m": 0.6,
                    "minimum_moving_robot_human_distance_m": 0.4,
                    "minimum_stopped_robot_human_distance_m": 0.25,
                    "minimum_human_human_distance_m": 0.05,
                    "mission_end_time_s": 3.0,
                    "expected_robot_motion": {"requires_stops": True},
                },
            }
        ],
        "social_structures": [],
        "event_log": {
            "events": [
                _stream_event(
                    "evt_release_001",
                    "mission_release",
                    "mission_dense_dynamic_combined_001",
                    0.0,
                ),
                _stream_event(
                    "evt_assign_001",
                    "robot_assignment",
                    "mission_dense_dynamic_combined_001",
                    0.1,
                ),
            ]
        },
        "expected_result": {
            "passed": nominal_conflict_count == 0,
            "metrics": {"fixture_expected_valid": True},
        },
        "metadata": {
            "collision_check": {
                "checked": True,
                "collision_free": True,
                "collision_count": 0,
                "min_clearance_m": 0.75,
                "closest_pair": {
                    "actor_a": "robot_001",
                    "actor_b": "human_dense_01",
                    "distance_m": 1.0,
                    "required_clearance_m": 0.25,
                    "clearance_m": 0.75,
                    "t": 1.0,
                },
            },
            "dense_dynamic_combined": {
                "nominal_robot_human_conflict_samples": {
                    "robot_001": nominal_conflict_count,
                    "robot_002": 0,
                },
                "robot_adjustments": {
                    "robot_001": {
                        "wait_count": 1,
                        "inserted_waits": 1,
                        "glide_count": 2,
                        "inserted_turns": 2,
                        "teleport_count": 0,
                        "yield_blockers": {"human_dense_01": 1},
                    },
                    "robot_002": {
                        "wait_count": 0,
                        "inserted_waits": 0,
                        "glide_count": 2,
                        "inserted_turns": 2,
                        "teleport_count": 0,
                        "yield_blockers": {},
                    },
                },
                "human_adjustments": {},
                "corner_case_recovery": {
                    "summary": {
                        "event_count": 1,
                        "by_issue_type": {"robot_robot_deadlock": 1},
                    }
                },
            },
        },
    }
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    return scenario_path


def _serve_queue_mission(
    mission_id: str,
    target_human_id: str,
    *,
    release_time: float,
    deadline: float,
    queue_index: int,
    previous_human_ids: list[str],
    goal_xy: list[float],
) -> dict:
    return {
        "mission_id": mission_id,
        "mission_type": "serve_queue",
        "assigned_robot_id": "robot_alpha",
        "release_time": release_time,
        "deadline": deadline,
        "priority": queue_index + 1,
        "target_human_id": target_human_id,
        "success_conditions": [
            "correct_human_reached",
            "queue_order_preserved",
            "previous_queue_missions_completed",
            "nearest_queue_contact_reached",
        ],
        "metadata": {
            "contact_distance_m": 0.05,
            "planned_goal_world": goal_xy,
            "previous_queue_human_ids": previous_human_ids,
            "queue_id": "queue_001",
            "queue_index": queue_index,
            "queue_order": ["human_a", "human_b"],
            "queue_position": queue_index + 1,
        },
    }


def _human(human_id: str, role: str, trajectory: list[dict]) -> dict:
    return {
        "human_id": human_id,
        "role": role,
        "social_defaults": {
            "personal_space_radius": 0.8,
            "metadata": {
                "collision_geometry": {
                    "radius_m": 0.4,
                    "shape": "cylinder",
                }
            },
        },
        "start_map_pose": trajectory[0]["map_pose"],
        "trajectory": trajectory,
    }


class _AgentConfig:
    START_POSITION = []
    START_ROTATION = []
    IS_SET_START_STATE = False

    def defrost(self) -> None:
        return None

    def freeze(self) -> None:
        return None


class _SimConfig:
    def __init__(self) -> None:
        self.SCENE = ""
        self.REF_JSON = ""
        self.AGENTS = ["AGENT_0"]
        self.DEFAULT_AGENT_ID = 0
        self.AGENT_0 = _AgentConfig()

    def defrost(self) -> None:
        return None

    def freeze(self) -> None:
        return None


if __name__ == "__main__":
    unittest.main()

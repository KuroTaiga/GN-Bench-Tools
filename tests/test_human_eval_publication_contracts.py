from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from GN_Bench.human_eval.baseline_runner import DEFAULT_POLICY_NAMES, POLICY_CLASSES
from GN_Bench.human_eval.dagger import export_iteration_zero
from GN_Bench.human_eval.evaluator import HumanCentricEvaluator
from GN_Bench.human_eval.results import CANONICAL_METRIC_KEYS, replay_result_row
from GN_Bench.human_eval.scenario_adapter import NavDPScenarioAdapter


class HumanEvalPublicationContractsTest(unittest.TestCase):
    def test_replay_exposes_canonical_publication_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path, _ = _write_fixture(root)
            episode = NavDPScenarioAdapter(navdp_root=root).load_split(manifest_path)[0]

            replay = HumanCentricEvaluator().replay(episode)
            row = replay_result_row(episode, replay)

            for metric_key in CANONICAL_METRIC_KEYS:
                self.assertIn(metric_key, replay.metrics)
                self.assertIn(metric_key, row)
            self.assertEqual(replay.metrics["success_rate"], 1.0)
            self.assertEqual(replay.metrics["navigation_error_m"], 0.0)
            self.assertEqual(replay.metrics["collision_rate"], 0.0)
            self.assertIn("human_present", replay.metrics["publication_variants"])

    def test_default_baselines_match_publication_contract(self) -> None:
        self.assertEqual(
            DEFAULT_POLICY_NAMES,
            (
                "oracle_route_follower",
                "shortest_path_no_human",
                "human_aware_greedy",
                "priority_deadline_greedy",
                "single_robot_serial",
            ),
        )
        self.assertTrue(set(DEFAULT_POLICY_NAMES) <= set(POLICY_CLASSES))
        self.assertIn("oracle_human_centric", POLICY_CLASSES)

    def test_dagger_iteration_zero_writes_required_files_and_packet_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest_path, _ = _write_fixture(root, split_shape=True)
            model_input_root = root / "model_inputs"
            hidden_gt_root = root / "hidden_gt"
            model_input_root.mkdir()
            hidden_gt_root.mkdir()
            _write_jsonl(
                model_input_root / "val_seen.jsonl",
                [
                    {
                        "observation_packet_id": "packet-val-seen-001",
                        "split": "val_seen",
                        "scenario_id": "fixture_publication",
                        "mission_id": "mission_deliver_001",
                        "instruction": "Deliver to the waiting person.",
                    }
                ],
            )
            _write_jsonl(
                hidden_gt_root / "val_seen.jsonl",
                [
                    {
                        "hidden_gt_packet_id": "hidden-val-seen-001",
                        "split": "val_seen",
                        "scenario_id": "fixture_publication",
                        "mission_id": "mission_deliver_001",
                        "target_human_id": "human_target",
                        "oracle_path": [[0.0, 0.0], [1.0, 0.0]],
                    }
                ],
            )

            episodes_by_split = NavDPScenarioAdapter(navdp_root=root).load_split_by_name(
                manifest_path
            )
            summary = export_iteration_zero(
                episodes_by_split,
                output_root=root / "out",
                model_input_root=model_input_root,
                hidden_gt_root=hidden_gt_root,
                episodes_per_split=1,
            )

            iteration_dir = root / "out" / "dagger" / "iteration_000"
            self.assertTrue((iteration_dir / "rollouts.jsonl").exists())
            self.assertTrue((iteration_dir / "teacher_actions.jsonl").exists())
            self.assertTrue((iteration_dir / "metrics.json").exists())
            self.assertTrue((iteration_dir / "failures.jsonl").exists())
            rollout = json.loads((iteration_dir / "rollouts.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(rollout["observation_packet_id"], "packet-val-seen-001")
            self.assertEqual(rollout["teacher_action"]["payload"]["target_human_id"], "human_target")
            self.assertEqual(summary["rollout_record_count"], 1)
            self.assertEqual(summary["missing_model_input_packet_count"], 0)
            self.assertEqual(summary["missing_hidden_gt_packet_count"], 0)


def _write_fixture(root: Path, *, split_shape: bool = False) -> tuple[Path, Path]:
    (root / "test_scenes" / "scene_001").mkdir(parents=True)
    scenario_dir = root / "scenarios"
    scenario_dir.mkdir()
    scenario_path = scenario_dir / "fixture_publication.json"
    scenario = {
        "schema_version": "0.1",
        "scenario_id": "fixture_publication",
        "scene_id": "scene_001",
        "scene_assets": {
            "dataset": "fixture_dataset",
            "scene_dir": "test_scenes/scene_001",
        },
        "robots": [
            {
                "robot_id": "robot_alpha",
                "capabilities": ["navigate", "carry"],
                "start_map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0},
                "trajectory": [
                    {"t": 0.0, "map_pose": {"x": 0.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 2.0, "map_pose": {"x": 1.0, "y": 0.0, "yaw": 0.0}},
                ],
            }
        ],
        "humans": [
            {
                "human_id": "human_target",
                "start_map_pose": {"x": 1.0, "y": 0.0, "yaw": 0.0},
                "trajectory": [
                    {"t": 0.0, "map_pose": {"x": 1.0, "y": 0.0, "yaw": 0.0}},
                    {"t": 2.0, "map_pose": {"x": 1.0, "y": 0.0, "yaw": 0.0}},
                ],
            }
        ],
        "missions": [
            {
                "mission_id": "mission_deliver_001",
                "mission_type": "deliver_to_human",
                "assigned_robot_id": "robot_alpha",
                "release_time": 0.0,
                "deadline": 5.0,
                "priority": 1,
                "target_human_id": "human_target",
                "success_conditions": ["correct_human_reached"],
                "metadata": {"planned_goal_world": [1.0, 0.0]},
            }
        ],
        "event_log": {
            "events": [
                {
                    "event_id": "release",
                    "event_type": "mission_release",
                    "mission_id": "mission_deliver_001",
                    "t": 0.0,
                },
                {
                    "event_id": "assign",
                    "event_type": "robot_assignment",
                    "mission_id": "mission_deliver_001",
                    "actor_id": "robot_alpha",
                    "t": 0.0,
                },
                {
                    "event_id": "complete",
                    "event_type": "completion",
                    "mission_id": "mission_deliver_001",
                    "actor_id": "robot_alpha",
                    "t": 2.0,
                },
            ]
        },
        "expected_result": {"passed": True},
        "metadata": {
            "collision_check": {
                "checked": True,
                "collision_free": True,
                "collision_count": 0,
                "min_clearance_m": 1.0,
            }
        },
    }
    scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
    manifest_path = root / "split_manifest.json"
    manifest = (
        {"splits": {"val_seen": [{"path": "scenarios/fixture_publication.json"}]}}
        if split_shape
        else {"examples": [{"path": "scenarios/fixture_publication.json"}]}
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, scenario_path


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    unittest.main()

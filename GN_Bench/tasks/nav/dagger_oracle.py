from __future__ import annotations

import json
import math
import re

import numpy as np
from scipy.spatial.transform import Rotation as R

from GN_Bench.sims.GN_Bench_simulator.actions import GNBenchSimActions
from GN_Bench.tasks.nav.mpc_tools import (
    Polyline2D,
    plan_actions_beam_mpc,
    smooth_polyline,
    wrap_pi,
)

ACTION_STOP = int(GNBenchSimActions.STOP)
ACTION_MOVE_FORWARD = int(GNBenchSimActions.MOVE_FORWARD)
ACTION_TURN_LEFT = int(GNBenchSimActions.TURN_LEFT)
ACTION_TURN_RIGHT = int(GNBenchSimActions.TURN_RIGHT)
NUM_ACTIONS = 6


class DaggerNavigationOracleMixin:
    """DAgger validation and oracle correction helpers for GN-Bench navigation agents.

    The host agent is expected to provide `prompt_type`, `save_image`, `sample_path`,
    `episode_id`, `rgb_list`, and `position_dump_counts`. Keeping this as a mixin
    avoids making GN-Bench depend on a concrete model implementation.
    """

    def _rollout_actions_valid(self, sim, actions):
        action_img = sim.get_occ_map_with_actions(actions, traj_color=(255, 165, 0))
        self.save_image(action_img, "action")

        if not actions or len(actions) != NUM_ACTIONS:
            return False, "invalid_actions_len"

        grid = self._get_collision_grid(sim)
        if grid is None:
            return False, "missing_passable_grid"

        step_size = float(sim.config.FORWARD_STEP_SIZE)
        turn_rad = np.deg2rad(float(sim.config.TURN_ANGLE))
        map_size = (int(sim.map_width), int(sim.map_height))

        rot_left = R.from_euler("z", turn_rad, degrees=False)
        rot_right = R.from_euler("z", -turn_rad, degrees=False)
        local_fwd_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        agent_state = sim.get_agent_state()
        curr_pos = np.array(agent_state.position, dtype=np.float32)
        curr_rot = R.from_quat(agent_state.rotation)

        for step_idx, action in enumerate(map(int, actions)):
            if action == ACTION_STOP:
                continue

            elif action == ACTION_TURN_LEFT:
                curr_rot = curr_rot * rot_left

            elif action == ACTION_TURN_RIGHT:
                curr_rot = curr_rot * rot_right

            elif action == ACTION_MOVE_FORWARD:
                world_fwd = curr_rot.apply(local_fwd_vec)
                next_pos = curr_pos + (world_fwd * step_size)

                px_start = sim.transform_from_world_to_pixel(curr_pos)
                px_end = sim.transform_from_world_to_pixel(next_pos)

                if not self._segment_is_clear(grid, *map_size, px_start, px_end, sim):
                    return False, f"action_rollout_hit_wall_step_{step_idx}"

                curr_pos = next_pos

            else:
                return False, f"unknown_action_{action}"

        return True, "ok"

    def _get_collision_grid(self, sim):
        margins = getattr(sim, "margins", [])
        safe_grids = getattr(sim, "safe_passable_grids", {})

        grid = None
        if margins and safe_grids:
            grid = safe_grids.get(margins[-1])

        if grid is None:
            grid = getattr(sim, "passable_grid", None)

        return grid

    def _is_agent_in_obstacle(self, sim) -> bool:
        margins = getattr(sim, "margins", [])
        safe_grids = getattr(sim, "safe_passable_grids", {})
        if not margins or not safe_grids:
            return False

        grid = safe_grids.get(margins[-1])
        if grid is None:
            return False

        try:
            agent_state = sim.get_agent_state()
            curr_pos = np.array(agent_state.position, dtype=np.float32)
            px, py = sim.transform_from_world_to_pixel(curr_pos)

            if not (0 <= px < int(sim.map_width) and 0 <= py < int(sim.map_height)):
                return False

            return not bool(grid[py][px])
        except Exception:
            return False

    def _segment_is_clear(self, grid, map_w, map_h, start_px, end_px, sim):
        for x, y in sim._bresenham_line(start_px, end_px):
            if not (0 <= x < map_w and 0 <= y < map_h):
                return False
            if not grid[y][x]:
                return False
        return True

    @staticmethod
    def _parse_rollout_hit_step(reason):
        m = re.match(r"^action_rollout_hit_wall_step_(\d+)$", str(reason))
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    @staticmethod
    def _parse_pixel_hit_step(reason):
        m = re.match(r"^pixel_branch_hit_wall_seg_(\d+)$", str(reason))
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _goal_zone_analysis_on_actions(self, sim, actions, goal_position):
        """
        Analyzes if the agent enters and then leaves the goal zone.
        Returns: (is_error, message, info_dict)
        """
        if goal_position is None or actions is None or len(actions) != NUM_ACTIONS:
            return False, "skip", {"first_enter_idx": None}

        GOAL_RADIUS = max(min(3, sim.path_length * 0.2), 1)
        step_size = float(sim.config.FORWARD_STEP_SIZE)
        turn_rad = np.deg2rad(float(sim.config.TURN_ANGLE))

        rot_left = R.from_euler("z", turn_rad, degrees=False)
        rot_right = R.from_euler("z", -turn_rad, degrees=False)
        local_fwd_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        agent_state = sim.get_agent_state()
        curr_pos = np.array(agent_state.position, dtype=np.float32)
        curr_rot = R.from_quat(agent_state.rotation)

        goal_xy = np.array(goal_position[:2], dtype=np.float32)

        curr_dist = float(np.linalg.norm(curr_pos[:2] - goal_xy))
        is_inside = curr_dist <= GOAL_RADIUS

        first_enter_idx = -1 if is_inside else None

        for step_idx, action in enumerate(map(int, actions)):
            if action == ACTION_TURN_LEFT:
                curr_rot = curr_rot * rot_left
            elif action == ACTION_TURN_RIGHT:
                curr_rot = curr_rot * rot_right
            elif action == ACTION_MOVE_FORWARD:
                world_fwd = curr_rot.apply(local_fwd_vec)
                curr_pos = curr_pos + (world_fwd * step_size)

            curr_dist = float(np.linalg.norm(curr_pos[:2] - goal_xy))
            is_now_inside = curr_dist <= GOAL_RADIUS

            #  Outside -> Inside
            if not is_inside and is_now_inside:
                is_inside = True
                first_enter_idx = step_idx

            #  Inside -> Outside
            elif is_inside and not is_now_inside:
                return (
                    True,
                    f"goal_zone_exit_by_action_step_{step_idx}",
                    {"first_enter_idx": first_enter_idx},
                )

        return False, "ok", {"first_enter_idx": first_enter_idx}

    def _pixel_path_valid(self, sim, pixel_waypoints_norm):
        if not pixel_waypoints_norm:
            return False, "missing_pixel_waypoints"

        grid = self._get_collision_grid(sim)
        if grid is None:
            return False, "missing_passable_grid"

        if hasattr(grid, "shape"):
            map_h, map_w = grid.shape[:2]
        else:
            map_h = len(grid)
            map_w = len(grid[0]) if map_h > 0 else 0

        if map_w <= 0 or map_h <= 0:
            return False, "bad_occ_shape"

        waypoints_px = []
        for pt in pixel_waypoints_norm:
            if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                return False, "bad_pixel_waypoint_format"
            raw_x, raw_y = int(pt[0]), int(pt[1])
            px_pt = self._denorm_to_occ_pixel(raw_x, raw_y, map_w, map_h)
            waypoints_px.append(px_pt)

        if waypoints_px:
            pixel_img = sim.get_occ_map_with_pixels(
                waypoints_px, path_color=(255, 0, 255)
            )
            self.save_image(pixel_img, "pixel")

        for idx, (p_start, p_end) in enumerate(zip(waypoints_px, waypoints_px[1:])):
            is_clear = self._segment_is_clear(grid, map_w, map_h, p_start, p_end, sim)

            if not is_clear:
                return False, f"pixel_branch_hit_wall_seg_{idx}"

        return True, "ok"

    def _denorm_to_occ_pixel(self, x_norm, y_norm, occ_w, occ_h):
        px = int(round((int(x_norm) / 1000.0) * float(occ_w)))
        py = int(round((int(y_norm) / 1000.0) * float(occ_h)))
        px = max(0, min(px, occ_w - 1))
        py = max(0, min(py, occ_h - 1))
        return px, py

    def _goal_zone_analysis_on_pixels(self, sim, pixel_waypoints_norm, goal_position):
        """
        Analyzes if the pixel path enters and then leaves the goal zone.
        Pipeline: Norm Coord -> Occ Pixel -> World Coord -> Distance Check.
        """
        if (
            goal_position is None
            or not pixel_waypoints_norm
            or self.prompt_type not in {"V1", "V2"}
        ):
            return False, "skip", {"first_enter_idx": None}

        GOAL_RADIUS = max(min(3, sim.path_length * 0.2), 1)
        map_w, map_h = int(sim.map_width), int(sim.map_height)

        goal_xy = np.array(goal_position[:2], dtype=np.float32)

        is_inside = False
        first_enter_idx = None

        for idx, pt_norm in enumerate(pixel_waypoints_norm):
            if not (isinstance(pt_norm, (list, tuple)) and len(pt_norm) == 2):
                return (
                    True,
                    "goal_zone_exit_bad_pixel_format",
                    {"first_enter_idx": first_enter_idx},
                )

            # Step A: Norm -> Occ Pixel
            norm_x, norm_y = int(pt_norm[0]), int(pt_norm[1])
            px_pt = self._denorm_to_occ_pixel(norm_x, norm_y, map_w, map_h)

            # Step B: Occ Pixel -> World
            world_pt_tuple = sim.transform_from_pixel_to_world(px_pt)
            world_pos = np.array(world_pt_tuple[:2], dtype=np.float32)

            dist = float(np.linalg.norm(world_pos - goal_xy))
            is_now_inside = dist <= GOAL_RADIUS

            if not is_inside and is_now_inside:
                is_inside = True
                first_enter_idx = idx

            elif is_inside and not is_now_inside:
                return (
                    True,
                    f"goal_zone_exit_by_pixel_step_{idx}",
                    {"first_enter_idx": first_enter_idx},
                )

        return False, "ok", {"first_enter_idx": first_enter_idx}

    def _log_inference_step(
        self,
        instruction,
        prompt_text,
        image_paths,
        cur_x,
        cur_y,
        occ_h,
        occ_w,
        goal_position,
        agent_pose,
        raw_text,
        actions,
        pixels,
        token_probs,
        valid,
        act_valid,
        act_reason,
        action_hit_step_idx,
        stop_in_actions,
        curr_dtg,
        exit_by_action,
        exit_action_reason,
        pix_valid,
        pix_reason,
        pixel_hit_step_idx,
        exit_by_pixel,
        exit_pixel_reason,
        valid_reasons,
        gt_true_pair,
        gt_build_reason,
    ):
        """Constructs and writes the comprehensive sample log."""
        payload = {
            "episode_id": self.episode_id,
            "llm_call_idx": self.llm_call_count,
            "step_idx": len(self.rgb_list) - 1,
            "prompt_type": self.prompt_type,
            "in_obstacle": False,
            "input": {
                "instruction": instruction,
                "prompt_text": prompt_text,
                "image_paths": image_paths,
                "cur_pixel": [cur_x, cur_y],
                "occ_shape": [occ_h, occ_w],
                "goal_position": goal_position,
                "agent_pose": agent_pose,
            },
            "output": {
                "raw_text": raw_text,
                "actions": actions,
                "pixels": pixels,
                "token_probs": token_probs,
            },
            "validation": {
                "is_true": bool(valid),
                "action_rollout_valid": bool(act_valid),
                "action_reason": act_reason,
                "action_hit_step_idx": action_hit_step_idx,
                "stop_in_actions": bool(stop_in_actions),
                "distance_to_goal": curr_dtg,
                "goal_zone_exit_action": bool(exit_by_action),
                "goal_zone_exit_action_reason": exit_action_reason,
                "pixel_path_valid": pix_valid,
                "pixel_reason": pix_reason,
                "pixel_hit_step_idx": pixel_hit_step_idx,
                "goal_zone_exit_pixel": bool(exit_by_pixel),
                "goal_zone_exit_pixel_reason": exit_pixel_reason,
                "reasons": valid_reasons,
            },
            "gt_true_pair": gt_true_pair,
            "gt_build_reason": gt_build_reason,
        }
        self._append_sample_log(payload)

    def _log_skip_supervision_step(
        self,
        instruction,
        prompt_text,
        image_paths,
        cur_x,
        cur_y,
        occ_h,
        occ_w,
        goal_position,
        agent_pose,
        raw_text,
        actions,
        pixels,
        token_probs,
    ):
        """Writes a lightweight dump when agent is inside obstacle in non-collidable mode."""
        payload = {
            "episode_id": self.episode_id,
            "llm_call_idx": self.llm_call_count,
            "step_idx": len(self.rgb_list) - 1,
            "prompt_type": self.prompt_type,
            "in_obstacle": True,
            "input": {
                "instruction": instruction,
                "prompt_text": prompt_text,
                "image_paths": image_paths,
                "cur_pixel": [cur_x, cur_y],
                "occ_shape": [occ_h, occ_w],
                "goal_position": goal_position,
                "agent_pose": agent_pose,
            },
            "output": {
                "raw_text": raw_text,
                "actions": actions,
                "pixels": pixels,
                "token_probs": token_probs,
            },
        }
        self._append_sample_log(payload)

    def _append_sample_log(self, payload: dict):
        if not self.sample_path:
            return

        # Limit repeated dumps at the same position within one episode.
        cur_pixel = payload.get("input", {}).get("cur_pixel")
        if isinstance(cur_pixel, (list, tuple)) and len(cur_pixel) == 2:
            try:
                pos_key = (int(cur_pixel[0]), int(cur_pixel[1]))
                seen = self.position_dump_counts.get(pos_key, 0)
                # if seen >= 2:
                #     return
                self.position_dump_counts[pos_key] = seen + 1
            except Exception:
                pass

        try:
            with open(self.sample_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Failed to write sample log: {e}")

    def _handle_goal_zone_exit(self, sim, actions, pixels, context):
        """
        Correction Strategy A: The agent walked THROUGH the goal.
        Fix: Truncate actions/pixels to stop exactly when it entered the zone.
        """
        # 1. Truncate Actions
        enter_idx_act = context["action_goal_info"].get("first_enter_idx")
        gt_actions = self._truncate_and_pad_actions(actions, enter_idx_act)
        gt_output = {"vlnce": self._action_xml_from_ids(gt_actions)}

        # 2. Truncate Pixels (Only for V1/V2)
        if self.prompt_type in {"V1", "V2"}:
            enter_idx_pix = context["pixel_goal_info"].get("first_enter_idx")
            pixel_str = self._truncate_and_pad_pixels(pixels, enter_idx_pix)

            if pixel_str:
                gt_output["pixel"] = pixel_str
            else:
                # Fallback: If pixel truncation fails, use current position stop
                stop_gt = self._build_gt_stop_pair(sim)
                if "pixel" in stop_gt:
                    gt_output["pixel"] = stop_gt["pixel"]

        meta = {"is_true": True, "source": "oracle_stop_at_goal"}
        reason = {"mode": "stop_at_goal"}

        return gt_output, meta, reason

    def _handle_standard_correction(self, sim, goal_position, oracle_cache=None):
        """
        Correction Strategy B: The agent hit a wall, is lost, or stopped too early.
        Fix: Generate a new path from current location using A* and MPC.
        """
        # 1. Plan Actions
        gt_actions = None
        act_reason = "cache_miss"
        if oracle_cache is not None:
            gt_actions = oracle_cache.get("gt_actions")
            act_reason = oracle_cache.get("actions_reason", act_reason)

        if gt_actions is None:
            gt_actions, act_reason = self._build_gt_actions6(sim, goal_position)

        if gt_actions is None:
            return None, {}, None

        gt_output = {"vlnce": self._action_xml_from_ids(gt_actions)}
        reason = {"actions": act_reason}

        # 2. Plan Pixels (Only for V1/V2)
        if self.prompt_type in {"V1", "V2"}:
            gt_pixel_str = None
            pix_reason = "cache_miss"
            if oracle_cache is not None:
                gt_pixel_str = oracle_cache.get("gt_pixel_str")
                pix_reason = oracle_cache.get("pixel_reason", pix_reason)

            if gt_pixel_str is None:
                gt_pixel_str, pix_reason = self._build_gt_pixel_tokstr_13(
                    sim, goal_position
                )

            reason["pixel"] = pix_reason
            if gt_pixel_str:
                gt_output["pixel"] = gt_pixel_str

        meta = {"is_true": True, "source": "oracle_astar_fallback"}

        return gt_output, meta, reason

    def _construct_gt_input(self, context):
        """Extracts standard input fields for the GT pair."""
        return {
            "instruction": context["instruction"],
            "prompt_text": context["prompt_text"],
            "image_paths": context["image_paths"],
            "cur_pixel": context["cur_pixel"],
            "occ_shape": context["occ_shape"],
        }

    def _truncate_and_pad_actions(self, actions, enter_idx):
        """Helper: Keeps actions up to entry index, fills rest with STOP."""
        if (
            not actions
            or len(actions) != NUM_ACTIONS
            or enter_idx is None
            or enter_idx < 0
        ):
            return [ACTION_STOP] * NUM_ACTIONS

        keep_count = enter_idx + 1
        pad_count = NUM_ACTIONS - keep_count
        return list(actions[:keep_count]) + [ACTION_STOP] * pad_count

    def _truncate_and_pad_pixels(self, pixels, enter_idx):
        """Helper: Keeps pixels up to entry index, pads last point to length 13."""
        if not pixels or enter_idx is None or not (0 <= enter_idx < len(pixels)):
            return None

        try:
            # 1. Slice valid path
            valid_path = [list(map(int, p)) for p in pixels[: enter_idx + 1]]

            if not valid_path:
                return None

            # 2. Pad to target length (13)
            TARGET_LEN = 13
            last_pt = valid_path[-1]
            while len(valid_path) < TARGET_LEN:
                valid_path.append(last_pt)

            # 3. Ensure exact length
            final_path = valid_path[:TARGET_LEN]

            return self._pixel_tokstr_from_norm_points(final_path)

        except Exception as e:
            print(f"Error truncating pixels: {e}")
            return None

    @staticmethod
    def _action_xml_from_ids(action_ids):
        id2tok = {
            ACTION_STOP: "<STOP>",
            ACTION_MOVE_FORWARD: "<FWD>",
            ACTION_TURN_LEFT: "<LEFT>",
            ACTION_TURN_RIGHT: "<RIGHT>",
        }
        toks = [id2tok.get(int(a), "<STOP>") for a in action_ids]
        return "<action>" + ",".join(toks) + "</action>"

    def _pixel_tokstr_from_norm_points(self, waypoints_norm):
        parts = []
        for x, y in waypoints_norm:
            xi = max(0, min(int(x), 999))
            yi = max(0, min(int(y), 999))
            parts.append(f"[{self._tok(xi)},{self._tok(yi)}]")
        return "[" + ",".join(parts) + "]"

    def _build_gt_pixel_tokstr_13(self, sim, goal_position):
        """
        Generates the ground truth pixel string (1 current + 12 future waypoints).
        Strategy: Plan full path -> Slice 60 steps -> Pad to 60 -> Sample 12 -> Tokenize.
        """
        # 1. Validation
        if goal_position is None or len(goal_position) < 2:
            return None, "missing_goal_position"

        # 2. Path Planning (A*)
        agent_state = sim.get_agent_state()
        pos_xy = np.array(agent_state.position[:2], dtype=np.float32)
        goal_xy = np.array(goal_position[:2], dtype=np.float32)

        try:
            dense_path = self._plan_dense_path_pixels(sim, pos_xy, goal_xy)
        except Exception as e:
            return None, f"gt_astar_failed:{e}"

        if len(dense_path) == 0:
            return None, "empty_dense_path"

        # 3. Windowing & Sampling Configuration
        LOOKAHEAD_STEPS = 60
        NUM_SAMPLES = 12

        current_px = dense_path[0]
        future_window = dense_path[1 : 1 + LOOKAHEAD_STEPS]

        if not future_window:
            future_window = [current_px]

        # 4. Padding (Extension Strategy)

        needed_pad = LOOKAHEAD_STEPS - len(future_window)
        if needed_pad > 0:
            future_window.extend([future_window[-1]] * needed_pad)

        # 5. Downsampling
        indices = np.linspace(0, LOOKAHEAD_STEPS - 1, NUM_SAMPLES)
        sampled_pts = [future_window[int(np.round(i))] for i in indices]

        # 6. Final Assembly (1 + 12 = 13 points)
        final_waypoints = [current_px] + sampled_pts

        astar_img = sim.get_occ_map_with_pixels(
            final_waypoints, path_color=(138, 43, 226)
        )
        self.save_image(astar_img, "astar")

        # 7. Tokenization
        occ_w, occ_h = int(sim.map_width), int(sim.map_height)
        return self._pixel_tokstr_from_pixels(final_waypoints, occ_w, occ_h), "ok"

    def _pixel_tokstr_from_pixels(self, waypoints_px, occ_w, occ_h):
        tokens = []
        for x, y in waypoints_px:
            xn, yn = self._norm_xy_from_pixel(x, y, occ_w, occ_h)
            tokens.append(f"[{self._tok(xn)},{self._tok(yn)}]")
        return f"[{','.join(tokens)}]"

    @staticmethod
    def _tok(n):
        return f"<{int(n)}>"

    def _build_gt_actions6(self, sim, goal_position):
        """
        Orchestrates the Oracle action generation pipeline:
        1. Validation & State Extraction
        2. A* Path Planning (Pixel Space)
        3. Path Smoothing (World Space)
        4. MPC Trajectory Generation
        5. Kinematic Simulation -> Discrete Actions
        """
        # 1. Validation & Pre-checks
        if goal_position is None or len(goal_position) < 2:
            return None, "missing_goal_position"

        grid = self._get_collision_grid(sim)
        if grid is None:
            return None, "missing_passable_grid"

        # 2. State Extraction
        agent_state = sim.get_agent_state()
        pos_xy = np.array(agent_state.position[:2], dtype=np.float32)
        goal_xy = np.array(goal_position[:2], dtype=np.float32)

        # Trivial success case
        if float(np.linalg.norm(goal_xy - pos_xy)) <= 1.0:
            return [ACTION_STOP] * NUM_ACTIONS, "ok_within_1m"

        # Calculate current Yaw
        rot = R.from_quat(agent_state.rotation)
        fwd_vec = rot.apply(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        curr_yaw = math.atan2(fwd_vec[1], fwd_vec[0])

        # 3. Path Planning (A* -> Dense Pixels)
        try:
            dense_path_px = self._plan_dense_path_pixels(sim, pos_xy, goal_xy)
        except Exception as e:
            return None, f"gt_astar_failed:{e}"

        if not dense_path_px or len(dense_path_px) <= 1:
            return [ACTION_STOP] * NUM_ACTIONS, "ok_no_path_or_already_there"

        # 4. Path Processing (Pixel -> World -> Smooth)
        # Convert pixels to world coordinates
        world_pts = [sim.transform_from_pixel_to_world(px)[:2] for px in dense_path_px]
        pts_array = np.array(world_pts, dtype=np.float64)

        if pts_array.shape[0] < 2:
            return [ACTION_STOP] * NUM_ACTIONS, "ok_path_too_short"

        # Smooth the path for the controller
        pts_smoothed = smooth_polyline(pts_array, mode="vel", win=9, kind="tri")
        poly = Polyline2D(pts_smoothed)

        # 5. MPC Control
        mpc_actions = self._run_mpc_solver(
            sim, poly, float(pos_xy[0]), float(pos_xy[1]), float(curr_yaw)
        )

        if not mpc_actions:
            return [ACTION_STOP] * NUM_ACTIONS, "ok_mpc_empty"

        # 6. Kinematic Simulation (MPC -> Gym Actions)
        final_actions = self._simulate_mpc_kinematics(
            sim, mpc_actions, pos_xy, curr_yaw, goal_xy
        )

        # Save MPC rollout visualization whenever oracle actions are built.
        mpc_img = sim.get_occ_map_with_actions(final_actions)
        self.save_image(mpc_img, "mpc")

        return final_actions, "ok_mpc"

    def _save_oracle_debug_images(self, sim, goal_position):
        """Best-effort dump of oracle MPC/A* visualizations for every step."""
        cache = {
            "gt_actions": None,
            "actions_reason": "missing_goal_position",
            "gt_pixel_str": None,
            "pixel_reason": "missing_goal_position",
        }

        if goal_position is None or len(goal_position) < 2:
            return cache

        try:
            gt_actions, actions_reason = self._build_gt_actions6(sim, goal_position)
            cache["gt_actions"] = gt_actions
            cache["actions_reason"] = actions_reason
        except Exception as e:
            cache["actions_reason"] = f"debug_save_failed:{e}"
            print(f"Warning: failed to save MPC debug image: {e}")

        try:
            gt_pixel_str, pixel_reason = self._build_gt_pixel_tokstr_13(
                sim, goal_position
            )
            cache["gt_pixel_str"] = gt_pixel_str
            cache["pixel_reason"] = pixel_reason
        except Exception as e:
            cache["pixel_reason"] = f"debug_save_failed:{e}"
            print(f"Warning: failed to save A* debug image: {e}")

        return cache

    def _run_mpc_solver(self, sim, poly, x0, y0, th0):
        """
        Helper: Encapsulates the massive configuration for the MPC solver.
        """
        return plan_actions_beam_mpc(
            poly=poly,
            x0=x0,
            y0=y0,
            th0=th0,
            lookahead_m=5.0,
            horizon=32,
            beam=300,
            step_m=float(sim.config.FORWARD_STEP_SIZE),
            turn_deg=float(sim.config.TURN_ANGLE),
            goal_stop_m=0.15,
            max_steps=6,
            relocalize=False,
            relocalize_thresh=0.35,
            w_step=1.0,
            w_turn=0.5,
            w_perp=300.0,
            d0=0.03,
            w_head=0.10,
            w_head_tangent=0.10,
            w_switch=10.0,
            w_terminal=120.0,
            w_progress=2.0,
            w_back=20.0,
            w_goal_heur=1.5,
            w_spin=10.0,
            turn_slack=1,
            commit=2,
            stall_steps=20,
            stall_ds_eps=1e-3,
            endgame_dist=0.0,
            endgame_turn_tol_deg=7.5,
            w_stop_good=-80.0,
            w_stop_bad=300.0,
        )

    def _simulate_mpc_kinematics(self, sim, mpc_actions, start_xy, start_yaw, goal_xy):
        """
        Helper: Simulates the execution of MPC actions to generate
        the final discrete action sequence, handling state updates and stop conditions.
        """
        out_actions = []
        x, y, th = float(start_xy[0]), float(start_xy[1]), float(start_yaw)

        step_m = float(sim.config.FORWARD_STEP_SIZE)
        turn_rad = np.deg2rad(float(sim.config.TURN_ANGLE))

        # MPC Action ID -> Gym Action ID Mapping
        # 1: FWD, 2: LEFT, 3: RIGHT
        mpc_to_gym = {
            1: ACTION_MOVE_FORWARD,
            2: ACTION_TURN_LEFT,
            3: ACTION_TURN_RIGHT,
        }

        for raw_a in mpc_actions:
            if len(out_actions) >= NUM_ACTIONS:
                break

            # Check goal condition BEFORE action
            if math.hypot(x - goal_xy[0], y - goal_xy[1]) <= 1.0:
                break

            # Map and Record Action
            gym_action = mpc_to_gym.get(int(raw_a), ACTION_STOP)
            out_actions.append(gym_action)

            # Kinematic Update (Simulate the move)
            if gym_action == ACTION_TURN_LEFT:
                th = wrap_pi(th + turn_rad)
            elif gym_action == ACTION_TURN_RIGHT:
                th = wrap_pi(th - turn_rad)
            elif gym_action == ACTION_MOVE_FORWARD:
                x += step_m * math.cos(th)
                y += step_m * math.sin(th)

            # Check goal condition AFTER action
            if math.hypot(x - goal_xy[0], y - goal_xy[1]) <= 1.0:
                break

        # Pad with STOP if sequence is shorter than NUM_ACTIONS
        if len(out_actions) < NUM_ACTIONS:
            out_actions.extend([ACTION_STOP] * (NUM_ACTIONS - len(out_actions)))

        return out_actions[:NUM_ACTIONS]

    def _plan_dense_path_pixels(self, sim, start_world_xy, goal_world_xy):
        start_world = np.array(
            [start_world_xy[0], start_world_xy[1], 0.0], dtype=np.float32
        )
        goal_world = np.array(
            [goal_world_xy[0], goal_world_xy[1], 0.0], dtype=np.float32
        )
        start_px = sim.transform_from_world_to_pixel(start_world)
        goal_px = sim.transform_from_world_to_pixel(goal_world)
        raw_path = sim._astar_with_fallback_margin(start_px, goal_px)
        smooth_path = sim._smooth_path(raw_path)
        dense_path = sim._densify_path(smooth_path)
        return dense_path

    @staticmethod
    def _round_half_up_float(x):
        if x >= 0:
            return int(np.floor(x + 0.5))
        return -int(np.floor(-x + 0.5))

    def _norm_xy_from_pixel(self, x, y, occ_w, occ_h):
        xx = max(0, min(int(x), int(occ_w) - 1))
        yy = max(0, min(int(y), int(occ_h) - 1))
        xn = self._round_half_up_float((xx / float(occ_w)) * 1000.0)
        yn = self._round_half_up_float((yy / float(occ_h)) * 1000.0)
        xn = max(0, min(xn, 999))
        yn = max(0, min(yn, 999))
        return xn, yn

    @staticmethod
    def _is_pixel_wall_reason(reason):
        return str(reason).startswith("pixel_branch_hit_wall_seg_")

    def _build_gt_stop_pair(self, sim):
        gt_output = {"vlnce": self._action_xml_from_ids([ACTION_STOP] * NUM_ACTIONS)}
        if self.prompt_type in {"V1", "V2"}:
            cur_px = sim.get_current_pixel_position()
            if cur_px is not None:
                occ_w, occ_h = int(sim.map_width), int(sim.map_height)
                pts = [cur_px] * 13
                gt_output["pixel"] = self._pixel_tokstr_from_pixels(pts, occ_w, occ_h)
        return gt_output

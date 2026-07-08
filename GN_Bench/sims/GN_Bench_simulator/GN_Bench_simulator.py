import json
import math
import numpy as np
import torch
import argparse
import zlib
import struct
import heapq
import cv2

from numpy import ndarray
from pathlib import Path
from typing import Any, Dict, List, Union, Tuple, Sequence, Optional
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from GN_Bench.core.registry import registry
from GN_Bench.config import Config
from GN_Bench.scene.gaussian_model import GaussianModel
from GN_Bench.scene.cameras import MiniCam
from GN_Bench.gaussian_renderer import render_or
from GN_Bench.utils.graphics_utils import getProjectionMatrix
from GN_Bench.arguments import PipelineParams
from GN_Bench.core.simulator import (
    Simulator,
    SensorSuite,
    Observations,
    RGBSensor,
    DepthSensor,
    AgentState,
)
from GN_Bench.utils.telesim_actor_utils import (
    ActorOptions,
    ActorRuntime,
    CombinedGaussianModel,
    DEFAULT_ACTOR_ANIMATION_CYCLE_MOD,
    DEFAULT_ACTOR_BUFFER_DISTANCE,
    DEFAULT_ACTOR_FOLLOW_DISTANCE,
    DEFAULT_ACTOR_FOOT_OFFSET,
    DEFAULT_ACTOR_HEIGHT,
    DEFAULT_ACTOR_LOOP,
    DEFAULT_ACTOR_PATTERN,
    DEFAULT_ACTOR_SPEED,
    DEFAULT_VIDEO_FPS,
    FORWARD_SMOOTH_BLEND,
    EPS,
    load_actor_sequence,
    actor_data_to_tensors,
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

DEFAULT_DOOR_BBOX_FILENAMES = (
    "door_bboxes.json",
    "doors.json",
    "bboxes.json",
    "object_bboxes.json",
    "semantic_annotations.json",
)


class PNGFormatError(RuntimeError):
    """Raised when the PNG file does not meet the expected constraints."""


def draw_solid_star(img, center, radius, color):
    cx, cy = center
    outer_radius = radius
    inner_radius = radius * 0.382

    pts = []
    for i in range(10):
        angle = i * math.pi / 5 - math.pi / 2
        r = outer_radius if i % 2 == 0 else inner_radius
        x = cx + r * math.cos(angle)
        y = cy + r * math.sin(angle)
        pts.append([int(x), int(y)])

    pts = np.array([pts], dtype=np.int32)

    cv2.fillPoly(img, pts, color)


def draw_solid_triangle(img, center, size, color):
    cx, cy = center

    pt1 = (cx, cy - size)
    pt2 = (cx - int(size * 0.866), cy + int(size * 0.5))
    pt3 = (cx + int(size * 0.866), cy + int(size * 0.5))

    pts = np.array([[pt1, pt2, pt3]], dtype=np.int32)

    cv2.fillPoly(img, pts, color)


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def load_grayscale_png(path: Path) -> Tuple[int, int, List[List[int]]]:
    if not path.exists():
        raise FileNotFoundError(f"Occupancy map not found at {path}")

    with path.open("rb") as fh:
        signature = fh.read(8)
        if signature != PNG_SIGNATURE:
            raise PNGFormatError("File is not a PNG image")

        width: Optional[int] = None
        height: Optional[int] = None
        bit_depth: Optional[int] = None
        color_type: Optional[int] = None
        compression: Optional[int] = None
        filter_method: Optional[int] = None
        interlace: Optional[int] = None
        idat_chunks: List[bytes] = []

        while True:
            length_bytes = fh.read(4)
            if not length_bytes:
                break
            length = struct.unpack(">I", length_bytes)[0]
            chunk_type = fh.read(4)
            chunk_data = fh.read(length)
            crc = fh.read(4)
            if len(chunk_type) != 4 or len(chunk_data) != length or len(crc) != 4:
                raise PNGFormatError("Incomplete PNG chunk")

            if chunk_type == b"IHDR":
                (
                    width,
                    height,
                    bit_depth,
                    color_type,
                    compression,
                    filter_method,
                    interlace,
                ) = struct.unpack(">IIBBBBB", chunk_data)
            elif chunk_type == b"IDAT":
                idat_chunks.append(chunk_data)
            elif chunk_type == b"IEND":
                break

        if color_type != 0 or bit_depth != 8:
            raise PNGFormatError("Only 8-bit grayscale PNGs are supported")

        raw = zlib.decompress(b"".join(idat_chunks))
        stride = width
        rows: List[List[int]] = []
        idx = 0
        previous = [0] * stride
        for _ in range(height):
            filter_type = raw[idx]
            idx += 1
            row_bytes = list(raw[idx : idx + stride])
            idx += stride
            row = [0] * stride

            if filter_type == 0:  # None
                row = row_bytes
            elif filter_type == 1:  # Sub
                for x in range(stride):
                    left = row[x - 1] if x > 0 else 0
                    row[x] = (row_bytes[x] + left) & 0xFF
            elif filter_type == 2:  # Up
                for x in range(stride):
                    up = previous[x]
                    row[x] = (row_bytes[x] + up) & 0xFF
            elif filter_type == 3:  # Average
                for x in range(stride):
                    left = row[x - 1] if x > 0 else 0
                    up = previous[x]
                    row[x] = (row_bytes[x] + ((left + up) >> 1)) & 0xFF
            elif filter_type == 4:  # Paeth
                for x in range(stride):
                    left = row[x - 1] if x > 0 else 0
                    up = previous[x]
                    up_left = previous[x - 1] if x > 0 else 0
                    row[x] = (row_bytes[x] + _paeth_predictor(left, up, up_left)) & 0xFF

            rows.append(row)
            previous = row

        return width, height, rows


def reconstruct_passable_grid(
    rows: List[List[int]], threshold: int
) -> List[List[bool]]:
    return [[value >= threshold for value in row] for row in rows]


def inflate_obstacles(grid: List[List[bool]], margin: int) -> List[List[bool]]:
    if margin <= 0:
        return [row[:] for row in grid]

    height = len(grid)
    width = len(grid[0]) if height else 0
    inflated = [row[:] for row in grid]
    blocked_cells = [
        (x, y) for y in range(height) for x in range(width) if not grid[y][x]
    ]
    radius_sq = margin * margin

    for bx, by in blocked_cells:
        y0 = max(by - margin, 0)
        y1 = min(by + margin, height - 1)
        for y in range(y0, y1 + 1):
            dy = y - by
            x0 = max(bx - margin, 0)
            x1 = min(bx + margin, width - 1)
            for x in range(x0, x1 + 1):
                dx = x - bx
                if dx * dx + dy * dy <= radius_sq:
                    inflated[y][x] = False
    return inflated


def read_png_size(path: Path) -> Tuple[int, int]:
    if not path.exists():
        return 0, 0  # Fallback
    with path.open("rb") as fh:
        fh.read(8)
        fh.read(4)
        fh.read(4)
        width = int.from_bytes(fh.read(4), "big")
        height = int.from_bytes(fh.read(4), "big")
    return width, height


def load_occupancy_metadata(scene_dir: Path) -> dict:
    occ_json = scene_dir / "occupancy.json"

    with occ_json.open("r", encoding="utf-8") as fh:
        occ = json.load(fh)

    scale = float(occ.get("scale"))

    occ_png = scene_dir / "occupancy.png"
    if occ_png.exists():
        width_px, height_px = read_png_size(occ_png)

    min_x, min_y, min_z = map(float, occ.get("min"))

    gs_left = min_x
    gs_right = gs_left + width_px * scale
    gs_top = occ.get("max", [0, 0, 0])[1]
    gs_bottom = gs_top - height_px * scale

    lower_vals = occ.get("lower")
    upper_vals = occ.get("upper")
    lower = list(map(float, lower_vals))
    upper = list(map(float, upper_vals))

    return {
        "width": int(width_px),
        "height": int(height_px),
        "scale": scale,
        # GS coordinate bounds
        "left": gs_left,
        "right": gs_right,
        "top": gs_top,
        "bottom": gs_bottom,
        "lower_z": float(min_z),
        # pixel coordinate bounds
        "lower": lower,
        "upper": upper,
    }


def derive_affine_transform(points, pixels, meta):
    n = len(points)
    if n < 2:
        return 1.0, 0.0, 1.0, 0.0

    scale, left, top = float(meta["scale"]), float(meta["left"]), float(meta["top"])

    sum_x, sum_y = sum(p[0] for p in points), sum(p[1] for p in points)
    sum_x2, sum_y2 = sum(p[0] ** 2 for p in points), sum(p[1] ** 2 for p in points)

    sum_map_x, sum_map_y, sum_x_map_x, sum_y_map_y = 0.0, 0.0, 0.0, 0.0

    for pt, pix in zip(points, pixels):
        map_x = left + int(pix[0]) * scale
        map_y = top - int(pix[1]) * scale
        sum_map_x += map_x
        sum_map_y += map_y
        sum_x_map_x += pt[0] * map_x
        sum_y_map_y += pt[1] * map_y

    denom_x = n * sum_x2 - sum_x**2
    denom_y = n * sum_y2 - sum_y**2

    if abs(denom_x) < 1e-8 or abs(denom_y) < 1e-8:
        return 1.0, 0.0, 1.0, 0.0

    a_x = (n * sum_x_map_x - sum_x * sum_map_x) / denom_x
    b_x = (sum_map_x - a_x * sum_x) / n
    a_y = (n * sum_y_map_y - sum_y * sum_map_y) / denom_y
    b_y = (sum_map_y - a_y * sum_y) / n
    return a_x, b_x, a_y, b_y


def transform_from_world_to_GS(
    pt_world: np.ndarray, affine: tuple, meta: dict
) -> np.ndarray:
    a_x, b_x, a_y, b_y = affine
    x_aff = a_x * pt_world[0] + b_x
    y_aff = a_y * pt_world[1] + b_y

    center_x = 0.5 * (meta["left"] + meta["right"])
    center_y = 0.5 * (meta["top"] + meta["bottom"])
    x_final = center_x * 2.0 - x_aff
    y_final = center_y * 2.0 - y_aff

    return np.array([x_final, y_final, pt_world[2]], dtype=np.float32)


def transform_from_GS_to_world(
    pt_gs: np.ndarray, affine: tuple, meta: dict
) -> np.ndarray:
    a_x, b_x, a_y, b_y = affine
    x_gs = pt_gs[0]
    y_gs = pt_gs[1]

    center_x = 0.5 * (meta["left"] + meta["right"])
    center_y = 0.5 * (meta["top"] + meta["bottom"])
    x_aff = center_x * 2.0 - x_gs
    y_aff = center_y * 2.0 - y_gs

    if abs(a_x) < 1e-8:
        x_nav = x_aff
    else:
        x_nav = (x_aff - b_x) / a_x

    if abs(a_y) < 1e-8:
        y_nav = y_aff
    else:
        y_nav = (y_aff - b_y) / a_y

    z_nav = pt_gs[2] if pt_gs.shape[0] > 2 else 0.0

    return np.array([x_nav, y_nav, z_nav], dtype=np.float32)


def build_look_at_matrix(eye, target, up):
    z_axis = target - eye
    norm = np.linalg.norm(z_axis)
    if norm < EPS:
        z_axis = np.array([0, 0, 1], dtype=np.float32)
    else:
        z_axis /= norm

    x_axis = np.cross(z_axis, up)
    norm_x = np.linalg.norm(x_axis)
    if norm_x < EPS:
        x_axis = np.array([1, 0, 0], dtype=np.float32)
    else:
        x_axis /= norm_x

    y_axis = np.cross(z_axis, x_axis)

    view = np.eye(4, dtype=np.float32)
    view[0, :3] = x_axis
    view[1, :3] = y_axis
    view[2, :3] = z_axis
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def rotation_matrix_z_np(theta: float) -> np.ndarray:
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return np.array(
        [
            [cos_t, -sin_t, 0.0],
            [sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def build_transform_matrix(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    if rotation.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    if translation.shape != (3,):
        raise ValueError("translation must be length-3 vector")
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
    return np.stack(
        [quat_wxyz[..., 1], quat_wxyz[..., 2], quat_wxyz[..., 3], quat_wxyz[..., 0]],
        axis=-1,
    )


def xyzw_to_wxyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return np.stack(
        [quat_xyzw[..., 3], quat_xyzw[..., 0], quat_xyzw[..., 1], quat_xyzw[..., 2]],
        axis=-1,
    )


def world_to_gs_linear_from_affine(affine: tuple) -> np.ndarray:
    a_x, _, a_y, _ = affine
    return np.array(
        [
            [-float(a_x), 0.0, 0.0],
            [0.0, -float(a_y), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _project_matrices_to_so3(mats: np.ndarray) -> np.ndarray:
    """Project batched 3x3 matrices to closest proper rotations."""
    u, _, vt = np.linalg.svd(mats)
    rot = u @ vt
    det = np.linalg.det(rot)
    neg_mask = det < 0.0
    if np.any(neg_mask):
        u_fix = u.copy()
        u_fix[neg_mask, :, -1] *= -1.0
        rot[neg_mask] = u_fix[neg_mask] @ vt[neg_mask]
    return rot


def transform_points_world_to_gs(
    points_world: np.ndarray, affine: tuple, meta: dict
) -> np.ndarray:
    points_world = np.asarray(points_world, dtype=np.float64)
    a_x, b_x, a_y, b_y = affine
    center_x = 0.5 * (meta["left"] + meta["right"])
    center_y = 0.5 * (meta["top"] + meta["bottom"])
    x_aff = float(a_x) * points_world[:, 0] + float(b_x)
    y_aff = float(a_y) * points_world[:, 1] + float(b_y)
    x_gs = center_x * 2.0 - x_aff
    y_gs = center_y * 2.0 - y_aff
    z_gs = points_world[:, 2]
    return np.stack([x_gs, y_gs, z_gs], axis=1)


def apply_transform_to_actor_frame(
    frame_data: np.ndarray,
    transform: np.ndarray,
    *,
    rot_names: Sequence[str],
    affine: tuple,
    meta: dict,
) -> np.ndarray:
    """Apply a 4x4 transform to one actor frame structured array."""
    transformed = np.array(frame_data, copy=True)

    xyz = np.stack(
        (transformed["x"], transformed["y"], transformed["z"]), axis=1
    ).astype(np.float64)
    a = transform[:3, :3].astype(np.float64)
    trans_vec = transform[:3, 3].astype(np.float64)
    rotation_world = a.T.copy()
    xyz_world = (xyz @ rotation_world.T) + trans_vec
    xyz_gs = transform_points_world_to_gs(xyz_world, affine, meta)
    transformed["x"] = xyz_gs[:, 0].astype(frame_data.dtype["x"])
    transformed["y"] = xyz_gs[:, 1].astype(frame_data.dtype["y"])
    transformed["z"] = xyz_gs[:, 2].astype(frame_data.dtype["z"])

    if len(rot_names) == 4:
        q_local_wxyz = np.stack(
            [frame_data[name] for name in rot_names], axis=1
        ).astype(np.float64)
        local_rot_mats = R.from_quat(wxyz_to_xyzw(q_local_wxyz)).as_matrix()
        world_rot_mats = np.einsum("ij,njk->nik", rotation_world, local_rot_mats)
        world_to_gs_linear = world_to_gs_linear_from_affine(affine)
        gs_rot_raw = world_to_gs_linear[None, :, :] @ world_rot_mats
        gs_rot_mats = _project_matrices_to_so3(gs_rot_raw)
        q_world_gs_xyzw = R.from_matrix(gs_rot_mats).as_quat()
        q_world_gs_wxyz = xyzw_to_wxyz(q_world_gs_xyzw)
        for i, name in enumerate(rot_names):
            transformed[name] = q_world_gs_wxyz[:, i].astype(frame_data.dtype[name])

    return transformed


def _as_float_list(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def _normalize_bbox_2d(raw_bbox: Any) -> Optional[Tuple[float, float, float, float]]:
    """Return an xy AABB as (min_x, min_y, max_x, max_y)."""
    if isinstance(raw_bbox, dict):
        min_vals = (
            raw_bbox.get("min")
            or raw_bbox.get("mins")
            or raw_bbox.get("lower")
            or raw_bbox.get("lower_bound")
        )
        max_vals = (
            raw_bbox.get("max")
            or raw_bbox.get("maxs")
            or raw_bbox.get("upper")
            or raw_bbox.get("upper_bound")
        )
        if min_vals is not None and max_vals is not None:
            lower = _as_float_list(min_vals)
            upper = _as_float_list(max_vals)
            if lower is not None and upper is not None and len(lower) >= 2 and len(upper) >= 2:
                x0, y0 = lower[0], lower[1]
                x1, y1 = upper[0], upper[1]
                return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

        center_vals = raw_bbox.get("center") or raw_bbox.get("centroid")
        size_vals = raw_bbox.get("size") or raw_bbox.get("extent") or raw_bbox.get("scale")
        center = _as_float_list(center_vals)
        size = _as_float_list(size_vals)
        if center is not None and size is not None and len(center) >= 2 and len(size) >= 2:
            half_x = abs(size[0]) * 0.5
            half_y = abs(size[1]) * 0.5
            return center[0] - half_x, center[1] - half_y, center[0] + half_x, center[1] + half_y

        if {"min_x", "min_y", "max_x", "max_y"}.issubset(raw_bbox):
            x0 = float(raw_bbox["min_x"])
            y0 = float(raw_bbox["min_y"])
            x1 = float(raw_bbox["max_x"])
            y1 = float(raw_bbox["max_y"])
            return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

        if {"x_min", "y_min", "x_max", "y_max"}.issubset(raw_bbox):
            x0 = float(raw_bbox["x_min"])
            y0 = float(raw_bbox["y_min"])
            x1 = float(raw_bbox["x_max"])
            y1 = float(raw_bbox["y_max"])
            return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

        return None

    vals = _as_float_list(raw_bbox)
    if vals is None:
        return None
    if len(vals) == 4:
        x0, y0, x1, y1 = vals
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
    if len(vals) >= 6:
        x0, y0, x1, y1 = vals[0], vals[1], vals[3], vals[4]
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
    return None


def _looks_like_door(entry: Dict[str, Any]) -> bool:
    for key in ("category", "class", "class_name", "label", "name", "object_type", "type"):
        value = entry.get(key)
        if isinstance(value, str) and "door" in value.lower():
            return True
    return False


def _extract_door_bboxes(
    payload: Any, *, accept_unlabeled: bool = False
) -> List[Tuple[float, float, float, float]]:
    bboxes: List[Tuple[float, float, float, float]] = []

    def visit(node: Any, parent_is_door: bool = False) -> None:
        if isinstance(node, dict):
            is_door = parent_is_door or _looks_like_door(node)
            bbox_keys = (
                "bbox",
                "bbox_2d",
                "bbox3d",
                "bbox_3d",
                "aabb",
                "bounds",
                "bounding_box",
            )
            for key in bbox_keys:
                if key in node and (is_door or accept_unlabeled):
                    bbox = _normalize_bbox_2d(node[key])
                    if bbox is not None:
                        bboxes.append(bbox)
            if (is_door or accept_unlabeled) and any(
                key in node
                for key in (
                    "min",
                    "mins",
                    "lower",
                    "lower_bound",
                    "max",
                    "maxs",
                    "upper",
                    "upper_bound",
                    "center",
                    "centroid",
                    "size",
                    "extent",
                    "scale",
                    "min_x",
                    "x_min",
                )
            ):
                bbox = _normalize_bbox_2d(node)
                if bbox is not None:
                    bboxes.append(bbox)
            for key, value in node.items():
                visit(value, is_door or "door" in str(key).lower())
        elif isinstance(node, list):
            bbox = _normalize_bbox_2d(node)
            if accept_unlabeled and bbox is not None:
                bboxes.append(bbox)
                return
            for item in node:
                visit(item, parent_is_door)

    visit(payload)

    deduped: List[Tuple[float, float, float, float]] = []
    seen = set()
    for bbox in bboxes:
        key = tuple(round(v, 4) for v in bbox)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(bbox)
    return deduped


# =============================================================================
#  GN_Bench Simulator Implementation
# =============================================================================


@registry.register_sensor
class GNBenchSimRGBSensor(RGBSensor):
    def __init__(self, config: Config) -> None:
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=255,
            shape=(self.config.HEIGHT, self.config.WIDTH, 3),
            dtype=np.uint8,
        )

    def get_observation(self, sim_obs: Dict[str, Any]) -> Any:
        return sim_obs.get(self.uuid, None)


@registry.register_sensor
class GNBenchSimDepthSensor(DepthSensor):
    def __init__(self, config: Config) -> None:
        super().__init__(config=config)

    def _get_observation_space(self, *args: Any, **kwargs: Any) -> spaces.Box:
        normalize_depth = bool(getattr(self.config, "NORMALIZE_DEPTH", True))
        high = 1.0 if normalize_depth else float(getattr(self.config, "MAX_DEPTH", 1.0))
        return spaces.Box(
            low=0.0,
            high=high,
            shape=(self.config.HEIGHT, self.config.WIDTH, 1),
            dtype=np.float32,
        )

    def get_observation(self, sim_obs: Dict[str, Any]) -> Any:
        return sim_obs.get(self.uuid, None)


@registry.register_simulator(name="Sim-v0")
class GNBenchSim(Simulator):
    r"""Pure Python Simulator using 3D Gaussian Splatting."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self._current_scene = None
        self._sensor_suite = None

        # --- 3DGS Components ---
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.gaussians = None
        self.pipeline_args = None
        self.bg_color = torch.tensor([1, 1, 1], dtype=torch.float32, device=self.device)
        self.meta = None
        self.affine = None
        self.occupancy_rows = None
        self.passable_grid = None
        self.safe_passable_grids = {}
        self.map_width = 0
        self.map_height = 0
        self.threshold = 200
        self.margins = [7, 6, 5, 4]

        # BEV and OCC map tracking
        self.bev_path = None  # Path to BEV map file
        self.occ_map_original = None  # Original OCC map (grayscale)
        self.occ_map_with_trajectory = None  # OCC map with trajectory drawn
        self.bev_map_original = None  # Original BEV map (RGB)
        self.bev_map_with_trajectory = None  # BEV map with trajectory drawn
        self.gt_trajectory_pixels = []  # Static GT path in pixel coordinates
        self.start_pixel = None  # Start position in pixel coordinates
        self.trajectory_pixels = []  # List of all visited pixel positions

        self.agent_state = AgentState()
        self.raster_world = None
        self.path_length: float = 0.0
        self._last_passable_position: Optional[np.ndarray] = None
        self.actor_plans: list[np.ndarray] = []
        self.actor_indices: list[int] = []
        self.actor_runtime: ActorRuntime | None = None
        self.actor_combined_model: CombinedGaussianModel | None = None
        self.actor_combined_size: int | None = None
        self.actor_render_cursor: int = 0
        self.scene_rest_dim: int = 0
        self.actor_seq_dir: Optional[Path] = None
        self.actor_enabled: bool = False
        self.door_bboxes: List[Tuple[float, float, float, float]] = []

        self._initialize_sensors()

    def _initialize_sensors(self):
        sim_sensors = []
        if hasattr(self.config, "AGENT_0"):
            sensor_names = self.config.AGENT_0.SENSORS
            for s_name in sensor_names:
                s_cfg = getattr(self.config, s_name)
                if "RGB" in s_name:
                    sim_sensors.append(GNBenchSimRGBSensor(s_cfg))
                elif "DEPTH" in s_name:
                    sim_sensors.append(GNBenchSimDepthSensor(s_cfg))

        self._sensor_suite = SensorSuite(sim_sensors)

    def reconfigure(self, config: Config, actor_seq_dir="") -> None:
        scene_dir = Path(config.SCENE)
        ref_json = Path(config.REF_JSON)
        self.actor_seq_dir = Path(actor_seq_dir) if actor_seq_dir else None
        self.actor_enabled = (
            self.actor_seq_dir is not None and self.actor_seq_dir.exists()
        )

        # Pipeline args (placeholder)
        parser = argparse.ArgumentParser()
        self.pipeline_args = PipelineParams(parser)

        self._current_scene = str(scene_dir)
        print(f"Loading scene: {scene_dir}")

        # 1. Load Metadata & Affine
        self.meta = load_occupancy_metadata(scene_dir)
        points, pixels = self._load_raster_points(ref_json)
        self.path_length = self._compute_path_length(points)
        self.affine = derive_affine_transform(points, pixels, self.meta)
        self.door_bboxes = self._load_door_bboxes(scene_dir)

        # 2. Load Occupancy Map for Pathfinding
        occ_path = scene_dir / "occupancy.png"
        self.map_width, self.map_height, self.occupancy_rows = load_grayscale_png(
            occ_path
        )

        self.passable_grid = reconstruct_passable_grid(
            self.occupancy_rows, self.threshold
        )

        self.safe_passable_grids = {}
        for margin in self.margins:
            self.safe_passable_grids[margin] = inflate_obstacles(
                self.passable_grid, margin
            )

        print(
            f"Pathfinding: Map loaded ({self.map_width}x{self.map_height}), Margins: {self.margins}"
        )
        self.gt_trajectory_pixels = list(pixels)

        # 2.5. Record BEV path
        bev_path = scene_dir / "bev_map.png"
        if bev_path.exists():
            self.bev_path = str(bev_path)
        else:
            self.bev_path = None

        # Load OCC map as RGB for drawing
        if occ_path.exists():
            self.occ_map_original = cv2.imread(str(occ_path))
            self.occ_map_original = cv2.cvtColor(
                self.occ_map_original, cv2.COLOR_BGR2RGB
            )
            self.occ_map_with_trajectory = self.draw_gt_on_map(
                self.occ_map_original.copy(), pixels, path_color=(0, 0, 0)
            )
        else:
            self.occ_map_original = None
            self.occ_map_with_trajectory = None

        if bev_path.exists():
            self.bev_map_original = cv2.imread(str(bev_path))
            self.bev_map_with_trajectory = self.draw_gt_on_map(
                self.bev_map_original.copy(), pixels, path_color=(0, 0, 0)
            )
        else:
            self.bev_map_original = None
            self.bev_map_with_trajectory = None

        # 3. Load Gaussian Model
        ply_path = scene_dir / "3dgs_compressed.ply"
        self.gaussians = GaussianModel(sh_degree=3)
        self.gaussians.load_ply(str(ply_path))
        self.scene_rest_dim = int(self.gaussians.get_features_rest.shape[1])

        # 4. Reset Agent to origin or config start
        self._load_initial_state(ref_json)

        # 5. Load Actor
        self.actor_runtime = None
        self.actor_plans = []
        self.actor_indices = []
        self.actor_combined_model = None
        self.actor_combined_size = None
        self.actor_render_cursor = 0

        if self.actor_enabled:
            actor_options = ActorOptions(
                sequence_dir=self.actor_seq_dir,
                pattern=DEFAULT_ACTOR_PATTERN,
                height=DEFAULT_ACTOR_HEIGHT,
                speed=DEFAULT_ACTOR_SPEED,
                fps=float(DEFAULT_VIDEO_FPS),
                loop=DEFAULT_ACTOR_LOOP,
                foot_offset=DEFAULT_ACTOR_FOOT_OFFSET,
                follow_distance=DEFAULT_ACTOR_FOLLOW_DISTANCE,
                buffer_distance=DEFAULT_ACTOR_BUFFER_DISTANCE,
                animation_cycle_mod=DEFAULT_ACTOR_ANIMATION_CYCLE_MOD,
            )
            actor_sequence = load_actor_sequence(actor_options)
            self.actor_runtime = ActorRuntime(
                options=actor_options, sequence=actor_sequence
            )
            self.build_actor_follow_plans()

    def build_actor_follow_plans(self):
        sampler = PathSampler(self.raster_world)
        distances = list(sampler.cumulative)
        total_length = sampler.total_length

        follow_distance_m = max(float(self.actor_runtime.options.follow_distance), 0.0)
        max_camera_distance = max(total_length - follow_distance_m, 0.0)
        actor_ground_z = float(
            self.meta["lower"][2] + self.actor_runtime.options.foot_offset
        )

        cycle_mod = max(
            1, int(getattr(self.actor_runtime.options, "animation_cycle_mod", 1))
        )
        anim_step = (
            self.actor_runtime.options.fps / float(DEFAULT_VIDEO_FPS)
        ) * cycle_mod
        anim_cursor = 0.0
        num_actor_frames = len(self.actor_runtime.sequence.frames)

        cached_direction = np.array([0.0, 1.0], dtype=np.float32)
        prev_actor_dir: np.ndarray | None = None

        for dist in distances:
            camera_distance = min(dist, max_camera_distance)
            actor_distance = min(camera_distance + follow_distance_m, total_length)

            direction_xy = sampler.direction_at(actor_distance)
            if np.linalg.norm(direction_xy) < EPS:
                direction_xy = cached_direction
            actor_dir = direction_xy.copy()
            if prev_actor_dir is not None:
                blended_actor = (
                    prev_actor_dir * (1.0 - FORWARD_SMOOTH_BLEND)
                    + actor_dir * FORWARD_SMOOTH_BLEND
                )
                norm_actor = np.linalg.norm(blended_actor)
                if norm_actor > EPS:
                    actor_dir = blended_actor / norm_actor
            if np.linalg.norm(direction_xy) >= EPS:
                cached_direction = actor_dir
            prev_actor_dir = actor_dir

            theta = math.atan2(actor_dir[0], actor_dir[1]) + math.pi
            rotation_np = rotation_matrix_z_np(theta)

            actor_pos_xy = sampler.position_at(actor_distance)
            translation_vec = np.array(
                [actor_pos_xy[0], actor_pos_xy[1], actor_ground_z], dtype=np.float64
            )
            transform = build_transform_matrix(rotation_np, translation_vec)

            if self.actor_runtime.options.loop:
                anim_idx = int(anim_cursor) % num_actor_frames
            else:
                anim_idx = min(int(anim_cursor), num_actor_frames - 1)
            anim_cursor += anim_step

            self.actor_plans.append(transform)
            self.actor_indices.append(anim_idx)

            if camera_distance >= max_camera_distance - EPS:
                break

    def _load_initial_state(self, json_path):
        with json_path.open("r") as fh:
            payload = json.load(fh)

        self.agent_state.position = np.array(
            [
                payload["start"]["world"]["x"],
                payload["start"]["world"]["y"],
                1.3,
            ],
            dtype=np.float32,
        )
        self.agent_state.rotation = R.from_euler(
            "z", payload["start_facing"]["world"]["heading_degrees"], degrees=True
        ).as_quat()

        self.raster_world = np.array(
            [[p["x"], p["y"]] for p in payload["path"]["raster_world"]]
        )

    def _load_raster_points(self, json_path):
        """Helper to load points for affine calc (copied logic from main)"""
        with json_path.open("r") as fh:
            payload = json.load(fh)
        path_data = payload.get("path", {})
        rw = path_data.get("raster_world", [])
        rp = path_data.get("raster_pixel", [])
        points = [
            np.array([float(e["x"]), float(e["y"])], dtype=np.float32) for e in rw
        ]
        pixels = [(int(p[0]), int(p[1])) for p in rp]
        return points, pixels

    def _compute_path_length(self, points: Sequence[np.ndarray]) -> float:
        if len(points) < 2:
            return 0.0

        return np.linalg.norm(np.diff(points, axis=0), axis=1).sum()

    # --- Core Simulator Methods ---

    def reset(self) -> Observations:
        # Initialize trajectory tracking
        self._update_last_passable_position(np.array(self.agent_state.position))
        pos = np.array([self.agent_state.position[0], self.agent_state.position[1], 0])
        self.start_pixel = self.transform_from_world_to_pixel(pos)
        self.trajectory_pixels = [self.start_pixel]
        return self._render_observations()

    def step(self, action: Union[str, int, Dict[str, Any]]) -> Observations:
        move_dist = self.config.FORWARD_STEP_SIZE
        turn_angle = math.radians(self.config.TURN_ANGLE)

        prev_pos = np.array(self.agent_state.position)
        prev_rot_obj = R.from_quat(self.agent_state.rotation)

        moved = False
        door_teleported = False
        if isinstance(action, dict) and "position" in action:
            self.agent_state.position = np.array(action["position"], dtype=np.float32)
            if "rotation" in action:
                self.agent_state.rotation = np.array(
                    action["rotation"], dtype=np.float32
                )
            else:
                delta_xy = self.agent_state.position[:2] - prev_pos[:2]
                if float(np.linalg.norm(delta_xy)) > EPS:
                    heading = math.atan2(float(delta_xy[1]), float(delta_xy[0]))
                    self.agent_state.rotation = R.from_euler(
                        "z", heading, degrees=False
                    ).as_quat()
            moved = True

        else:
            action_id = action
            if isinstance(action, dict):
                action_id = action.get("action", 0)

            if action_id == 1:
                local_fwd = np.array([1, 0, 0], dtype=np.float32)
                world_fwd = prev_rot_obj.apply(local_fwd)
                teleport_pos = self._get_door_teleport_position(prev_pos, world_fwd)
                if teleport_pos is not None:
                    new_pos = teleport_pos
                    door_teleported = True
                else:
                    new_pos = prev_pos + world_fwd * move_dist
                self.agent_state.position = new_pos
                moved = True

            if action_id == 2:  # TURN_LEFT
                delta_rot = R.from_euler("z", turn_angle, degrees=False)
                new_rot_obj = prev_rot_obj * delta_rot
                self.agent_state.rotation = new_rot_obj.as_quat()

            elif action_id == 3:  # TURN_RIGHT
                delta_rot = R.from_euler("z", -turn_angle, degrees=False)
                new_rot_obj = prev_rot_obj * delta_rot
                self.agent_state.rotation = new_rot_obj.as_quat()

        # Always compute clipped passable position so geodesic fallback can use it.
        # COLLIDABLE only controls whether agent pose is physically clamped.
        if moved and not door_teleported:
            min_margin = self.margins[-1]
            grid = self.safe_passable_grids.get(min_margin)
            if grid is not None:
                adjusted_pos, collided = self._clip_motion_to_free_space(
                    prev_pos, np.array(self.agent_state.position), grid
                )
                if collided:
                    # Use the clipped free-space point as geodesic fallback.
                    self._update_last_passable_position(adjusted_pos)
                    if self.config.COLLIDABLE:
                        self.agent_state.position = adjusted_pos
                        moved = not np.allclose(adjusted_pos, prev_pos)

        # Update trajectory on occ_map only if agent moved forward
        if (
            moved
            and self.occ_map_with_trajectory is not None
            and self.bev_map_with_trajectory is not None
        ):
            pos = np.array(
                [self.agent_state.position[0], self.agent_state.position[1], 0]
            )
            current_pixel = self.transform_from_world_to_pixel(pos)
            if len(self.trajectory_pixels) > 0:
                self.occ_map_with_trajectory = self.draw_step(
                    self.occ_map_with_trajectory,
                    self.trajectory_pixels[-1],
                    current_pixel,
                    (0, 140, 255),
                )
                self.bev_map_with_trajectory = self.draw_step(
                    self.bev_map_with_trajectory,
                    self.trajectory_pixels[-1],
                    current_pixel,
                    (0, 140, 255),
                )
            self.trajectory_pixels.append(current_pixel)

        self._update_last_passable_position(np.array(self.agent_state.position))

        return self._render_observations()

    def get_sensor_observations(self) -> Observations:
        return self._render_observations()

    def _render_observations(self) -> Observations:
        obs = {}

        pos = self.agent_state.position

        r = R.from_quat(self.agent_state.rotation)
        local_fwd = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        world_fwd = r.apply(local_fwd)
        target = pos + world_fwd

        gs_pos = transform_from_world_to_GS(pos, self.affine, self.meta)
        gs_target = transform_from_world_to_GS(target, self.affine, self.meta)

        render_gaussians = self.gaussians
        if (
            self.actor_enabled
            and self.actor_seq_dir is not None
            and self.actor_runtime is not None
            and len(self.actor_plans) > 0
            and len(self.actor_indices) > 0
        ):
            plan_count = min(len(self.actor_plans), len(self.actor_indices))
            if self.actor_render_cursor > plan_count:
                obs["follow"] = False
            else:
                obs["follow"] = True
            plan_idx = min(self.actor_render_cursor, plan_count - 1)
            actor_transform = self.actor_plans[plan_idx]
            actor_frame_idx = self.actor_indices[plan_idx]
            actor_frame = self.actor_runtime.sequence.frames[actor_frame_idx]
            actor_data = apply_transform_to_actor_frame(
                actor_frame.base_data,
                actor_transform,
                rot_names=self.actor_runtime.sequence.rot_names,
                affine=self.affine,
                meta=self.meta,
            )
            actor_render = actor_data_to_tensors(
                actor_data,
                self.actor_runtime.sequence,
                device=self.gaussians.get_xyz.device,
                target_rest_dim=self.scene_rest_dim,
            )
            current_actor_size = int(actor_render.xyz.shape[0])
            if (
                self.actor_combined_model is None
                or self.actor_combined_size != current_actor_size
            ):
                self.actor_combined_size = current_actor_size
                self.actor_combined_model = CombinedGaussianModel(
                    self.gaussians, actor_render
                )
            else:
                self.actor_combined_model.update_actor(actor_render)
            render_gaussians = self.actor_combined_model
            self.actor_render_cursor += 5

        for sensor_uuid, sensor in self._sensor_suite.sensors.items():
            if isinstance(sensor, (GNBenchSimRGBSensor, GNBenchSimDepthSensor)):
                width = sensor.config.WIDTH
                height = sensor.config.HEIGHT
                fov = sensor.config.HFOV

                fovy = math.radians(fov)
                fovx = 2.0 * math.atan(math.tan(fovy * 0.5) * (width / height))

                up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

                view_mat = build_look_at_matrix(gs_pos, gs_target, up)

                world_view_transform = (
                    torch.from_numpy(view_mat).transpose(0, 1).to(self.device)
                )
                projection_matrix = (
                    getProjectionMatrix(znear=0.01, zfar=100.0, fovX=fovx, fovY=fovy)
                    .transpose(0, 1)
                    .to(self.device)
                )
                full_proj_transform = (
                    world_view_transform.unsqueeze(0) @ projection_matrix.unsqueeze(0)
                ).squeeze(0)

                cam = MiniCam(
                    width,
                    height,
                    fovy,
                    fovx,
                    0.01,
                    100.0,
                    world_view_transform,
                    full_proj_transform,
                )

                with torch.no_grad():
                    out = render_or(
                        cam, render_gaussians, self.pipeline_args, self.bg_color
                    )

                if isinstance(sensor, GNBenchSimRGBSensor):
                    rgb = out["render"].detach().cpu().numpy()
                    rgb = np.transpose(rgb, (1, 2, 0))
                    rgb = np.clip(rgb, 0.0, 1.0) * 255.0
                    obs[sensor_uuid] = rgb.astype(np.uint8)
                else:
                    depth = out["depth"].detach().float().cpu().numpy()
                    if depth.ndim == 3 and depth.shape[0] == 1:
                        depth = depth[0]
                    elif depth.ndim == 3 and depth.shape[-1] == 1:
                        depth = depth[..., 0]

                    # The differentiable rasterizer returns inverse depth.
                    # Convert to metric depth to match VLN-CE depth semantics.
                    depth = np.asarray(depth, dtype=np.float32)
                    valid_inv = depth > 1e-6
                    metric_depth = np.empty_like(depth, dtype=np.float32)
                    metric_depth[valid_inv] = 1.0 / depth[valid_inv]
                    metric_depth[~valid_inv] = np.finfo(np.float32).max
                    depth = metric_depth

                    min_depth = float(getattr(sensor.config, "MIN_DEPTH", 0.0))
                    max_depth = float(getattr(sensor.config, "MAX_DEPTH", 1.0))
                    normalize_depth = bool(
                        getattr(sensor.config, "NORMALIZE_DEPTH", True)
                    )

                    if normalize_depth:
                        denom = max(max_depth - min_depth, 1e-6)
                        depth = (depth - min_depth) / denom
                        depth = np.clip(depth, 0.0, 1.0)
                    else:
                        depth = np.clip(depth, min_depth, max_depth)

                    obs[sensor_uuid] = depth.astype(np.float32)[..., None]

        return obs

    # --- Utils ---
    def get_agent_state(self, agent_id: int = 0) -> AgentState:
        return self.agent_state

    def _door_teleport_config(self) -> Any:
        return getattr(self.config, "DOOR_TELEPORT", None)

    def _door_teleport_enabled(self) -> bool:
        cfg = self._door_teleport_config()
        if cfg is None:
            return True
        return bool(getattr(cfg, "ENABLED", True))

    def _door_teleport_trigger_distance(self) -> float:
        cfg = self._door_teleport_config()
        if cfg is None:
            return 0.5
        return float(getattr(cfg, "TRIGGER_DISTANCE", 0.5))

    def _door_teleport_landing_distance(self) -> float:
        cfg = self._door_teleport_config()
        if cfg is None:
            return 0.3
        return float(getattr(cfg, "LANDING_DISTANCE", 0.3))

    def _load_door_bboxes(
        self, scene_dir: Path
    ) -> List[Tuple[float, float, float, float]]:
        cfg = self._door_teleport_config()
        configured = ""
        if cfg is not None:
            configured = str(getattr(cfg, "METADATA_FILE", "") or "")

        candidates: List[Tuple[Path, bool]] = []
        if configured:
            metadata_path = Path(configured)
            if not metadata_path.is_absolute():
                metadata_path = scene_dir / metadata_path
            candidates.append((metadata_path, True))
        else:
            for filename in DEFAULT_DOOR_BBOX_FILENAMES:
                candidates.append(
                    (
                        scene_dir / filename,
                        filename in {"door_bboxes.json", "doors.json"},
                    )
                )

        for path, accept_unlabeled in candidates:
            if not path.exists():
                continue
            try:
                with path.open("r", encoding="utf-8") as fh:
                    payload = json.load(fh)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"Warning: Failed to load door bbox metadata from {path}: {exc}")
                continue

            bboxes = _extract_door_bboxes(payload, accept_unlabeled=accept_unlabeled)
            if bboxes:
                print(f"Door teleport: loaded {len(bboxes)} door bbox(es) from {path}")
                return bboxes

        return []

    def _distance_to_bbox_edge(
        self, xy: np.ndarray, bbox: Tuple[float, float, float, float]
    ) -> float:
        x, y = float(xy[0]), float(xy[1])
        min_x, min_y, max_x, max_y = bbox

        if min_x <= x <= max_x and min_y <= y <= max_y:
            return min(x - min_x, max_x - x, y - min_y, max_y - y)

        dx = max(min_x - x, 0.0, x - max_x)
        dy = max(min_y - y, 0.0, y - max_y)
        return math.hypot(dx, dy)

    def _ray_intersects_bbox(
        self,
        origin_xy: np.ndarray,
        direction_xy: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> bool:
        min_x, min_y, max_x, max_y = bbox
        t_min = 0.0
        t_max = float("inf")

        for axis, lower, upper in ((0, min_x, max_x), (1, min_y, max_y)):
            origin = float(origin_xy[axis])
            direction = float(direction_xy[axis])
            if abs(direction) < EPS:
                if origin < lower or origin > upper:
                    return False
                continue

            t1 = (lower - origin) / direction
            t2 = (upper - origin) / direction
            t_near = min(t1, t2)
            t_far = max(t1, t2)
            t_min = max(t_min, t_near)
            t_max = min(t_max, t_far)
            if t_min > t_max:
                return False

        return t_max >= 0.0

    def _door_crossing_axis(
        self, bbox: Tuple[float, float, float, float], direction_xy: np.ndarray
    ) -> int:
        min_x, min_y, max_x, max_y = bbox
        size_x = max_x - min_x
        size_y = max_y - min_y
        if abs(size_x - size_y) > EPS:
            return 0 if size_x < size_y else 1
        return 0 if abs(float(direction_xy[0])) >= abs(float(direction_xy[1])) else 1

    def _door_teleport_candidate(
        self,
        position: np.ndarray,
        direction_xy: np.ndarray,
        bbox: Tuple[float, float, float, float],
    ) -> Optional[np.ndarray]:
        origin_xy = np.asarray(position[:2], dtype=np.float32)
        trigger_distance = self._door_teleport_trigger_distance()
        if self._distance_to_bbox_edge(origin_xy, bbox) > trigger_distance:
            return None
        if not self._ray_intersects_bbox(origin_xy, direction_xy, bbox):
            return None

        min_x, min_y, max_x, max_y = bbox
        center_x = 0.5 * (min_x + max_x)
        center_y = 0.5 * (min_y + max_y)
        landing_distance = self._door_teleport_landing_distance()

        target_xy = np.array([center_x, center_y], dtype=np.float32)
        axis = self._door_crossing_axis(bbox, direction_xy)
        bounds = (min_x, max_x) if axis == 0 else (min_y, max_y)
        center = center_x if axis == 0 else center_y

        side = -1.0 if float(origin_xy[axis]) < center else 1.0
        if min_x <= origin_xy[0] <= max_x and min_y <= origin_xy[1] <= max_y:
            side = -1.0 if float(direction_xy[axis]) >= 0.0 else 1.0
        target_xy[axis] = (
            bounds[1] + landing_distance
            if side < 0.0
            else bounds[0] - landing_distance
        )

        return np.array([target_xy[0], target_xy[1], position[2]], dtype=np.float32)

    def _get_door_teleport_position(
        self, position: np.ndarray, forward: np.ndarray
    ) -> Optional[np.ndarray]:
        if not self._door_teleport_enabled() or not self.door_bboxes:
            return None

        direction_xy = np.asarray(forward[:2], dtype=np.float32)
        norm = float(np.linalg.norm(direction_xy))
        if norm < EPS:
            return None
        direction_xy = direction_xy / norm

        best_pos = None
        best_distance = float("inf")
        origin_xy = np.asarray(position[:2], dtype=np.float32)
        for bbox in self.door_bboxes:
            candidate = self._door_teleport_candidate(position, direction_xy, bbox)
            if candidate is None:
                continue
            distance = self._distance_to_bbox_edge(origin_xy, bbox)
            if distance < best_distance:
                best_distance = distance
                best_pos = candidate
        return best_pos

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        self.agent_state.position = position
        self.agent_state.rotation = rotation
        return True

    def seed(self, seed: int):
        np.random.seed(seed)
        torch.manual_seed(seed)

    def close(self):
        pass

    def get_occ_map_with_trajectory(
        self, pos_color: Tuple[int, int, int] = (255, 0, 0)
    ) -> Optional[np.ndarray]:
        """Get OCC map with trajectory and current position marked.
        Returns a COPY with only the current position marked in red.
        """
        if self.occ_map_with_trajectory is None or len(self.trajectory_pixels) == 0:
            return None

        # Make a copy to draw current position
        occ_copy = self.occ_map_with_trajectory.copy()
        current_pixel = self.trajectory_pixels[-1]
        # Draw red dot at current position
        cv2.circle(occ_copy, current_pixel, 10, pos_color, -1)
        return occ_copy

    def get_bev_map_with_trajectory(
        self, pos_color: Tuple[int, int, int] = (255, 0, 0)
    ) -> Optional[np.ndarray]:
        """Get BEV map with trajectory and current position marked.
        Returns a COPY with only the current position marked in red.
        """
        if self.bev_map_with_trajectory is None or len(self.trajectory_pixels) == 0:
            return None

        # Make a copy to draw current position
        bev_copy = self.bev_map_with_trajectory.copy()
        current_pixel = self.trajectory_pixels[-1]
        # Draw red dot at current position
        cv2.circle(bev_copy, current_pixel, 10, pos_color, -1)
        return bev_copy

    def get_occ_map_with_actions(
        self,
        actions: List[int],
        traj_color: Tuple[int, int, int] = (0, 0, 255),
        pos_color: Tuple[int, int, int] = (255, 0, 0),
    ) -> Optional[np.ndarray]:
        """
        Visualize the planned MPC trajectory on the occupancy map.
        Returns a COPY with trajectory and mpc plan marked.
        """
        if self.occ_map_with_trajectory is None or len(self.trajectory_pixels) == 0:
            return None

        # Start with the current map (with historical trajectory)
        occ_copy = self.occ_map_with_trajectory.copy()

        # Draw current position
        current_pixel = self.trajectory_pixels[-1]
        cv2.circle(occ_copy, current_pixel, 10, pos_color, -1)

        # Iterate actions to simulate future trajectory
        curr_pos = np.array(self.agent_state.position)
        curr_rot_obj = R.from_quat(self.agent_state.rotation)

        prev_pixel = current_pixel

        move_dist = self.config.FORWARD_STEP_SIZE
        turn_angle = math.radians(self.config.TURN_ANGLE)

        for action_id in actions:
            action_id = int(action_id)
            if action_id == 1:  # FWD
                local_fwd = np.array([1, 0, 0], dtype=np.float32)
                world_fwd = curr_rot_obj.apply(local_fwd)
                curr_pos = curr_pos + world_fwd * move_dist

                # Draw
                new_pixel = self.transform_from_world_to_pixel(
                    np.array([curr_pos[0], curr_pos[1], 0])
                )
                cv2.line(occ_copy, prev_pixel, new_pixel, traj_color, 3)
                prev_pixel = new_pixel

            elif action_id == 2:  # TURN_LEFT
                delta_rot = R.from_euler("z", turn_angle, degrees=False)
                curr_rot_obj = curr_rot_obj * delta_rot

            elif action_id == 3:  # TURN_RIGHT
                delta_rot = R.from_euler("z", -turn_angle, degrees=False)
                curr_rot_obj = curr_rot_obj * delta_rot

        return occ_copy

    def get_occ_map_with_pixels(
        self,
        pixels: List[Tuple[int, int]],
        path_color: Tuple[int, int, int] = (255, 255, 0),
        pos_color: Tuple[int, int, int] = (255, 0, 0),
    ) -> Optional[np.ndarray]:
        """
        Visualize the predicted pixel path on the occupancy map.
        Returns a COPY with trajectory and pixel path marked.
        """
        if self.occ_map_with_trajectory is None or len(self.trajectory_pixels) == 0:
            return None

        # Start with the current map
        occ_copy = self.occ_map_with_trajectory.copy()

        # Draw current position
        current_pixel = self.trajectory_pixels[-1]
        cv2.circle(occ_copy, current_pixel, 10, pos_color, -1)

        if not pixels:
            return occ_copy

        prev_pixel = current_pixel
        # Draw Cyan lines for pixel path (BGR: 255, 255, 0)

        for px in pixels:
            cv2.line(occ_copy, prev_pixel, px, path_color, 3)
            prev_pixel = px

        return occ_copy

    def draw_gt_on_map(
        self,
        map: np.ndarray,
        pixels: List[Tuple[int, int]],
        path_color: Tuple[int, int, int] = (255, 255, 0),
    ) -> Optional[np.ndarray]:
        if not pixels:
            return None

        prev_pixel = pixels[0]
        for px in pixels[1:]:
            cv2.line(map, prev_pixel, px, path_color, 3)
            prev_pixel = px
        draw_solid_star(map, pixels[0], 15, (0, 255, 0))
        draw_solid_triangle(map, pixels[-1], 15, (110, 28, 140))

        return map

    def draw_step(
        self,
        map: np.ndarray,
        start: Tuple[int, int],
        end: Tuple[int, int],
        path_color: Tuple[int, int, int] = (255, 255, 0),
    ):
        return cv2.line(
            map,
            start,
            end,
            path_color,
            3,
        )

    def get_occ_map(
        self,
        start_color: Tuple[int, int, int] = (0, 255, 0),
        pos_color: Tuple[int, int, int] = (255, 0, 0),
    ) -> Optional[np.ndarray]:
        """Get OCC map with only start and current position marked.
        No trajectory lines. Returns a COPY.
        """
        if (
            self.occ_map_original is None
            or self.start_pixel is None
            or len(self.trajectory_pixels) == 0
        ):
            return None

        # Make a copy from original (without trajectory)
        occ_copy = self.occ_map_original.copy()
        # Draw green dot at start position
        cv2.circle(occ_copy, self.start_pixel, 3, start_color, -1)
        # Draw red dot at current position
        current_pixel = self.trajectory_pixels[-1]
        cv2.circle(occ_copy, current_pixel, 3, pos_color, -1)
        return occ_copy

    def get_current_pixel_position(self) -> Optional[Tuple[int, int]]:
        """Get current agent position in pixel coordinates."""
        if len(self.trajectory_pixels) == 0:
            return None
        return self.trajectory_pixels[-1]

    # --- Properties needed by GN_Bench ---
    @property
    def sensor_suite(self):
        return self._sensor_suite

    @property
    def action_space(self):
        return spaces.Discrete(4)

    @property
    def is_episode_active(self):
        return True

    def geodesic_distance(
        self,
        position_a: Union[Sequence[float], ndarray],
        position_b: Union[Sequence[float], Sequence[Sequence[float]]],
    ) -> float:
        # 1. Convert Start to Pixel
        if not isinstance(position_a, np.ndarray):
            position_a = np.array(position_a)
        start_pos = np.array(position_a, dtype=np.float32)
        start_px = self.transform_from_world_to_pixel(start_pos)
        if not self._is_passable_pixel(start_px):
            fallback_px = None
            if self._last_passable_position is not None:
                candidate_pos = np.array(self._last_passable_position, dtype=np.float32)
                candidate_px = self.transform_from_world_to_pixel(candidate_pos)
                if self._is_passable_pixel(candidate_px):
                    start_pos = candidate_pos
                    fallback_px = candidate_px

            if fallback_px is None:
                nearest_px = self._find_nearest_passable_pixel(start_px)
                if nearest_px is not None:
                    nearest_xy = self.transform_from_pixel_to_world(nearest_px)
                    start_pos = np.array(
                        [nearest_xy[0], nearest_xy[1], start_pos[2]], dtype=np.float32
                    )
                    fallback_px = nearest_px

            if fallback_px is not None:
                start_px = fallback_px

        # 2. Handle Goal(s)
        if len(position_b) == 0:
            return 0.0

        if isinstance(position_b[0], (float, int)):
            targets_world = [position_b]
        else:
            targets_world = position_b

        min_distance = float("inf")

        # 3. Iterate through all goals to find the closest reachable one
        raw_path = None
        path_errors: List[str] = []
        for target_world in targets_world:
            if not isinstance(target_world, np.ndarray):
                target_world = np.array(target_world)

            goal_px = self.transform_from_world_to_pixel(target_world)
            if not self._is_passable_pixel(goal_px):
                snapped_goal_px = self._find_nearest_passable_pixel(goal_px)
                if snapped_goal_px is not None:
                    goal_px = snapped_goal_px

            if start_px == goal_px:
                return 0.0

            # Run A* with margin fallback: 7px -> 6px -> 5px
            try:
                raw_path = self._astar_with_fallback_margin(start_px, goal_px)
            except RuntimeError as e:
                path_errors.append(str(e))
                raw_path = None

            if raw_path is not None:
                smoothed_nodes = self._smooth_path(raw_path)
                path = self._densify_path(smoothed_nodes)
                world_path = [self.transform_from_pixel_to_world(p) for p in path]

                current_world_dist = 0.0
                for i in range(len(world_path) - 1):
                    wx0, wy0 = world_path[i]
                    wx1, wy1 = world_path[i + 1]

                    current_world_dist += math.hypot(wx1 - wx0, wy1 - wy0)

                if current_world_dist < min_distance:
                    min_distance = current_world_dist

        if min_distance == float("inf"):
            tried_s = "->".join(f"{m}px" for m in self.margins)
            err_msg = (
                f"No path found from {start_px} to any goal with margins {tried_s}."
            )
            if len(path_errors) > 0:
                err_msg += f" Last error: {path_errors[-1]}"
            print(f"Warning: {err_msg}")

        return min_distance

    def _is_passable_pixel(self, px: Tuple[int, int]) -> bool:
        x, y = px
        if not (0 <= x < self.map_width and 0 <= y < self.map_height):
            return False

        grid = None
        if self.margins:
            grid = self.safe_passable_grids.get(self.margins[-1])
        if grid is None:
            grid = self.passable_grid
        if grid is None:
            return False
        return bool(grid[y][x])

    def _update_last_passable_position(self, position: np.ndarray) -> None:
        if self.passable_grid is None:
            return

        px = self.transform_from_world_to_pixel(position)
        if self._is_passable_pixel(px):
            self._last_passable_position = np.array(position, dtype=np.float32)

    def _find_nearest_passable_pixel(
        self, px: Tuple[int, int], max_radius: Optional[int] = None
    ) -> Optional[Tuple[int, int]]:
        if self.passable_grid is None or self.map_width <= 0 or self.map_height <= 0:
            return None

        x0 = int(min(max(px[0], 0), self.map_width - 1))
        y0 = int(min(max(px[1], 0), self.map_height - 1))

        if self._is_passable_pixel((x0, y0)):
            return (x0, y0)

        if max_radius is None:
            max_radius = max(self.map_width, self.map_height)

        for r in range(1, max_radius + 1):
            left = max(x0 - r, 0)
            right = min(x0 + r, self.map_width - 1)
            top = max(y0 - r, 0)
            bottom = min(y0 + r, self.map_height - 1)

            for x in range(left, right + 1):
                if self._is_passable_pixel((x, top)):
                    return (x, top)
                if self._is_passable_pixel((x, bottom)):
                    return (x, bottom)

            for y in range(top + 1, bottom):
                if self._is_passable_pixel((left, y)):
                    return (left, y)
                if self._is_passable_pixel((right, y)):
                    return (right, y)

        return None

    def get_observations_at(
        self,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        keep_agent_at_new_pose: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()
        if position is None or rotation is None:
            success = True
        else:
            success = self.set_agent_state(position, rotation, reset_sensors=False)

        if success:
            sim_obs = self.get_sensor_observations()

            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose:
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )
            return observations
        else:
            return None

    def transform_from_world_to_pixel(self, position: np.ndarray) -> Tuple[int, int]:
        wx, wy = position[0], position[1]

        span_x = self.meta["upper"][0] - self.meta["lower"][0]
        span_y = self.meta["upper"][1] - self.meta["lower"][1]

        scale_x = span_x / max(self.map_width, 1)
        scale_y = span_y / max(self.map_height, 1)

        px = (-wx - self.meta["lower"][0]) / scale_x - 0.5
        py = (self.meta["upper"][1] + wy) / scale_y - 0.5

        px_int = int(round(px))
        py_int = int(round(py))

        px_final = max(0, min(px_int, self.map_width - 1))
        py_final = max(0, min(py_int, self.map_height - 1))

        return px_final, py_final

    def transform_from_pixel_to_world(
        self, pixel: Tuple[int, int]
    ) -> Tuple[float, float]:
        if self.meta is None:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)

        px, py = pixel

        span_x = self.meta["upper"][0] - self.meta["lower"][0]
        span_y = self.meta["upper"][1] - self.meta["lower"][1]

        scale_x = span_x / max(self.map_width, 1)
        scale_y = span_y / max(self.map_height, 1)

        flipped_x = self.meta["lower"][0] + (px + 0.5) * scale_x
        flipped_y = self.meta["upper"][1] - (py + 0.5) * scale_y

        world_x = -flipped_x
        world_y = -flipped_y

        return float(world_x), float(world_y)

    def _astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid: Optional[List[List[bool]]] = None,
    ) -> float:
        """
        A* pathfinding on provided grid. Returns path in pixels.
        """
        # width, height = self.map_width, self.map_height
        height = len(grid)
        width = len(grid[0]) if height else 0

        def heuristic(a, b):
            # Euclidean distance heuristic
            return math.hypot(a[0] - b[0], a[1] - b[1])

        neighbor_offsets = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]

        open_heap = []
        heapq.heappush(open_heap, (heuristic(start, goal), start))
        came_from = {}
        g_score = {start: 0.0}

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            cx, cy = current
            for dx, dy in neighbor_offsets:
                nx, ny = cx + dx, cy + dy

                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                if not grid[ny][nx]:
                    continue
                if dx != 0 and dy != 0:
                    if not (
                        0 <= cy < height
                        and 0 <= nx < width
                        and 0 <= ny < height
                        and 0 <= cx < width
                    ):
                        continue
                    if not grid[cy][nx] or not grid[ny][cx]:
                        continue

                step_cost = math.hypot(dx, dy)
                tentative_g = g_score[current] + step_cost
                neighbor = (nx, ny)

                if tentative_g >= g_score.get(neighbor, float("inf")):
                    continue

                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_heap, (f_score, neighbor))

        return None

    def _astar_with_fallback_margin(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """
        Try A* with primary margin first, then fallback margins.
        Raises RuntimeError if all attempts fail.
        """
        if self.passable_grid is None:
            raise RuntimeError("passable_grid is not initialized")

        margins = [int(m) for m in self.margins] if self.margins else [0]
        tried: List[int] = []

        for margin in margins:
            grid = self.safe_passable_grids.get(margin)
            tried.append(margin)
            path = self._astar(start, goal, grid)
            if path is not None:
                return path

        tried_s = " -> ".join(f"{m}px" for m in tried)
        raise RuntimeError(f"A* path not found with fallback margins: {tried_s}.")

    def _bresenham_line(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Bresenham's Line Algorithm to get all pixels between two points."""
        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x1 >= x0 else -1
        sy = 1 if y1 >= y0 else -1
        x, y = x0, y0
        points = [(x, y)]

        if dx >= dy:
            err = dx // 2
            while x != x1:
                x += sx
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                points.append((x, y))
        else:
            err = dy // 2
            while y != y1:
                y += sy
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                points.append((x, y))
        return points

    def _has_line_of_sight(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if there is a straight line collision-free path."""
        primary_margin = int(self.margins[0]) if self.margins else 0
        grid = self.safe_passable_grids.get(primary_margin)
        if grid is None:
            return False

        for x, y in self._bresenham_line(start, end):
            if not (0 <= x < self.map_width and 0 <= y < self.map_height):
                return False
            if not grid[y][x]:
                return False
        return True

    def _smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Greedy path smoothing.
        Reduces a zig-zag grid path to a sequence of straight line keypoints.
        """
        if not path or len(path) <= 2:
            return path

        smoothed = [path[0]]
        index = 0
        last_idx = len(path) - 1

        while index < last_idx:
            next_index = index + 1
            for candidate in range(last_idx, index, -1):
                if self._has_line_of_sight(path[index], path[candidate]):
                    next_index = candidate
                    break
            smoothed.append(path[next_index])
            index = next_index

        return smoothed

    def _densify_path(self, nodes: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not nodes:
            return []
        expanded: List[Tuple[int, int]] = [nodes[0]]
        for a, b in zip(nodes, nodes[1:]):
            segment = self._bresenham_line(a, b)
            expanded.extend(segment[1:])
        return expanded

    def _clip_motion_to_free_space(
        self,
        start_world: np.ndarray,
        end_world: np.ndarray,
        grid: List[List[bool]],
    ) -> Tuple[np.ndarray, bool]:
        """Clip motion to the last free cell along the segment.

        Returns the adjusted world position and whether a collision happened.
        """
        start_px = self.transform_from_world_to_pixel(start_world)
        end_px = self.transform_from_world_to_pixel(end_world)

        if start_px == end_px:
            return end_world, False

        last_free = start_px
        collided = False
        for px in self._bresenham_line(start_px, end_px):
            x, y = px
            if not (0 <= x < self.map_width and 0 <= y < self.map_height):
                collided = True
                break
            if not grid[y][x]:
                collided = True
                break
            last_free = px

        if not collided:
            return end_world, False

        clipped_xy = self.transform_from_pixel_to_world(last_free)
        return np.array(
            [clipped_xy[0], clipped_xy[1], start_world[2]], dtype=np.float32
        ), True


class PathSampler:
    """Lightweight sampler to query positions/tangents along a 2D polyline."""

    def __init__(self, points: Sequence[np.ndarray]):
        if len(points) < 2:
            raise ValueError("PathSampler requires at least two points.")
        raw = np.asarray(points, dtype=np.float32)
        diffs = raw[1:] - raw[:-1]
        lengths = np.linalg.norm(diffs, axis=1)
        valid = lengths > EPS
        cleaned = [raw[0]]
        vectors: list[np.ndarray] = []
        seg_lengths: list[float] = []
        for idx, ok in enumerate(valid):
            if not ok:
                continue
            cleaned.append(raw[idx + 1])
            vectors.append(diffs[idx])
            seg_lengths.append(float(lengths[idx]))
        self.points = np.asarray(cleaned, dtype=np.float32)
        self.segment_vectors = np.asarray(vectors, dtype=np.float32)
        self.segment_lengths = np.asarray(seg_lengths, dtype=np.float32)
        self.cumulative = np.concatenate(
            [np.array([0.0], dtype=np.float32), np.cumsum(self.segment_lengths)]
        )

    @property
    def total_length(self) -> float:
        return float(self.cumulative[-1])

    def position_at(self, distance: float) -> np.ndarray:
        if distance <= 0.0:
            direction = self.segment_vectors[0] / self.segment_lengths[0]
            return self.points[0] + direction * distance
        total = self.total_length
        if distance >= total:
            direction = self.segment_vectors[-1] / self.segment_lengths[-1]
            return self.points[-1] + direction * (distance - total)
        seg_idx = int(np.searchsorted(self.cumulative, distance, side="right") - 1)
        seg_offset = distance - self.cumulative[seg_idx]
        ratio = seg_offset / self.segment_lengths[seg_idx]
        return self.points[seg_idx] + self.segment_vectors[seg_idx] * ratio

    def direction_at(self, distance: float) -> np.ndarray:
        if distance <= 0.0:
            vec = self.segment_vectors[0]
        elif distance >= self.total_length:
            vec = self.segment_vectors[-1]
        else:
            seg_idx = int(np.searchsorted(self.cumulative, distance, side="right") - 1)
            vec = self.segment_vectors[seg_idx]
        norm = np.linalg.norm(vec)
        if norm < EPS:
            return np.array([0.0, 1.0], dtype=np.float32)
        return vec / norm

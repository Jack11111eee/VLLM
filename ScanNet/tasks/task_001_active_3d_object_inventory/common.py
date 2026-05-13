#!/usr/bin/env python3
"""Shared helpers for Task 001 active 3D object inventory."""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any, Iterable


TASK_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASK_DIR.parents[2]
DEFAULT_CONFIG = TASK_DIR / "config.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = load_json(path)
    for key in ("manifest", "scene_list", "scannet_root", "output_dir"):
        config[key] = str(resolve_repo_path(config[key]))
    return config


def read_scene_list(path: Path) -> list[str]:
    scenes = []
    for line in path.read_text().splitlines():
        scene = line.strip()
        if scene and not scene.startswith("#"):
            scenes.append(scene)
    return scenes


def as_object_key(object_id: int | str) -> str:
    return str(int(object_id))


def sorted_int_keys(mapping: dict[str, Any]) -> list[str]:
    return sorted(mapping, key=lambda value: int(value))


def read_ply_vertices_xyz(path: Path) -> list[tuple[float, float, float]]:
    """Read vertex xyz coordinates from an ASCII or binary little-endian PLY file."""
    with path.open("rb") as file:
        header_lines: list[str] = []
        while True:
            line = file.readline()
            if not line:
                raise ValueError(f"PLY header is missing end_header: {path}")
            decoded = line.decode("ascii").strip()
            header_lines.append(decoded)
            if decoded == "end_header":
                break

        fmt = None
        vertex_count = None
        vertex_properties: list[tuple[str, str]] = []
        in_vertex = False
        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif in_vertex and parts[0] == "property" and len(parts) == 3:
                vertex_properties.append((parts[1], parts[2]))

        if fmt not in {"ascii", "binary_little_endian"}:
            raise ValueError(f"Unsupported PLY format {fmt!r}: {path}")
        if vertex_count is None:
            raise ValueError(f"PLY has no vertex element: {path}")

        property_types = [prop_type for prop_type, _ in vertex_properties]
        property_names = [name for _, name in vertex_properties]
        try:
            x_index = property_names.index("x")
            y_index = property_names.index("y")
            z_index = property_names.index("z")
        except ValueError as exc:
            raise ValueError(f"PLY vertex properties must include x/y/z: {path}") from exc

        if fmt == "ascii":
            vertices = []
            for _ in range(vertex_count):
                values = file.readline().decode("ascii").split()
                vertices.append(
                    (float(values[x_index]), float(values[y_index]), float(values[z_index]))
                )
            return vertices

        struct_format = "<" + "".join(_PLY_STRUCT_TYPES[prop_type] for prop_type in property_types)
        row_size = struct.calcsize(struct_format)
        unpack = struct.Struct(struct_format).unpack
        vertices = []
        for _ in range(vertex_count):
            row = file.read(row_size)
            if len(row) != row_size:
                raise ValueError(f"Unexpected EOF while reading vertices from {path}")
            values = unpack(row)
            vertices.append((float(values[x_index]), float(values[y_index]), float(values[z_index])))
        return vertices


_PLY_STRUCT_TYPES = {
    "char": "b",
    "int8": "b",
    "uchar": "B",
    "uint8": "B",
    "short": "h",
    "int16": "h",
    "ushort": "H",
    "uint16": "H",
    "int": "i",
    "int32": "i",
    "uint": "I",
    "uint32": "I",
    "float": "f",
    "float32": "f",
    "double": "d",
    "float64": "d",
}


def compute_aabb(points: Iterable[tuple[float, float, float]]) -> dict[str, list[float]]:
    iterator = iter(points)
    try:
        first = next(iterator)
    except StopIteration as exc:
        raise ValueError("Cannot compute an AABB from zero points") from exc

    mins = [first[0], first[1], first[2]]
    maxs = [first[0], first[1], first[2]]
    for point in iterator:
        for axis in range(3):
            value = point[axis]
            if value < mins[axis]:
                mins[axis] = value
            if value > maxs[axis]:
                maxs[axis] = value

    size = [maxs[axis] - mins[axis] for axis in range(3)]
    center = [(mins[axis] + maxs[axis]) / 2.0 for axis in range(3)]
    return {
        "center": [round(value, 6) for value in center],
        "size": [round(value, 6) for value in size],
        "min": [round(value, 6) for value in mins],
        "max": [round(value, 6) for value in maxs],
    }


def bbox_iou_3d(box_a: dict[str, list[float]], box_b: dict[str, list[float]]) -> float:
    if box_a == box_b:
        return 1.0

    mins = [max(box_a["min"][axis], box_b["min"][axis]) for axis in range(3)]
    maxs = [min(box_a["max"][axis], box_b["max"][axis]) for axis in range(3)]
    intersection_size = [max(0.0, maxs[axis] - mins[axis]) for axis in range(3)]
    intersection = intersection_size[0] * intersection_size[1] * intersection_size[2]
    if intersection <= 0.0:
        return 0.0
    volume_a = _bbox_volume(box_a)
    volume_b = _bbox_volume(box_b)
    union = volume_a + volume_b - intersection
    iou = intersection / union if union > 0.0 else 0.0
    return min(1.0, max(0.0, iou))


def _bbox_volume(box: dict[str, list[float]]) -> float:
    size = box["size"]
    return max(0.0, size[0]) * max(0.0, size[1]) * max(0.0, size[2])

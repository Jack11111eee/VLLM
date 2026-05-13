#!/usr/bin/env python3
"""Build Task 001 GT object boxes and frame-level visibility targets."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from common import (
    DEFAULT_CONFIG,
    as_object_key,
    compute_aabb,
    load_config,
    load_json,
    read_ply_vertices_xyz,
    read_scene_list,
    sorted_int_keys,
    write_json,
)


STRUCTURE_DEFAULTS = {"wall", "floor", "ceiling"}


def build_segment_to_vertices(seg_indices: list[int]) -> dict[int, list[int]]:
    segment_to_vertices: dict[int, list[int]] = defaultdict(list)
    for vertex_index, segment_id in enumerate(seg_indices):
        segment_to_vertices[int(segment_id)].append(vertex_index)
    return dict(segment_to_vertices)


def build_objects(
    aggregation: dict[str, Any],
    segments: dict[str, Any],
    vertices: list[tuple[float, float, float]],
    exclude_categories: set[str],
) -> dict[str, dict[str, Any]]:
    segment_to_vertices = build_segment_to_vertices(segments["segIndices"])
    objects: dict[str, dict[str, Any]] = {}

    for group in aggregation["segGroups"]:
        category = str(group["label"]).strip()
        if category.lower() in exclude_categories:
            continue

        object_id = int(group["objectId"])
        vertex_indices: list[int] = []
        missing_segments: list[int] = []
        for segment_id in group["segments"]:
            segment_vertices = segment_to_vertices.get(int(segment_id))
            if not segment_vertices:
                missing_segments.append(int(segment_id))
                continue
            vertex_indices.extend(segment_vertices)

        if not vertex_indices:
            continue

        bbox = compute_aabb(vertices[index] for index in vertex_indices)
        if any(axis_size <= 0.0 for axis_size in bbox["size"]):
            continue

        objects[as_object_key(object_id)] = {
            "object_id": object_id,
            "instance_image_value": object_id + 1,
            "category": category,
            "bbox_3d": bbox,
            "vertex_count": len(vertex_indices),
            "segment_count": len(group["segments"]),
            "missing_segment_count": len(missing_segments),
        }

    return {key: objects[key] for key in sorted_int_keys(objects)}


def zip_png_frame_ids(path: Path, prefix: str) -> set[int]:
    ids: set[int] = set()
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.startswith(prefix) or not name.endswith(".png"):
                continue
            try:
                ids.add(int(Path(name).stem))
            except ValueError:
                continue
    return ids


def read_png_from_zip(archive: zipfile.ZipFile, name: str) -> np.ndarray:
    with archive.open(name) as file:
        image = Image.open(BytesIO(file.read()))
        return np.array(image)


def build_frame_visibility(
    scene: dict[str, Any],
    scene_raw_dir: Path,
    scene_id: str,
    objects: dict[str, dict[str, Any]],
    min_visible_pixels: int,
) -> tuple[dict[str, list[int]], dict[str, Any]]:
    instance_zip = scene_raw_dir / f"{scene_id}_2d-instance-filt.zip"
    label_zip = scene_raw_dir / f"{scene_id}_2d-label-filt.zip"
    if not instance_zip.exists():
        raise FileNotFoundError(f"Missing 2D instance annotations: {instance_zip}")
    if not label_zip.exists():
        raise FileNotFoundError(f"Missing 2D label annotations: {label_zip}")

    instance_ids = zip_png_frame_ids(instance_zip, "instance-filt/")
    label_ids = zip_png_frame_ids(label_zip, "label-filt/")
    manifest_frame_ids = {int(frame_id) for frame_id in scene["frame_ids"]}
    matched_frame_ids = sorted(manifest_frame_ids & instance_ids & label_ids)

    object_ids_by_instance_value = {
        int(obj["instance_image_value"]): int(obj["object_id"]) for obj in objects.values()
    }

    frame_visibility: dict[str, list[int]] = {}
    visible_pixel_counts: dict[str, dict[str, int]] = {}
    with zipfile.ZipFile(instance_zip) as archive:
        for frame_id in matched_frame_ids:
            image = read_png_from_zip(archive, f"instance-filt/{frame_id}.png")
            counts = Counter(int(value) for value in image.reshape(-1) if int(value) != 0)
            visible_objects = []
            frame_counts: dict[str, int] = {}
            for instance_value, pixel_count in sorted(counts.items()):
                object_id = object_ids_by_instance_value.get(instance_value)
                if object_id is None or pixel_count < min_visible_pixels:
                    continue
                visible_objects.append(object_id)
                frame_counts[as_object_key(object_id)] = int(pixel_count)
            frame_visibility[str(frame_id)] = visible_objects
            visible_pixel_counts[str(frame_id)] = frame_counts

    metadata = {
        "instance_zip": str(instance_zip),
        "label_zip": str(label_zip),
        "manifest_frame_count": len(manifest_frame_ids),
        "instance_annotation_frame_count": len(instance_ids),
        "label_annotation_frame_count": len(label_ids),
        "matched_frame_count": len(matched_frame_ids),
        "missing_instance_frames": sorted(manifest_frame_ids - instance_ids),
        "missing_label_frames": sorted(manifest_frame_ids - label_ids),
        "visible_pixel_counts": visible_pixel_counts,
    }
    return frame_visibility, metadata


def build_scene_targets(
    scene: dict[str, Any],
    scannet_root: Path,
    exclude_categories: set[str],
    min_visible_pixels: int,
) -> dict[str, Any]:
    scene_id = scene["scene_id"]
    raw_dir = scannet_root / "raw" / "scans" / scene_id
    aggregation_path = raw_dir / f"{scene_id}.aggregation.json"
    segments_path = raw_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    labels_mesh_path = raw_dir / f"{scene_id}_vh_clean_2.labels.ply"

    aggregation = load_json(aggregation_path)
    segments = load_json(segments_path)
    vertices = read_ply_vertices_xyz(labels_mesh_path)
    objects = build_objects(
        aggregation=aggregation,
        segments=segments,
        vertices=vertices,
        exclude_categories=exclude_categories,
    )
    frame_visibility, visibility_metadata = build_frame_visibility(
        scene=scene,
        scene_raw_dir=raw_dir,
        scene_id=scene_id,
        objects=objects,
        min_visible_pixels=min_visible_pixels,
    )

    visible_object_ids = {
        object_id for visible in frame_visibility.values() for object_id in visible
    }
    return {
        "scene_id": scene_id,
        "objects": objects,
        "frame_visibility": frame_visibility,
        "summary": {
            "object_count": len(objects),
            "visible_object_count": len(visible_object_ids),
            "category_count": len({obj["category"] for obj in objects.values()}),
            "frame_count": len(scene["frame_ids"]),
            "frames_with_visible_objects": sum(1 for visible in frame_visibility.values() if visible),
        },
        "source_files": {
            "aggregation": str(aggregation_path),
            "segments": str(segments_path),
            "labels_mesh": str(labels_mesh_path),
        },
        "visibility_metadata": visibility_metadata,
    }


def build_targets(config: dict[str, Any]) -> dict[str, Any]:
    manifest = load_json(Path(config["manifest"]))
    scene_filter = set(read_scene_list(Path(config["scene_list"])))
    scannet_root = Path(config["scannet_root"])
    exclude_categories = {
        category.lower() for category in config.get("exclude_categories", STRUCTURE_DEFAULTS)
    }
    min_visible_pixels = int(config["min_visible_pixels"])

    scenes = {}
    for scene_id, scene in manifest["scenes"].items():
        if scene_filter and scene_id not in scene_filter:
            continue
        scenes[scene_id] = build_scene_targets(
            scene=scene,
            scannet_root=scannet_root,
            exclude_categories=exclude_categories,
            min_visible_pixels=min_visible_pixels,
        )

    return {
        "version": 1,
        "task": config["task_name"],
        "dataset": manifest["dataset"],
        "min_visible_pixels": min_visible_pixels,
        "exclude_categories": sorted(exclude_categories),
        "object_id_convention": {
            "object_id": "ScanNet aggregation objectId, zero-indexed",
            "instance_image_value": "Filtered 2D instance PNG value; equals object_id + 1",
        },
        "scenes": scenes,
        "summary": {
            "scene_count": len(scenes),
            "object_count": sum(scene["summary"]["object_count"] for scene in scenes.values()),
            "visible_object_count": sum(
                scene["summary"]["visible_object_count"] for scene in scenes.values()
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-visible-pixels", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.min_visible_pixels is not None:
        config["min_visible_pixels"] = args.min_visible_pixels
    targets = build_targets(config)
    output = args.output or (Path(config["output_dir"]) / "targets.json")
    write_json(output, targets)

    for scene_id, scene in targets["scenes"].items():
        summary = scene["summary"]
        print(
            f"{scene_id}: {summary['object_count']} objects, "
            f"{summary['visible_object_count']} visible, "
            f"{summary['frames_with_visible_objects']} frames with visible objects"
        )
    print(
        f"total: {targets['summary']['scene_count']} scenes, "
        f"{targets['summary']['object_count']} objects"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

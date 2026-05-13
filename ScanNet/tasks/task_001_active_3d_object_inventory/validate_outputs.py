#!/usr/bin/env python3
"""Validate Task 001 generated targets, predictions, and metrics."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from typing import Any

from common import DEFAULT_CONFIG, load_config, load_json


def validate_targets(targets: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for scene_id, scene in targets["scenes"].items():
        if not scene["objects"]:
            errors.append(f"{scene_id}: no target objects")

        for object_key, obj in scene["objects"].items():
            if str(obj["object_id"]) != object_key:
                errors.append(f"{scene_id}: object key/id mismatch for {object_key}")
            bbox = obj["bbox_3d"]
            if any(axis <= 0 for axis in bbox["size"]):
                errors.append(f"{scene_id}: non-positive bbox size for object {object_key}")
            for axis in range(3):
                expected = round(bbox["max"][axis] - bbox["min"][axis], 6)
                if abs(expected - bbox["size"][axis]) > 1e-5:
                    errors.append(f"{scene_id}: bbox size mismatch for object {object_key}")

        metadata = scene["visibility_metadata"]
        if metadata["missing_instance_frames"]:
            errors.append(
                f"{scene_id}: exported frames missing from instance zip: "
                f"{metadata['missing_instance_frames'][:5]}"
            )
        if metadata["missing_label_frames"]:
            errors.append(
                f"{scene_id}: exported frames missing from label zip: "
                f"{metadata['missing_label_frames'][:5]}"
            )
        with zipfile.ZipFile(metadata["instance_zip"]) as archive:
            instance_names = set(archive.namelist())
        with zipfile.ZipFile(metadata["label_zip"]) as archive:
            label_names = set(archive.namelist())
        for frame_id in scene["frame_visibility"]:
            expected_instance = f"instance-filt/{frame_id}.png"
            expected_label = f"label-filt/{frame_id}.png"
            if expected_instance not in instance_names:
                errors.append(
                    f"{scene_id}: visible frame missing from instance zip: {expected_instance}"
                )
            if expected_label not in label_names:
                errors.append(f"{scene_id}: visible frame missing from label zip: {expected_label}")

        object_ids = set(scene["objects"])
        for frame_id, visible_ids in scene["frame_visibility"].items():
            for object_id in visible_ids:
                if str(object_id) not in object_ids:
                    errors.append(
                        f"{scene_id}: frame {frame_id} references unknown object {object_id}"
                    )
    return errors


def validate_predictions(targets: dict[str, Any], predictions: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    max_steps = int(predictions["max_steps"])
    for scene_id, scene_predictions in predictions["scenes"].items():
        scene_targets = targets["scenes"][scene_id]
        visited = scene_predictions["visited_frames"]
        if not visited:
            errors.append(f"{scene_id}: no visited frames")
        if len(visited) > max_steps + 1:
            errors.append(f"{scene_id}: trajectory longer than max_steps + start frame")

        visited_set = set(visited)
        target_objects = scene_targets["objects"]
        for object_key, obj in scene_predictions["objects"].items():
            if object_key not in target_objects:
                errors.append(f"{scene_id}: predicted unknown object {object_key}")
                continue
            if obj["bbox_3d"] != target_objects[object_key]["bbox_3d"]:
                errors.append(f"{scene_id}: predicted bbox does not match target for {object_key}")
            evidence = obj["evidence_frames"]
            if not evidence:
                errors.append(f"{scene_id}: object {object_key} has no evidence frames")
            for frame_id in evidence:
                if frame_id not in visited_set:
                    errors.append(
                        f"{scene_id}: evidence frame {frame_id} was not visited for {object_key}"
                    )
                if int(object_key) not in scene_targets["frame_visibility"].get(str(frame_id), []):
                    errors.append(
                        f"{scene_id}: evidence frame {frame_id} does not show {object_key}"
                    )
    return errors


def validate_metrics(metrics: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for scene_id, scene_metrics in metrics["per_scene"].items():
        for name in ("instance_recall", "category_recall", "oracle_box_iou_mean"):
            value = scene_metrics[name]
            if value < 0.0 or value > 1.0:
                errors.append(f"{scene_id}: {name} out of range: {value}")
        curve = scene_metrics["discovery_curve"]
        if any(curve[index] > curve[index + 1] for index in range(len(curve) - 1)):
            errors.append(f"{scene_id}: discovery curve is not monotonic")

    for name in ("instance_recall", "mean_scene_instance_recall", "mean_scene_category_recall"):
        value = metrics["metrics"][name]
        if value < 0.0 or value > 1.0:
            errors.append(f"aggregate: {name} out of range: {value}")
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--policy", choices=("random", "forward", "all"), default="all")
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir or Path(config["output_dir"])
    targets = load_json(output_dir / "targets.json")

    errors = validate_targets(targets)
    policies = ("random", "forward") if args.policy == "all" else (args.policy,)
    for policy in policies:
        predictions = load_json(output_dir / f"predictions_{policy}.json")
        metrics = load_json(output_dir / f"metrics_{policy}.json")
        errors.extend(validate_predictions(targets, predictions))
        errors.extend(validate_metrics(metrics))

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)

    print(f"validated targets plus {', '.join(policies)} predictions/metrics")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate Task 001 object inventory predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import DEFAULT_CONFIG, bbox_iou_3d, load_config, load_json, write_json


def evaluate_scene(scene_targets: dict[str, Any], scene_predictions: dict[str, Any]) -> dict[str, Any]:
    target_objects = scene_targets["objects"]
    predicted_objects = scene_predictions["objects"]
    target_ids = set(target_objects)
    predicted_ids = set(predicted_objects)
    matched_ids = target_ids & predicted_ids

    target_categories = {obj["category"] for obj in target_objects.values()}
    predicted_categories = {
        predicted_objects[object_id]["category"]
        for object_id in predicted_ids
        if object_id in target_objects
    }

    discovery_curve = []
    discovered_so_far: set[str] = set()
    for frame_id in scene_predictions["visited_frames"]:
        for object_id in scene_targets["frame_visibility"].get(str(frame_id), []):
            object_key = str(int(object_id))
            if object_key in target_ids and object_key in predicted_ids:
                discovered_so_far.add(object_key)
        discovery_curve.append(len(discovered_so_far))

    box_ious = [
        bbox_iou_3d(
            target_objects[object_id]["bbox_3d"],
            predicted_objects[object_id]["bbox_3d"],
        )
        for object_id in sorted(matched_ids, key=int)
    ]

    target_count = len(target_ids)
    category_count = len(target_categories)
    return {
        "scene_id": scene_targets["scene_id"],
        "target_object_count": target_count,
        "predicted_object_count": len(predicted_ids),
        "matched_object_count": len(matched_ids),
        "instance_recall": len(matched_ids) / target_count if target_count else 0.0,
        "category_recall": (
            len(target_categories & predicted_categories) / category_count if category_count else 0.0
        ),
        "discovery_curve": discovery_curve,
        "mean_discovered_objects_per_step": (
            sum(discovery_curve) / len(discovery_curve) if discovery_curve else 0.0
        ),
        "oracle_box_iou_mean": sum(box_ious) / len(box_ious) if box_ious else 0.0,
        "oracle_box_iou_min": min(box_ious) if box_ious else 0.0,
    }


def aggregate(per_scene: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scene_count = len(per_scene)
    total_targets = sum(scene["target_object_count"] for scene in per_scene.values())
    total_matches = sum(scene["matched_object_count"] for scene in per_scene.values())
    total_predictions = sum(scene["predicted_object_count"] for scene in per_scene.values())
    return {
        "scene_count": scene_count,
        "target_object_count": total_targets,
        "predicted_object_count": total_predictions,
        "matched_object_count": total_matches,
        "instance_recall": total_matches / total_targets if total_targets else 0.0,
        "mean_scene_instance_recall": (
            sum(scene["instance_recall"] for scene in per_scene.values()) / scene_count
            if scene_count
            else 0.0
        ),
        "mean_scene_category_recall": (
            sum(scene["category_recall"] for scene in per_scene.values()) / scene_count
            if scene_count
            else 0.0
        ),
        "mean_discovered_objects_per_step": (
            sum(scene["mean_discovered_objects_per_step"] for scene in per_scene.values())
            / scene_count
            if scene_count
            else 0.0
        ),
        "oracle_box_iou_mean": (
            sum(scene["oracle_box_iou_mean"] for scene in per_scene.values()) / scene_count
            if scene_count
            else 0.0
        ),
    }


def evaluate_predictions(targets: dict[str, Any], predictions: dict[str, Any]) -> dict[str, Any]:
    per_scene = {}
    for scene_id, scene_targets in targets["scenes"].items():
        if scene_id not in predictions["scenes"]:
            raise ValueError(f"Missing predictions for scene {scene_id}")
        per_scene[scene_id] = evaluate_scene(scene_targets, predictions["scenes"][scene_id])

    return {
        "version": 1,
        "task": targets["task"],
        "policy": predictions["policy"],
        "metrics": aggregate(per_scene),
        "per_scene": per_scene,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--policy", choices=("random", "forward", "all"), default="all")
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    output_dir = args.output_dir or Path(config["output_dir"])
    targets = load_json(args.targets or (output_dir / "targets.json"))

    if args.predictions and args.policy == "all":
        raise ValueError("--predictions can only be used with --policy random or --policy forward")

    policies = ("random", "forward") if args.policy == "all" else (args.policy,)
    for policy in policies:
        predictions_path = args.predictions or (output_dir / f"predictions_{policy}.json")
        predictions = load_json(predictions_path)
        metrics = evaluate_predictions(targets, predictions)
        output_path = output_dir / f"metrics_{policy}.json"
        write_json(output_path, metrics)
        aggregate_metrics = metrics["metrics"]
        print(
            f"{policy}: instance_recall={aggregate_metrics['instance_recall']:.4f}, "
            f"mean_scene_category_recall={aggregate_metrics['mean_scene_category_recall']:.4f}, "
            f"oracle_box_iou_mean={aggregate_metrics['oracle_box_iou_mean']:.4f}"
        )
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

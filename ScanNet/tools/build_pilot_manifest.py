#!/usr/bin/env python3
"""Build a ScanNet pilot manifest with a temporal trajectory graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


INTRINSIC_FILES = (
    "intrinsic_color.txt",
    "intrinsic_depth.txt",
    "extrinsic_color.txt",
    "extrinsic_depth.txt",
)


def read_scene_list(path: Path) -> list[str]:
    scenes = []
    for line in path.read_text().splitlines():
        scene = line.strip()
        if scene and not scene.startswith("#"):
            scenes.append(scene)
    return scenes


def relpath(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def frame_ids(directory: Path, suffix: str) -> set[int]:
    ids = set()
    for path in directory.glob(f"*{suffix}"):
        try:
            ids.add(int(path.stem))
        except ValueError:
            continue
    return ids


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def annotation_paths(scene_raw_dir: Path, scene_id: str, repo_root: Path) -> dict[str, str]:
    candidates = {
        "aggregation": scene_raw_dir / f"{scene_id}.aggregation.json",
        "segments": scene_raw_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json",
        "semantic_mesh": scene_raw_dir / f"{scene_id}_vh_clean_2.labels.ply",
        "mesh": scene_raw_dir / f"{scene_id}_vh_clean_2.ply",
        "metadata": scene_raw_dir / f"{scene_id}.txt",
    }
    return {
        name: relpath(path, repo_root)
        for name, path in candidates.items()
        if path.exists()
    }


def build_neighbors(frame_ids_sorted: list[int], index: int, neighbor_hops: int) -> list[int]:
    neighbors = []
    for hop in range(1, neighbor_hops + 1):
        previous_index = index - hop
        next_index = index + hop
        if previous_index >= 0:
            neighbors.append(frame_ids_sorted[previous_index])
        if next_index < len(frame_ids_sorted):
            neighbors.append(frame_ids_sorted[next_index])
    return sorted(neighbors)


def build_scene(
    scene_id: str,
    scannet_root: Path,
    repo_root: Path,
    neighbor_hops: int,
) -> dict[str, Any]:
    exported_dir = scannet_root / "exported" / scene_id
    raw_dir = scannet_root / "raw" / "scans" / scene_id
    color_dir = exported_dir / "color"
    depth_dir = exported_dir / "depth"
    pose_dir = exported_dir / "pose"
    intrinsic_dir = exported_dir / "intrinsic"

    for directory, label in (
        (color_dir, "color directory"),
        (depth_dir, "depth directory"),
        (pose_dir, "pose directory"),
        (intrinsic_dir, "intrinsic directory"),
    ):
        if not directory.exists():
            raise FileNotFoundError(f"Missing {label} for {scene_id}: {directory}")

    for filename in INTRINSIC_FILES:
        require_file(intrinsic_dir / filename, f"{scene_id} intrinsic file {filename}")

    color_ids = frame_ids(color_dir, ".jpg")
    depth_ids = frame_ids(depth_dir, ".png")
    pose_ids = frame_ids(pose_dir, ".txt")
    common_ids = sorted(color_ids & depth_ids & pose_ids)
    if not common_ids:
        raise ValueError(f"No aligned RGB/depth/pose frames found for {scene_id}")

    dropped = {
        "color_only": sorted(color_ids - depth_ids - pose_ids),
        "depth_only": sorted(depth_ids - color_ids - pose_ids),
        "pose_only": sorted(pose_ids - color_ids - depth_ids),
        "not_in_all_modalities": sorted((color_ids | depth_ids | pose_ids) - set(common_ids)),
    }

    states = []
    for index, frame_id in enumerate(common_ids):
        state_id = f"{scene_id}:{frame_id}"
        states.append(
            {
                "state_id": state_id,
                "frame_index": index,
                "frame_id": frame_id,
                "rgb": relpath(color_dir / f"{frame_id}.jpg", repo_root),
                "depth": relpath(depth_dir / f"{frame_id}.png", repo_root),
                "pose": relpath(pose_dir / f"{frame_id}.txt", repo_root),
                "intrinsic_color": relpath(intrinsic_dir / "intrinsic_color.txt", repo_root),
                "intrinsic_depth": relpath(intrinsic_dir / "intrinsic_depth.txt", repo_root),
                "extrinsic_color": relpath(intrinsic_dir / "extrinsic_color.txt", repo_root),
                "extrinsic_depth": relpath(intrinsic_dir / "extrinsic_depth.txt", repo_root),
                "neighbors": build_neighbors(common_ids, index, neighbor_hops),
            }
        )

    return {
        "scene_id": scene_id,
        "frame_count": len(states),
        "frame_ids": common_ids,
        "exported_dir": relpath(exported_dir, repo_root),
        "raw_dir": relpath(raw_dir, repo_root) if raw_dir.exists() else None,
        "annotations": annotation_paths(raw_dir, scene_id, repo_root) if raw_dir.exists() else {},
        "dropped_frames": dropped,
        "states": states,
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    scenes = args.scene or read_scene_list(args.scene_list)
    if not scenes:
        raise ValueError("No scenes provided")

    manifest = {
        "version": 1,
        "dataset": "ScanNet pilot",
        "path_mode": "repo_relative",
        "scannet_root": relpath(args.scannet_root, args.repo_root),
        "graph": {
            "type": "temporal_trajectory",
            "rule": "adjacent_plus_skip_edges",
            "neighbor_hops": args.neighbor_hops,
            "neighbor_values": "frame_id",
        },
        "scenes": {},
    }

    for scene_id in scenes:
        manifest["scenes"][scene_id] = build_scene(
            scene_id=scene_id,
            scannet_root=args.scannet_root,
            repo_root=args.repo_root,
            neighbor_hops=args.neighbor_hops,
        )

    return manifest


def print_summary(manifest: dict[str, Any]) -> None:
    total_states = 0
    for scene_id, scene in manifest["scenes"].items():
        total_states += scene["frame_count"]
        dropped_count = len(scene["dropped_frames"]["not_in_all_modalities"])
        print(
            f"{scene_id}: {scene['frame_count']} states, "
            f"{dropped_count} dropped modality-mismatch frames"
        )
    print(f"total: {len(manifest['scenes'])} scenes, {total_states} states")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a manifest and temporal graph for ScanNet pilot scenes."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene", action="append", default=[], help="Scene id; repeatable.")
    source.add_argument("--scene-list", type=Path, help="Text file with one scene id per line.")
    parser.add_argument("--scannet-root", type=Path, default=Path("ScanNet"))
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("ScanNet/processed/pilot_manifest.json"),
    )
    parser.add_argument(
        "--neighbor-hops",
        type=int,
        default=2,
        help="Connect each frame to +/- N temporal neighbors. Default: 2.",
    )
    args = parser.parse_args()

    if args.neighbor_hops < 1:
        parser.error("--neighbor-hops must be >= 1")

    args.scannet_root = args.scannet_root.resolve()
    args.repo_root = args.repo_root.resolve()
    args.output = args.output.resolve()
    if args.scene_list:
        args.scene_list = args.scene_list.resolve()
    return args


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print_summary(manifest)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()

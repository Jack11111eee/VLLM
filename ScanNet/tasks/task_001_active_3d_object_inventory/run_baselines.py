#!/usr/bin/env python3
"""Run oracle-visibility random/forward baselines for Task 001."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

from common import DEFAULT_CONFIG, as_object_key, load_config, load_json, sorted_int_keys, write_json


def state_lookup(scene: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(state["frame_id"]): state for state in scene["states"]}


def choose_random(
    current: dict[str, Any],
    visited: set[int],
    rng: random.Random,
) -> int | None:
    neighbors = list(current["neighbors"])
    if not neighbors:
        return None
    unvisited = [int(frame_id) for frame_id in neighbors if int(frame_id) not in visited]
    return rng.choice(unvisited or [int(frame_id) for frame_id in neighbors])


def choose_forward(current: dict[str, Any], visited: set[int]) -> int | None:
    current_frame_id = int(current["frame_id"])
    neighbors = [int(frame_id) for frame_id in current["neighbors"]]
    forward = sorted(frame_id for frame_id in neighbors if frame_id > current_frame_id)
    unvisited_forward = [frame_id for frame_id in forward if frame_id not in visited]
    if unvisited_forward:
        return unvisited_forward[0]
    if forward:
        return forward[0]

    unvisited = sorted(frame_id for frame_id in neighbors if frame_id not in visited)
    if unvisited:
        return unvisited[0]
    return sorted(neighbors)[0] if neighbors else None


def run_policy_on_scene(
    scene: dict[str, Any],
    scene_targets: dict[str, Any],
    policy: str,
    max_steps: int,
    start_index: int,
    rng: random.Random,
) -> dict[str, Any]:
    states = scene["states"]
    if not states:
        raise ValueError(f"Scene {scene['scene_id']} has no states")
    if start_index < 0 or start_index >= len(states):
        raise ValueError(
            f"start_index {start_index} outside scene {scene['scene_id']} "
            f"with {len(states)} states"
        )

    states_by_frame = state_lookup(scene)
    current = states[start_index]
    visited = {int(current["frame_id"])}
    trajectory = [int(current["frame_id"])]
    steps = []
    discovered: dict[str, dict[str, Any]] = {}

    def observe(frame_id: int, step_index: int) -> list[int]:
        newly_discovered = []
        visible_ids = scene_targets["frame_visibility"].get(str(frame_id), [])
        for object_id in visible_ids:
            key = as_object_key(object_id)
            target = scene_targets["objects"][key]
            if key not in discovered:
                discovered[key] = {
                    "object_id": int(object_id),
                    "category": target["category"],
                    "bbox_3d": target["bbox_3d"],
                    "first_seen_step": step_index,
                    "evidence_frames": [],
                }
                newly_discovered.append(int(object_id))
            if frame_id not in discovered[key]["evidence_frames"]:
                discovered[key]["evidence_frames"].append(frame_id)
        return newly_discovered

    observe(int(current["frame_id"]), 0)

    for step_index in range(max_steps):
        if policy == "random":
            next_frame_id = choose_random(current, visited, rng)
        elif policy == "forward":
            next_frame_id = choose_forward(current, visited)
        else:
            raise ValueError(f"Unsupported policy: {policy}")

        step = {
            "step": step_index,
            "current_frame_id": int(current["frame_id"]),
            "candidate_neighbors": [int(frame_id) for frame_id in current["neighbors"]],
            "next_frame_id": next_frame_id,
            "discovered_object_count": len(discovered),
            "newly_discovered_object_ids": [],
        }
        steps.append(step)

        if next_frame_id is None:
            break

        current = states_by_frame[int(next_frame_id)]
        visited.add(int(next_frame_id))
        trajectory.append(int(next_frame_id))
        newly_discovered = observe(int(next_frame_id), step_index + 1)
        step["newly_discovered_object_ids"] = newly_discovered
        step["discovered_object_count"] = len(discovered)

    for obj in discovered.values():
        obj["evidence_frames"] = sorted(obj["evidence_frames"])

    return {
        "scene_id": scene["scene_id"],
        "policy": policy,
        "start_frame_id": int(states[start_index]["frame_id"]),
        "visited_frames": trajectory,
        "unique_visited": len(visited),
        "coverage": len(visited) / scene["frame_count"],
        "objects": {key: discovered[key] for key in sorted_int_keys(discovered)},
        "steps": steps,
    }


def run_policy(
    manifest: dict[str, Any],
    targets: dict[str, Any],
    policy: str,
    max_steps: int,
    start_index: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    scenes = {}
    for scene_id, scene_targets in targets["scenes"].items():
        scenes[scene_id] = run_policy_on_scene(
            scene=manifest["scenes"][scene_id],
            scene_targets=scene_targets,
            policy=policy,
            max_steps=max_steps,
            start_index=start_index,
            rng=rng,
        )

    return {
        "version": 1,
        "task": targets["task"],
        "policy": policy,
        "perception": "gt_2d_visibility_oracle",
        "seed": seed,
        "max_steps": max_steps,
        "start_index": start_index,
        "summary": {
            "scene_count": len(scenes),
            "total_discovered_objects": sum(len(scene["objects"]) for scene in scenes.values()),
            "mean_coverage": (
                sum(scene["coverage"] for scene in scenes.values()) / len(scenes) if scenes else 0.0
            ),
        },
        "scenes": scenes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--targets", type=Path)
    parser.add_argument("--policy", choices=("random", "forward", "all"), default="all")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    manifest = load_json(Path(config["manifest"]))
    output_dir = args.output_dir or Path(config["output_dir"])
    targets_path = args.targets or (output_dir / "targets.json")
    targets = load_json(targets_path)
    max_steps = int(args.max_steps if args.max_steps is not None else config["max_steps"])
    start_index = int(args.start_index if args.start_index is not None else config["start_index"])
    seed = int(args.seed if args.seed is not None else config["seed"])

    policies = ("random", "forward") if args.policy == "all" else (args.policy,)
    for policy in policies:
        result = run_policy(
            manifest=manifest,
            targets=targets,
            policy=policy,
            max_steps=max_steps,
            start_index=start_index,
            seed=seed,
        )
        output_path = output_dir / f"predictions_{policy}.json"
        write_json(output_path, result)
        summary = result["summary"]
        print(
            f"{policy}: {summary['scene_count']} scenes, "
            f"{summary['total_discovered_objects']} discovered objects, "
            f"mean coverage={summary['mean_coverage']:.4f}"
        )
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

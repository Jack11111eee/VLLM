#!/usr/bin/env python3
"""Run minimal random/forward baselines on a ScanNet pilot manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def state_lookup(scene: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {state["frame_id"]: state for state in scene["states"]}


def choose_random(
    current: dict[str, Any],
    visited: set[int],
    rng: random.Random,
) -> int | None:
    neighbors = list(current["neighbors"])
    if not neighbors:
        return None
    unvisited = [frame_id for frame_id in neighbors if frame_id not in visited]
    return rng.choice(unvisited or neighbors)


def choose_forward(current: dict[str, Any], visited: set[int]) -> int | None:
    current_frame_id = current["frame_id"]
    neighbors = list(current["neighbors"])
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
    visited = {current["frame_id"]}
    trajectory = [current["frame_id"]]
    steps = []

    for step_index in range(max_steps):
        if policy == "random":
            next_frame_id = choose_random(current, visited, rng)
        elif policy == "forward":
            next_frame_id = choose_forward(current, visited)
        else:
            raise ValueError(f"Unsupported policy: {policy}")

        steps.append(
            {
                "step": step_index,
                "current_frame_id": current["frame_id"],
                "candidate_neighbors": current["neighbors"],
                "next_frame_id": next_frame_id,
            }
        )

        if next_frame_id is None:
            break

        current = states_by_frame[next_frame_id]
        visited.add(next_frame_id)
        trajectory.append(next_frame_id)

    return {
        "scene_id": scene["scene_id"],
        "policy": policy,
        "start_frame_id": states[start_index]["frame_id"],
        "trajectory": trajectory,
        "unique_visited": len(visited),
        "coverage": len(visited) / scene["frame_count"],
        "steps": steps,
    }


def run_policy(
    manifest: dict[str, Any],
    policy: str,
    max_steps: int,
    start_index: int,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    trajectories = {}
    for scene_id, scene in manifest["scenes"].items():
        trajectories[scene_id] = run_policy_on_scene(
            scene=scene,
            policy=policy,
            max_steps=max_steps,
            start_index=start_index,
            rng=rng,
        )

    coverages = [result["coverage"] for result in trajectories.values()]
    return {
        "version": 1,
        "policy": policy,
        "seed": seed,
        "max_steps": max_steps,
        "start_index": start_index,
        "summary": {
            "scene_count": len(trajectories),
            "mean_coverage": sum(coverages) / len(coverages) if coverages else 0.0,
            "total_unique_visited": sum(
                result["unique_visited"] for result in trajectories.values()
            ),
        },
        "trajectories": trajectories,
    }


def print_summary(result: dict[str, Any]) -> None:
    summary = result["summary"]
    print(
        f"{result['policy']}: {summary['scene_count']} scenes, "
        f"mean coverage={summary['mean_coverage']:.4f}, "
        f"total unique visited={summary['total_unique_visited']}"
    )
    for scene_id, trajectory in result["trajectories"].items():
        print(
            f"  {scene_id}: {trajectory['unique_visited']} unique, "
            f"coverage={trajectory['coverage']:.4f}, "
            f"path_len={len(trajectory['trajectory'])}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run random/forward pilot baselines over a ScanNet manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("ScanNet/processed/pilot_manifest.json"),
    )
    parser.add_argument(
        "--policy",
        choices=("random", "forward", "all"),
        default="all",
    )
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ScanNet/processed/baselines"),
    )
    args = parser.parse_args()

    if args.max_steps < 0:
        parser.error("--max-steps must be >= 0")
    args.manifest = args.manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    policies = ("random", "forward") if args.policy == "all" else (args.policy,)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for policy in policies:
        result = run_policy(
            manifest=manifest,
            policy=policy,
            max_steps=args.max_steps,
            start_index=args.start_index,
            seed=args.seed,
        )
        output_path = args.output_dir / f"{policy}_trajectories.json"
        output_path.write_text(json.dumps(result, indent=2) + "\n")
        print_summary(result)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()

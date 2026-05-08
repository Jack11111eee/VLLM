#!/usr/bin/env python3
"""Export ScanNet .sens files with Python 3.

The official ScanNet SensReader targets Python 2. This script keeps the same
core behavior for the pilot pipeline: export RGB, depth, camera poses, and
intrinsics from one .sens file or from a list of ScanNet scene ids.
"""

from __future__ import annotations

import argparse
import io
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image


COLOR_COMPRESSION_TYPES = {
    -1: "unknown",
    0: "raw",
    1: "png",
    2: "jpeg",
}

DEPTH_COMPRESSION_TYPES = {
    -1: "unknown",
    0: "raw_ushort",
    1: "zlib_ushort",
    2: "occi_ushort",
}


def read_exact(handle, num_bytes: int) -> bytes:
    data = handle.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError(f"Expected {num_bytes} bytes, got {len(data)}")
    return data


def read_matrix(handle) -> np.ndarray:
    return np.asarray(
        struct.unpack("f" * 16, read_exact(handle, 16 * 4)),
        dtype=np.float32,
    ).reshape(4, 4)


def write_matrix(path: Path, matrix: np.ndarray) -> None:
    np.savetxt(path, matrix, fmt="%.9f")


def is_valid_pose(matrix: np.ndarray) -> bool:
    return bool(np.isfinite(matrix).all())


class RGBDFrame:
    def __init__(self) -> None:
        self.camera_to_world: np.ndarray | None = None
        self.timestamp_color = 0
        self.timestamp_depth = 0
        self.color_data = b""
        self.depth_data = b""

    def load(self, handle) -> None:
        self.camera_to_world = read_matrix(handle)
        self.timestamp_color = struct.unpack("Q", read_exact(handle, 8))[0]
        self.timestamp_depth = struct.unpack("Q", read_exact(handle, 8))[0]

        color_size_bytes = struct.unpack("Q", read_exact(handle, 8))[0]
        depth_size_bytes = struct.unpack("Q", read_exact(handle, 8))[0]

        self.color_data = read_exact(handle, color_size_bytes)
        self.depth_data = read_exact(handle, depth_size_bytes)

    def color_image(self, compression_type: str) -> Image.Image:
        if compression_type == "raw":
            raise ValueError("Raw color frames are not supported by this exporter")
        if compression_type not in {"jpeg", "png"}:
            raise ValueError(f"Unsupported color compression: {compression_type}")

        image = Image.open(io.BytesIO(self.color_data))
        return image.convert("RGB")

    def depth_array(
        self,
        compression_type: str,
        depth_height: int,
        depth_width: int,
    ) -> np.ndarray:
        if compression_type == "zlib_ushort":
            depth_bytes = zlib.decompress(self.depth_data)
        elif compression_type == "raw_ushort":
            depth_bytes = self.depth_data
        else:
            raise ValueError(f"Unsupported depth compression: {compression_type}")

        expected_values = depth_height * depth_width
        depth = np.frombuffer(depth_bytes, dtype=np.uint16)
        if depth.size != expected_values:
            raise ValueError(
                f"Depth frame has {depth.size} values; expected {expected_values}"
            )
        return depth.reshape(depth_height, depth_width)


class SensorData:
    def __init__(self, filename: Path) -> None:
        self.filename = filename
        self.sensor_name = ""
        self.intrinsic_color: np.ndarray | None = None
        self.extrinsic_color: np.ndarray | None = None
        self.intrinsic_depth: np.ndarray | None = None
        self.extrinsic_depth: np.ndarray | None = None
        self.color_compression_type = "unknown"
        self.depth_compression_type = "unknown"
        self.color_width = 0
        self.color_height = 0
        self.depth_width = 0
        self.depth_height = 0
        self.depth_shift = 0.0
        self.frames: list[RGBDFrame] = []
        self.load(filename)

    def load(self, filename: Path) -> None:
        with filename.open("rb") as handle:
            version = struct.unpack("I", read_exact(handle, 4))[0]
            if version != 4:
                raise ValueError(f"Unsupported .sens version {version}; expected 4")

            sensor_name_len = struct.unpack("Q", read_exact(handle, 8))[0]
            self.sensor_name = read_exact(handle, sensor_name_len).decode(
                "utf-8",
                errors="replace",
            )

            self.intrinsic_color = read_matrix(handle)
            self.extrinsic_color = read_matrix(handle)
            self.intrinsic_depth = read_matrix(handle)
            self.extrinsic_depth = read_matrix(handle)

            color_compression_id = struct.unpack("i", read_exact(handle, 4))[0]
            depth_compression_id = struct.unpack("i", read_exact(handle, 4))[0]
            self.color_compression_type = COLOR_COMPRESSION_TYPES[color_compression_id]
            self.depth_compression_type = DEPTH_COMPRESSION_TYPES[depth_compression_id]

            self.color_width = struct.unpack("I", read_exact(handle, 4))[0]
            self.color_height = struct.unpack("I", read_exact(handle, 4))[0]
            self.depth_width = struct.unpack("I", read_exact(handle, 4))[0]
            self.depth_height = struct.unpack("I", read_exact(handle, 4))[0]
            self.depth_shift = struct.unpack("f", read_exact(handle, 4))[0]

            num_frames = struct.unpack("Q", read_exact(handle, 8))[0]
            for _ in range(num_frames):
                frame = RGBDFrame()
                frame.load(handle)
                self.frames.append(frame)

    def export(
        self,
        output_path: Path,
        frame_skip: int,
        start_frame: int,
        max_frames: int | None,
        skip_invalid_poses: bool,
        progress_every: int,
    ) -> None:
        if frame_skip < 1:
            raise ValueError("--frame-skip must be >= 1")
        if start_frame < 0:
            raise ValueError("--start-frame must be >= 0")

        color_dir = output_path / "color"
        depth_dir = output_path / "depth"
        pose_dir = output_path / "pose"
        intrinsic_dir = output_path / "intrinsic"

        for directory in (color_dir, depth_dir, pose_dir, intrinsic_dir):
            directory.mkdir(parents=True, exist_ok=True)

        write_matrix(intrinsic_dir / "intrinsic_color.txt", self.intrinsic_color)
        write_matrix(intrinsic_dir / "extrinsic_color.txt", self.extrinsic_color)
        write_matrix(intrinsic_dir / "intrinsic_depth.txt", self.intrinsic_depth)
        write_matrix(intrinsic_dir / "extrinsic_depth.txt", self.extrinsic_depth)

        indices = range(start_frame, len(self.frames), frame_skip)
        if max_frames is not None:
            indices = list(indices)[:max_frames]

        exported = 0
        skipped = 0
        for frame_id in indices:
            frame = self.frames[frame_id]
            if frame.camera_to_world is None:
                skipped += 1
                continue
            if skip_invalid_poses and not is_valid_pose(frame.camera_to_world):
                skipped += 1
                continue

            color = frame.color_image(self.color_compression_type)
            color.save(color_dir / f"{frame_id}.jpg", quality=95)

            depth = frame.depth_array(
                self.depth_compression_type,
                self.depth_height,
                self.depth_width,
            )
            Image.fromarray(depth).save(depth_dir / f"{frame_id}.png")

            write_matrix(pose_dir / f"{frame_id}.txt", frame.camera_to_world)
            exported += 1

        print(
            f"{self.filename.name}: exported {exported} frames to {output_path}"
            f" (skipped {skipped})"
        )


class StreamingSensorData:
    """Header-only .sens reader that exports selected frames while scanning."""

    def __init__(self, filename: Path) -> None:
        self.filename = filename
        self.sensor_name = ""
        self.intrinsic_color: np.ndarray | None = None
        self.extrinsic_color: np.ndarray | None = None
        self.intrinsic_depth: np.ndarray | None = None
        self.extrinsic_depth: np.ndarray | None = None
        self.color_compression_type = "unknown"
        self.depth_compression_type = "unknown"
        self.color_width = 0
        self.color_height = 0
        self.depth_width = 0
        self.depth_height = 0
        self.depth_shift = 0.0
        self.num_frames = 0

    def read_header(self, handle) -> None:
        version = struct.unpack("I", read_exact(handle, 4))[0]
        if version != 4:
            raise ValueError(f"Unsupported .sens version {version}; expected 4")

        sensor_name_len = struct.unpack("Q", read_exact(handle, 8))[0]
        self.sensor_name = read_exact(handle, sensor_name_len).decode(
            "utf-8",
            errors="replace",
        )

        self.intrinsic_color = read_matrix(handle)
        self.extrinsic_color = read_matrix(handle)
        self.intrinsic_depth = read_matrix(handle)
        self.extrinsic_depth = read_matrix(handle)

        color_compression_id = struct.unpack("i", read_exact(handle, 4))[0]
        depth_compression_id = struct.unpack("i", read_exact(handle, 4))[0]
        self.color_compression_type = COLOR_COMPRESSION_TYPES[color_compression_id]
        self.depth_compression_type = DEPTH_COMPRESSION_TYPES[depth_compression_id]

        self.color_width = struct.unpack("I", read_exact(handle, 4))[0]
        self.color_height = struct.unpack("I", read_exact(handle, 4))[0]
        self.depth_width = struct.unpack("I", read_exact(handle, 4))[0]
        self.depth_height = struct.unpack("I", read_exact(handle, 4))[0]
        self.depth_shift = struct.unpack("f", read_exact(handle, 4))[0]
        self.num_frames = struct.unpack("Q", read_exact(handle, 8))[0]

    def export(
        self,
        output_path: Path,
        frame_skip: int,
        start_frame: int,
        max_frames: int | None,
        skip_invalid_poses: bool,
        progress_every: int,
    ) -> None:
        if frame_skip < 1:
            raise ValueError("--frame-skip must be >= 1")
        if start_frame < 0:
            raise ValueError("--start-frame must be >= 0")

        color_dir = output_path / "color"
        depth_dir = output_path / "depth"
        pose_dir = output_path / "pose"
        intrinsic_dir = output_path / "intrinsic"

        for directory in (color_dir, depth_dir, pose_dir, intrinsic_dir):
            directory.mkdir(parents=True, exist_ok=True)

        exported = 0
        skipped = 0
        seen_selected = 0

        with self.filename.open("rb") as handle:
            self.read_header(handle)

            write_matrix(intrinsic_dir / "intrinsic_color.txt", self.intrinsic_color)
            write_matrix(intrinsic_dir / "extrinsic_color.txt", self.extrinsic_color)
            write_matrix(intrinsic_dir / "intrinsic_depth.txt", self.intrinsic_depth)
            write_matrix(intrinsic_dir / "extrinsic_depth.txt", self.extrinsic_depth)

            print(
                f"{self.filename.name}: scanning {self.num_frames} frames "
                f"(frame_skip={frame_skip}, max_frames={max_frames})",
                flush=True,
            )

            for frame_id in range(self.num_frames):
                frame = RGBDFrame()
                frame.load(handle)

                if progress_every > 0 and frame_id > 0 and frame_id % progress_every == 0:
                    print(
                        f"{self.filename.name}: scanned {frame_id}/{self.num_frames}, "
                        f"exported {exported}",
                        flush=True,
                    )

                if frame_id < start_frame or (frame_id - start_frame) % frame_skip != 0:
                    continue

                seen_selected += 1
                if max_frames is not None and seen_selected > max_frames:
                    break

                if frame.camera_to_world is None:
                    skipped += 1
                    continue
                if skip_invalid_poses and not is_valid_pose(frame.camera_to_world):
                    skipped += 1
                    continue

                color = frame.color_image(self.color_compression_type)
                color.save(color_dir / f"{frame_id}.jpg", quality=95)

                depth = frame.depth_array(
                    self.depth_compression_type,
                    self.depth_height,
                    self.depth_width,
                )
                Image.fromarray(depth).save(depth_dir / f"{frame_id}.png")

                write_matrix(pose_dir / f"{frame_id}.txt", frame.camera_to_world)
                exported += 1

        print(
            f"{self.filename.name}: exported {exported} frames to {output_path}"
            f" (skipped {skipped})",
            flush=True,
        )


def read_scene_list(path: Path) -> list[str]:
    scenes = []
    for line in path.read_text().splitlines():
        scene = line.strip()
        if scene and not scene.startswith("#"):
            scenes.append(scene)
    return scenes


def export_one_file(args: argparse.Namespace) -> None:
    sensor_data = StreamingSensorData(args.filename)
    sensor_data.export(
        output_path=args.output_path,
        frame_skip=args.frame_skip,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        skip_invalid_poses=args.skip_invalid_poses,
        progress_every=args.progress_every,
    )


def export_scene(args: argparse.Namespace, scene: str) -> None:
    sens_path = args.scannet_root / "raw" / "scans" / scene / f"{scene}.sens"
    output_path = args.scannet_root / "exported" / scene
    if not sens_path.exists():
        raise FileNotFoundError(f"Missing .sens file for {scene}: {sens_path}")

    sensor_data = StreamingSensorData(sens_path)
    sensor_data.export(
        output_path=output_path,
        frame_skip=args.frame_skip,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        skip_invalid_poses=args.skip_invalid_poses,
        progress_every=args.progress_every,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ScanNet .sens RGB, depth, pose, and intrinsics with Python 3."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--filename", type=Path, help="Path to one .sens file")
    source.add_argument(
        "--scene",
        action="append",
        default=[],
        help="Scene id to export, e.g. scene0568_00. Can be repeated.",
    )
    source.add_argument(
        "--scene-list",
        type=Path,
        help="Text file containing one ScanNet scene id per line.",
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        help="Output directory for --filename mode.",
    )
    parser.add_argument(
        "--scannet-root",
        type=Path,
        default=Path("ScanNet"),
        help="Root with raw/scans and exported directories for scene modes.",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=10,
        help="Export every Nth frame. Default: 10.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame id to consider. Default: 0.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum number of frames to export per scene.",
    )
    parser.add_argument(
        "--skip-invalid-poses",
        action="store_true",
        help="Skip frames whose camera_to_world matrix contains NaN or inf.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print scan progress every N frames. Use 0 to disable. Default: 500.",
    )

    args = parser.parse_args()

    if args.filename and args.output_path is None:
        parser.error("--output-path is required with --filename")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be >= 1")
    if args.progress_every < 0:
        parser.error("--progress-every must be >= 0")

    args.scannet_root = args.scannet_root.resolve()
    if args.filename:
        args.filename = args.filename.resolve()
        args.output_path = args.output_path.resolve()
    if args.scene_list:
        args.scene_list = args.scene_list.resolve()

    return args


def main() -> None:
    args = parse_args()

    if args.filename:
        export_one_file(args)
        return

    scenes = args.scene or read_scene_list(args.scene_list)
    if not scenes:
        raise ValueError("No scenes to export")

    for scene in scenes:
        export_scene(args, scene)


if __name__ == "__main__":
    main()

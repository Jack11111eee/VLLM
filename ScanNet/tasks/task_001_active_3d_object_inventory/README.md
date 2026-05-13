# Task 001: Active 3D Object Inventory

This task turns the ScanNet pilot trajectory graph into an active object inventory benchmark.
An agent visits exported RGB-D frames in a scene and returns the object instances it has discovered,
with each object reported as a scene-coordinate AABB 3D box.

The first version uses ScanNet GT 2D filtered instance annotations as an oracle perception backend:
when a frame is visited, all target objects with at least `min_visible_pixels` in that frame are marked
as visible. This keeps the first benchmark focused on the active exploration loop, output format, and
quantitative evaluation.

## Data Contract

Targets are written to `outputs/targets.json`:

```json
{
  "scenes": {
    "scene0568_00": {
      "objects": {
        "0": {
          "object_id": 0,
          "category": "chair",
          "bbox_3d": {
            "center": [0.0, 0.0, 0.0],
            "size": [1.0, 1.0, 1.0],
            "min": [-0.5, -0.5, -0.5],
            "max": [0.5, 0.5, 0.5]
          }
        }
      },
      "frame_visibility": {
        "0": [0, 5, 12]
      }
    }
  }
}
```

Predictions are written to `outputs/predictions_random.json` and
`outputs/predictions_forward.json`. Each scene contains visited frames, discovered objects, and
evidence frames. The oracle baseline copies AABB boxes from the matched GT instance once that instance
is visible.

Metrics are written to `outputs/metrics_random.json` and `outputs/metrics_forward.json`.
The primary metrics are instance recall, category recall, discovery curve, and mean discovered objects
per step. Box IoU is included only as an oracle sanity check because the baseline uses matched GT boxes.

## Usage

Build or refresh the pilot manifest if needed:

```bash
python3 ScanNet/tools/build_pilot_manifest.py \
  --scannet-root ScanNet \
  --scene-list ScanNet/splits/pilot_3.txt \
  --output ScanNet/processed/pilot_manifest.json \
  --neighbor-hops 2
```

Build targets:

```bash
python3 ScanNet/tasks/task_001_active_3d_object_inventory/build_targets.py
```

Run both oracle baselines:

```bash
python3 ScanNet/tasks/task_001_active_3d_object_inventory/run_baselines.py --policy all
```

Evaluate both baselines:

```bash
python3 ScanNet/tasks/task_001_active_3d_object_inventory/evaluate.py --policy all
```

Validate generated artifacts:

```bash
python3 ScanNet/tasks/task_001_active_3d_object_inventory/validate_outputs.py --policy all
```

All generated artifacts live under this task's `outputs/` directory and are ignored by git.

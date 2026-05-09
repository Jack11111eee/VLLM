# ScanNet Pilot Active Exploration

This pilot treats exported ScanNet keyframes as a discrete trajectory graph.
Each state is one RGB-D frame with a camera pose and intrinsics.

Build the manifest:

```bash
python3 ScanNet/tools/build_pilot_manifest.py \
  --scannet-root ScanNet \
  --scene-list ScanNet/splits/pilot_3.txt \
  --output ScanNet/processed/pilot_manifest.json \
  --neighbor-hops 2
```

Run the minimal baselines:

```bash
python3 ScanNet/tools/run_pilot_baselines.py \
  --manifest ScanNet/processed/pilot_manifest.json \
  --policy all \
  --max-steps 25 \
  --seed 42
```

The graph uses adjacent plus skip edges: each state connects to the previous and
next 1-hop and 2-hop keyframes when those states exist.

Outputs are written under `ScanNet/processed/`, which is intentionally ignored
by git because it is generated from downloaded ScanNet data.

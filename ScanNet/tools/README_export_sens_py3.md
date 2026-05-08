# Python 3 ScanNet `.sens` Exporter

Use `export_sens_py3.py` when the official ScanNet Python 2 `reader.py` is not
usable. It exports the pieces needed for the pilot active-exploration pipeline:

- `color/<frame_id>.jpg`
- `depth/<frame_id>.png`
- `pose/<frame_id>.txt`
- `intrinsic/*.txt`

Install the minimal dependencies:

```bash
python3 -m pip install numpy pillow
```

Export the three downloaded pilot scenes:

```bash
python3 ScanNet/tools/export_sens_py3.py \
  --scannet-root ScanNet \
  --scene-list ScanNet/splits/pilot_3.txt \
  --frame-skip 10 \
  --skip-invalid-poses
```

Run a quick smoke test on one scene:

```bash
python3 ScanNet/tools/export_sens_py3.py \
  --scannet-root ScanNet \
  --scene scene0568_00 \
  --frame-skip 1 \
  --max-frames 1 \
  --skip-invalid-poses
```

Export a single `.sens` file to a custom directory:

```bash
python3 ScanNet/tools/export_sens_py3.py \
  --filename ScanNet/raw/scans/scene0568_00/scene0568_00.sens \
  --output-path ScanNet/exported/scene0568_00 \
  --frame-skip 10 \
  --skip-invalid-poses
```

For the first pipeline pass, use `--frame-skip 10` or `--frame-skip 20`. Full
frame export is much larger and slower.

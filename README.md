# License plate recognition & traffic counting (YOLOv5-based)

Two scripts from a computer-vision internship: a Chinese licence-plate
detector/recogniser with colour classification, and a per-second people/vehicle
counter for traffic footage.

**These are not standalone packages.** Both extend the [YOLOv5](https://github.com/ultralytics/yolov5)
codebase and import from `models.*` / `utils.*`, so they must be run from inside a
YOLOv5 checkout. That is also why this repository is GPL-3.0 rather than MIT — see
[License](#license).

---

## What each file does

### `detect_plate.py`

A fork of YOLOv5's `detect.py` specialised for plates:

| Piece | Function |
|---|---|
| detection + landmarks | `load_model`, `scale_coords_landmarks` — the model returns bounding boxes **and four corner points** |
| rectification | `order_points`, `four_point_transform` — corners are ordered TL/TR/BR/BL, then a perspective warp crops a square plate image |
| recognition | `get_plate_rec_landmark`, `detect_Recognition_plate` — runs the recognition head on the warped crop, returns plate text and colour |
| filtering | `is_valid_plate` — regex over province abbreviations and the 7-character format, to reject garbage reads |
| rendering | `draw_result` — box, corner points, text with a CJK-capable overlay |
| side effects | `save_to_db` — optional de-duplicating insert into a MySQL table |
| throughput | `get_second` — per-video timing loop, prints parameter count and FPS |

The interesting part is not the detection — YOLOv5 does that — it is the
**detect → 4-point warp → recognise** chain. Plates in real footage are tilted and
perspective-distorted; recognition accuracy is dominated by the quality of the
rectified crop, not by the detector.

### `traffic_count.py`

A fork of YOLOv5's inference loop that aggregates detections into a per-second
count: people and vehicles tallied separately, counter reset every second, series
collected in memory, then plotted with matplotlib (blue = people, red = vehicles).
Optional CSV dump of raw detections.

### `numpy_mlp_from_scratch.py`

Small and unrelated, kept because it is useful: a 3-layer MLP forward pass written
with NumPy only — no framework — with sigmoid/softmax/identity activations and a
hand-initialised weight dictionary. Good for explaining what `nn.Linear` actually does.

## Running

```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
cp /path/to/this/repo/*.py .

# plate recognition (weights are NOT included, see below)
python detect_plate.py --detect_model weights/plate_detect.pt \
                       --rec_model    weights/plate_rec_color.pth \
                       --image_path   samples/            # or --video clip.mp4

# traffic counting
python traffic_count.py --weights yolov5s.pt --source clip.mp4 --save-csv
```

The database insert is off unless you call `save_to_db` yourself; credentials are read
from the environment:

```bash
export PLATE_DB_HOST=localhost
export PLATE_DB_USER=root
export PLATE_DB_PASSWORD=...     # required
export PLATE_DB_NAME=car_number
```

## Weights are not included

`plate_detect.pt` and `plate_rec_color.pth` were trained during the internship and are
not redistributed here. `traffic_count.py` only needs the public `yolov5s.pt`, which
YOLOv5 downloads automatically.

Without the plate weights the script is readable but not runnable. This is a known
limitation, stated up front rather than hidden.

## What is mine and what is not

The inference scaffolding (`run`, dataloaders, NMS, annotation) is Ultralytics' code,
lightly modified. The plate-specific logic — corner ordering, perspective
rectification, landmark handling, the recognition call sequence, the format validator,
the rendering, the per-second aggregation — is mine.

## License

**GPL-3.0.** These files are derivative works of YOLOv5, which is distributed under
GPL-3.0; a derivative must keep the same licence. Any redistribution of this code, in
whole or in part, has to remain GPL-3.0 and keep the source available.
Full text: https://www.gnu.org/licenses/gpl-3.0.txt

`numpy_mlp_from_scratch.py` is entirely my own and could be MIT, but it ships inside a
GPL-3.0 repository, so treat the repository as a whole under GPL-3.0.

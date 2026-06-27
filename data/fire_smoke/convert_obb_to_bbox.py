"""
One-off conversion: this Roboflow export stored labels as 4-corner polygons
(class x1 y1 x2 y2 x3 y3 x4 y4) even though every box is axis-aligned (verified: 0 of
6240 boxes have any rotation) -- ultralytics RT-DETR detection training needs standard
YOLO bbox format (class x_center y_center width height). Converts in place.

Also fixes data.yaml: the export shipped with class names literally "0" and "1" instead
of semantic names -- confirmed by inspecting sample images per class (0 = fire, 1 = smoke).

Run: `python data/fire_smoke/convert_obb_to_bbox.py`
"""
from __future__ import annotations

from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent / "raw"


def convert_file(path: Path):
    lines_out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        cls = parts[0]
        coords = list(map(float, parts[1:9]))
        xs = coords[0::2]
        ys = coords[1::2]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        xc = (x_min + x_max) / 2
        yc = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min
        lines_out.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))


def main():
    label_files = list(RAW_DIR.glob("*/labels/*.txt"))
    for f in label_files:
        convert_file(f)
    print(f"Converted {len(label_files)} label files to standard YOLO bbox format.")

    yaml_path = RAW_DIR / "data.yaml"
    yaml_text = yaml_path.read_text()
    yaml_text = yaml_text.replace("0: 0\n  1: 1", "0: fire\n  1: smoke")
    yaml_path.write_text(yaml_text)
    print("Updated data.yaml class names: 0=fire, 1=smoke")


if __name__ == "__main__":
    main()

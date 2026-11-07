"""Export a trained detector to a deployment format."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a YOLO model")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--format", default="onnx")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    model = YOLO(str(args.weights))
    path = model.export(format=args.format, imgsz=args.imgsz)
    print(path)


if __name__ == "__main__":
    main()

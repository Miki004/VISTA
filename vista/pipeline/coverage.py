"""
Usage
-----
python measure_coverage.py \
    --video data/VISTA/test/seq01.mp4 \
    --config config/qwenyolo/cfg.yaml \
    --yolo-weights runs/yoloe11m_lp/weights/best.pt \
    --yolo-model YOLOE --category-map yoloe
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from PIL import Image

from vista.pipeline.mypipeline import (
    build_mypipeline_from_config,
    COCO_CATEGORY_MAP,
    YOLOE_CATEGORY_MAP,
)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
CATEGORY_MAPS = {"yoloe": YOLOE_CATEGORY_MAP, "coco": COCO_CATEGORY_MAP}


def iter_frames(path: Path):
    if path.suffix.lower() in VID_EXTS:
        import cv2
        import numpy as np
        cap = cv2.VideoCapture(str(path))
        i = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            yield i, Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            i += 1
        cap.release()
    else:  # directory of ordered frame images
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMG_EXTS)
        for i, f in enumerate(files):
            with Image.open(f) as im:
                yield i, im.convert("RGB")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--config", required=True, type=str,
                    help="qwen yaml config (the captioner must be on).")
    ap.add_argument("--yolo-weights", required=True, type=str)
    ap.add_argument("--yolo-model", default="YOLOE", choices=["YOLO", "YOLOE"])
    ap.add_argument("--category-map", default="yoloe", choices=list(CATEGORY_MAPS))
    ap.add_argument("--yolo-conf", default=None, type=float)
    ap.add_argument("--min-hits", default=3, type=int)
    args = ap.parse_args()

    pipe = build_mypipeline_from_config(
        config_path=args.config,
        yolo_weights=args.yolo_weights,
        yolo_model=args.yolo_model,
        use_qwen=True,                       # captioner ON: coverage needs it
        yolo_conf=args.yolo_conf,
        min_hits=args.min_hits,
        category_map=CATEGORY_MAPS[args.category_map],
    )
    pipe.reset()

    seen: set[int] = set()                   # every emitted track id
    has_caption: dict[int, bool] = defaultdict(bool)

    for idx, frame in iter_frames(args.video):
        res = pipe.forward(frame, idx)
        for det in res.detections:
            seen.add(det.track_id)
            if det.caption and str(det.caption).strip():
                has_caption[det.track_id] = True

    total = len(seen)
    captioned = sum(1 for t in seen if has_caption[t])
    empty = total - captioned
    cov = 100.0 * captioned / total if total else 0.0

    print(f"emitted tracks       : {total}")
    print(f"tracks with a caption: {captioned}")
    print(f"tracks without one   : {empty}")
    print(f"caption coverage     : {cov:.1f}%")


if __name__ == "__main__":
    main()
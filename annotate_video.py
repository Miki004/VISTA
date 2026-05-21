"""Render annotated output videos (tracking + captioning) from a VISTA pipeline.

Runs any ``VistaPipeline`` over a video and writes a new video with, for every
frame, the per-object bounding box, persistent track id, category and caption
drawn on top. Box colour is keyed on the track id so identities stay visually
stable across frames.

Use from a notebook with an already-built pipeline:

    from vista.annotate_video import annotate_video
    annotate_video(pipeline, "input.mp4", "annotated.mp4")

Or from the CLI (builds a QwenYolo pipeline from a config):

    python -m vista.annotate_video --video input.mp4 --out annotated.mp4 \
        --config config/qwenyolo/cfg.yaml --yolo-weights yolov8s.pt
    # tracking only, no captioner:
    python -m vista.annotate_video --video input.mp4 --out annotated.mp4 --no-qwen
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from vista.pipeline.base import FrameResult, VistaPipeline

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _color_for(track_id: int | None) -> tuple[int, int, int]:
    """Deterministic, well-separated BGR colour for a track id."""
    if track_id is None:
        return (180, 180, 180)
    h = (int(track_id) * 47) % 180  # spread hues
    hsv = np.uint8([[[h, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def _wrap(text: str, max_chars: int, max_lines: int) -> list[str]:
    """Greedy word-wrap, truncated to ``max_lines`` with an ellipsis."""
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        cand = f"{cur} {w}".strip()
        if len(cand) <= max_chars:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w
        if len(lines) == max_lines:
            break
    if cur and len(lines) < max_lines:
        lines.append(cur)
    if len(lines) == max_lines and (len(words) > sum(len(l.split()) for l in lines)):
        lines[-1] = lines[-1][: max_chars - 1] + "…"
    return lines


def _draw_text_block(
    frame: np.ndarray,
    lines: list[str],
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """Draw text lines with a filled background box for readability."""
    x, y = org
    pad = 3
    line_h = int(20 * scale / 0.5)
    widths = [cv2.getTextSize(ln, _FONT, scale, thickness)[0][0] for ln in lines] or [0]
    box_w = max(widths) + 2 * pad
    box_h = line_h * len(lines) + 2 * pad
    y_top = max(0, y - box_h)
    cv2.rectangle(frame, (x, y_top), (x + box_w, y_top + box_h), color, -1)
    for i, ln in enumerate(lines):
        ty = y_top + pad + line_h * (i + 1) - 4
        cv2.putText(frame, ln, (x + pad, ty), _FONT, scale, (0, 0, 0), thickness, cv2.LINE_AA)


def draw_frame(frame_bgr: np.ndarray, result: FrameResult) -> np.ndarray:
    """Draw all detections of one FrameResult onto a BGR frame (in place)."""
    h, w = frame_bgr.shape[:2]
    for det in result.detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        color = _color_for(det.track_id)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        tid = "" if det.track_id is None else f"#{det.track_id} "
        header = f"{tid}{det.category} {det.confidence:.2f}"
        lines = [header]
        if det.caption:
            box_chars = max(8, (x2 - x1) // 8)
            lines += _wrap(det.caption, max_chars=box_chars, max_lines=2)
        _draw_text_block(frame_bgr, lines, (x1, y1), color)

    cv2.putText(frame_bgr, f"frame {result.frame_idx}", (10, h - 12),
                _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return frame_bgr


def annotate_video(
    pipeline: VistaPipeline,
    video_path: str | Path,
    out_path: str | Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    fourcc: str = "mp4v",
    progress_every: int = 30,
) -> Path:
    """Run ``pipeline`` over a video and write an annotated copy.

    Args:
        pipeline:      Any VistaPipeline (e.g. QwenYoloPipeline).
        video_path:    Source video file.
        out_path:      Destination video file (.mp4 recommended).
        start_frame:   First frame to process (inclusive).
        end_frame:     Last frame to process (exclusive); None = until the end.
        fourcc:        VideoWriter codec, e.g. "mp4v" or "avc1".
        progress_every: Log progress every N frames (0 = silent).

    Returns:
        The output path.
    """
    video_path, out_path = Path(video_path), Path(out_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*fourcc),
                             fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open VideoWriter for: {out_path} (codec {fourcc})")

    pipeline.reset()
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frame_idx = start_frame
    try:
        while True:
            if end_frame is not None and frame_idx >= end_frame:
                break
            ret, bgr = cap.read()
            if not ret:
                break
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            result = pipeline.forward(pil, frame_idx)
            writer.write(draw_frame(bgr, result))
            if progress_every and frame_idx % progress_every == 0:
                print(f"[annotate] frame {frame_idx}: {len(result.detections)} detections",
                      flush=True)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()

    print(f"[annotate] wrote {out_path} ({frame_idx - start_frame} frames)", flush=True)
    return out_path


def _build_qwenyolo_from_config(
    config_path: str | None, yolo_weights: str, caption_stride: int, use_qwen: bool
) -> VistaPipeline:
    """Construct a QwenYoloPipeline for the CLI."""
    import yaml
    from ultralytics import YOLO
    from vista.pipeline.qwen_yolo import QwenYoloPipeline

    cfg: dict[str, Any] = {}
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

    yolo = YOLO(yolo_weights)
    qwen = None
    if use_qwen:
        from vista.qwen import get_model
        qwen = get_model(cfg)
    return QwenYoloPipeline(yolo_model=yolo, qwen_model=qwen, caption_stride=caption_stride)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--out", required=True, help="Output (annotated) video path")
    parser.add_argument("--config", default=None, help="QwenYolo config yaml (for the captioner)")
    parser.add_argument("--yolo-weights", default="yolov8s.pt", help="YOLO weights")
    parser.add_argument("--caption-stride", type=int, default=30, help="Run the VLM every N frames")
    parser.add_argument("--no-qwen", action="store_true", help="Tracking only, skip the captioner")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--fourcc", default="mp4v", help="VideoWriter codec")
    args = parser.parse_args()

    pipeline = _build_qwenyolo_from_config(
        config_path=args.config,
        yolo_weights=args.yolo_weights,
        caption_stride=args.caption_stride,
        use_qwen=not args.no_qwen,
    )
    annotate_video(
        pipeline, args.video, args.out,
        start_frame=args.start_frame, end_frame=args.end_frame, fourcc=args.fourcc,
    )


if __name__ == "__main__":
    main()
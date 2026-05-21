from __future__ import annotations

from collections import deque
from typing import Any

import json
from PIL import Image
from json_repair import repair_json
from ultralytics import YOLO

from vista.pipeline.base import Detection, FrameResult, VistaPipeline
from vista.utils_fun import image_to_base64, resize_image, log, IGNORE_CATEGORIES, get_emergency_level


def _iou(a, b) -> float:
    if not (a and b and len(a) >= 4 and len(b) >= 4):
        return 0.0
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    aA = (a[2] - a[0]) * (a[3] - a[1])
    aB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aA + aB - inter)


class QwenYoloPipeline(VistaPipeline):
    """YOLO tracker + Qwen-VL captioner fused via IoU matching.

    Detection and tracking are handled by YOLO (``ultralytics``); captions are
    produced by the Qwen-VL model every ``caption_stride`` frames and propagated
    to all matching tracks via IoU overlap.

    Args:
        yolo_model:      Loaded ``ultralytics.YOLO`` instance.
        qwen_model:      Any object with a ``generate(frame, history)`` method
                        that returns a JSON string of detections with ``bbox_2d``
                        and ``label`` fields (see ``vista.qwen``).
        caption_stride:  Run the VLM every this many frames. Captions are
                        propagated between calls.
        iou_threshold:   Minimum IoU to match a Qwen detection to a YOLO track.
        history_len:     Number of past (frame, response) pairs fed to the VLM
                        as conversational context.
        yolo_conf:       YOLO detection confidence threshold.
    """

    def __init__(
        self,
        yolo_model: YOLO,
        qwen_model: Any,
        caption_stride: int = 30,
        iou_threshold: float = 0.3,
        history_len: int = 1,
        yolo_conf: float | None = None,
    ) -> None:
        self.yolo = yolo_model
        self.qwen = qwen_model
        self.caption_stride = caption_stride
        self.iou_threshold = iou_threshold
        self.yolo_conf = yolo_conf

        # per-video state
        self._track_db: dict[int, dict] = {}
        self._history: deque = deque(maxlen=history_len)

    # ── VistaPipeline interface ───────────────────────────────────────────────

    def reset(self) -> None:
        self._track_db.clear()
        self._history.clear()

    def forward(self, frame: Image.Image, frame_idx: int) -> FrameResult:
        import cv2, numpy as np

        bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)

        # ── 1. YOLO tracking ─────────────────────────────────────────────────
        results = self.yolo.track(
            bgr, persist=True, verbose=False,
            conf=self.yolo_conf,
        )[0]

        active: dict[int, dict] = {}
        if results.boxes.id is not None:
            for box, tid, cls in zip(
                results.boxes.xyxy, results.boxes.id, results.boxes.cls
            ):
                tid = int(tid.item())
                cat = results.names.get(int(cls.item()), "unknown")
                if cat in IGNORE_CATEGORIES:
                    continue
                active[tid] = {
                    "bbox":     box.cpu().numpy().tolist(),
                    "category": self._track_db.get(tid, {}).get("category", cat),
                    "caption":  self._track_db.get(tid, {}).get("caption"),
                    "conf":     float(results.boxes.conf[list(results.boxes.id).index(tid)].item()),
                }

        # remove stale tracks
        for tid in set(self._track_db) - set(active):
            del self._track_db[tid]

        # ── 2. Qwen captioning (every caption_stride frames) ─────────────────
        if frame_idx % self.caption_stride == 0 and self.qwen is not None:
            log(f"Qwen inference at frame {frame_idx}")
            raw = self.qwen.generate(frame=frame, history=list(self._history))
            self._history.append((frame, raw))

            try:
                parsed = json.loads(repair_json(raw))
            except Exception:
                parsed = []
                if isinstance(parsed, dict):
                    parsed = parsed.get("detections", [])
                if not isinstance(parsed, list):
                    parsed = []

            # match each Qwen detection to the best-overlap YOLO track
            for det in parsed:
                if not isinstance(det, dict):
                    continue
                qb    = det.get("bbox_2d", [])
                label = det.get("label", "unknown")
                if not isinstance(qb, (list, tuple)) or len(qb) != 4:
                    continue
                best_tid, best_iou = None, 0.0
                for tid, tr in active.items():
                    s = _iou(qb, tr["bbox"])
                    if s > best_iou:
                        best_iou, best_tid = s, tid

                if best_tid is not None and best_iou >= self.iou_threshold:
                    active[best_tid]["caption"]  = label
                    active[best_tid]["category"] = _resolve_category(label)

        # ── 3. Merge into track DB and build FrameResult ──────────────────────
        for tid, tr in active.items():
            self._track_db[tid] = tr

        detections = [
            Detection(
                bbox=tuple(tr["bbox"]),
                category=tr["category"],
                confidence=tr.get("conf", 1.0),
                track_id=tid,
                caption=tr.get("caption"),
            )
            for tid, tr in self._track_db.items()
        ]

        return FrameResult(detections=detections, frame_idx=frame_idx)


# ── helpers ───────────────────────────────────────────────────────────────────

def _resolve_category(label: str) -> str:
    """Map a free-form VLM label back to a canonical category."""
    label_l = label.lower()
    if "emergency" in label_l or "ambulance" in label_l or "fire" in label_l:
        return "emergency_vehicle"
    if "person" in label_l or "injured" in label_l or "pedestrian" in label_l:
        return "person"
    return "car"

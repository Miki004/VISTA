"""MyPipeline: YOLO detection+tracking with VLM *crop captioning*.

Approach #1 to fix the low caption coverage of QwenYoloPipeline: instead of
asking the VLM to re-detect and then trying to match its boxes to YOLO tracks
via IoU, we hand the VLM the YOLO crops directly and ask it for one short
label per crop, in the same order. Every active track at the captioning frame
gets a label — no IoU step, no dropped assignments.

Captions are propagated across frames via an internal track DB (the last
non-empty label per track survives until the track ends), so downstream
consumers (e.g. submission.csv aggregation) see good coverage.

The pipeline is decoupled from any specific VLM: pass any callable
``captioner(crops: list[PIL.Image]) -> list[str]``. ``QwenCropCaptioner`` is
provided as a drop-in adapter for the existing ``vista.qwen.QwenVLHF`` model
loaded via ``vista.qwen.get_model``.
"""

from __future__ import annotations

from typing import Any, Callable

from PIL import Image
from ultralytics import YOLO

from vista.BindingOutput import LABEL_TO_CATEGORY
from vista.pipeline.base import Detection, FrameResult, VistaPipeline
from vista.utils_fun import IGNORE_CATEGORIES, log


CropCaptioner = Callable[[list[Image.Image]], list[str]]


# ── pipeline ──────────────────────────────────────────────────────────────────

class MyPipeline(VistaPipeline):
    """YOLO tracker + per-crop VLM captioner (no VLM re-detection)."""

    def __init__(
        self,
        yolo_model: YOLO | YOLOE,
        captioner: CropCaptioner | None,
        category_map: dict[str, str],
        caption_stride: int = 15,
        yolo_conf: float | None = None,
        max_crops_per_call: int = 12,
        crop_padding: int = 8,
    ) -> None:
        self.yolo = yolo_model
        self.captioner = captioner
        self.category_map = category_map
        self.caption_stride = caption_stride
        self.yolo_conf = yolo_conf
        self.max_crops_per_call = max_crops_per_call
        self.crop_padding = crop_padding
        self._track_db: dict[int, dict] = {}

    def reset(self) -> None:
        self._track_db.clear()

    def forward(self, frame: Image.Image, frame_idx: int) -> FrameResult:
        import cv2, numpy as np

        bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        H, W = bgr.shape[:2]

        # 1) YOLO detect + track
        results = self.yolo.track(
            bgr, persist=True, verbose=False, conf=self.yolo_conf,
        )[0]

        active: dict[int, dict] = {}
        if results.boxes.id is not None:
            for i, (box, tid, cls) in enumerate(
                zip(results.boxes.xyxy, results.boxes.id, results.boxes.cls)
            ):
                tid = int(tid.item())
                yolo_cat = results.names.get(int(cls.item()), "unknown")
                mapped_cat = self.category_map.get(yolo_cat)
                if mapped_cat is None: #class outside of the taxonomy is discarded
                    continue
                x1, y1, x2, y2 = box.cpu().numpy().tolist()
                prev = self._track_db.get(tid, {})
                active[tid] = {
                    "bbox": [x1, y1, x2, y2],
                    "yolo_category": yolo_cat,
                    "category": prev.get("category", mapped_cat),
                    "caption":  prev.get("caption"),
                    "conf":     float(results.boxes.conf[i].item()),
                }

        # drop stale tracks
        for tid in set(self._track_db) - set(active):
            del self._track_db[tid]

        # 2) VLM crop captioning every caption_stride frames
        if (frame_idx % self.caption_stride == 0) and active and self.captioner is not None:
            tids = list(active)
            crops = [_crop(frame, active[t]["bbox"], W, H, self.crop_padding) for t in tids]
            labels = self._caption_in_batches(crops, frame_idx)
            for tid, label in zip(tids, labels):
                if not label:
                    continue
                active[tid]["caption"] = label
                active[tid]["category"] = LABEL_TO_CATEGORY.get(
                    label.lower(), active[tid]["category"]
                )

        # 3) merge and emit
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

    def _caption_in_batches(self, crops: list[Image.Image], frame_idx: int) -> list[str]:
        out: list[str] = []
        for start in range(0, len(crops), self.max_crops_per_call):
            chunk = crops[start: start + self.max_crops_per_call]
            try:
                labels = list(self.captioner(chunk))  # type: ignore[misc]
            except Exception as e:
                log(f"[mypipeline] captioner error at frame {frame_idx}: {e}")
                labels = []
            labels = (labels + [""] * len(chunk))[: len(chunk)]
            out.extend(labels)
        return out


def _crop(frame: Image.Image, bbox, W: int, H: int, pad: int) -> Image.Image:
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(W, int(x2) + pad)
    y2 = min(H, int(y2) + pad)
    if x2 <= x1 or y2 <= y1:
        return frame.crop((0, 0, 1, 1))
    return frame.crop((x1, y1, x2, y2))


# ── Qwen adapter ──────────────────────────────────────────────────────────────

class QwenCropCaptioner:
    """Caption a batch of crops in a single Qwen call.

    Reuses the ``processor`` and ``model`` exposed by ``vista.qwen.QwenVLHF``
    (the object returned by ``vista.qwen.get_model``). Builds a single user
    turn with all crops and asks the model to emit a JSON list with one label
    per crop, in order, drawn from the ``vista.BindingOutput.LABELS``
    taxonomy. Returns a list of strings of length ``len(crops)`` (empty
    strings for missing/garbled entries).
    """

    def __init__(self, qwen_model, allowed_labels: list[str] | None = None,
                system_prompt: str | None = None):
        self.qwen = qwen_model
        from vista.BindingOutput import LABELS
        self.allowed_labels = allowed_labels or list(LABELS.__args__)
        self.system_prompt = system_prompt or (
            "You are an aerial-incident scene labeller. Each user message contains "
            "several image crops, presented in order. For each crop output exactly "
            "one short status label, chosen from the allowed taxonomy."
        )

    def __call__(self, crops: list[Image.Image]) -> list[str]:
        import json
        import torch
        from json_repair import repair_json
        from qwen_vl_utils import process_vision_info

        if not crops:
            return []

        labels_csv = ", ".join(f'"{lbl}"' for lbl in self.allowed_labels)
        user_text = (
            f"There are {len(crops)} crops, in order. For each crop, return one "
            f"short status label from this taxonomy: [{labels_csv}]. "
            f'Respond with JSON of the form '
            f'{{"captions":[{{"label":"..."}}, ...]}} with exactly {len(crops)} '
            "entries in the same order as the crops."
        )

        content: list[dict] = [{"type": "image", "image": c} for c in crops]
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system",
             "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": content},
        ]

        processor = self.qwen.processor
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_kwargs=True, image_patch_size=16,
        )
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt", **video_kwargs,
        )
        target_device = next(self.qwen.model.parameters()).device
        inputs = {k: v.to(target_device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}

        sampling = self.qwen.sampling_params \
            if isinstance(self.qwen.sampling_params, dict) else {}
        with torch.no_grad():
            out_ids = self.qwen.model.generate(**inputs, **sampling)
        gen_ids = [out[len(inp):] for inp, out in zip(inputs["input_ids"], out_ids)]
        raw = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        try:
            parsed = json.loads(repair_json(raw))
        except Exception:
            parsed = {}
        if isinstance(parsed, dict):
            entries = parsed.get("captions", [])
        elif isinstance(parsed, list):
            entries = parsed
        else:
            entries = []

        labels: list[str] = []
        for e in entries[: len(crops)]:
            if isinstance(e, dict):
                labels.append(str(e.get("label", "")).strip())
            elif isinstance(e, str):
                labels.append(e.strip())
            else:
                labels.append("")
        labels += [""] * (len(crops) - len(labels))
        return labels
# ── builder ───────────────────────────────────────────────────────────────────

def build_mypipeline_from_config(
    config_path: str | None,
    yolo_weights: str = "yolov8s.pt",
    caption_stride: int = 15,
    use_qwen: bool = True,
    yolo_conf: float | None = None,
    max_crops_per_call: int = 12,
    crop_padding: int = 8,
) -> MyPipeline:
    """Construct a MyPipeline from a yaml config (same shape as cfg used by
    ``vista.qwen.get_model``).

    Args:
        config_path:        Path to a yaml config (e.g. config/qwenyolo/cfg7.yaml);
                            None / "" = no config (only valid with use_qwen=False).
        yolo_weights:       Ultralytics YOLO weights file.
        caption_stride:     Run the VLM on the crops every N frames.
        use_qwen:           If False, skip the captioner (tracking-only mode).
        yolo_conf:          YOLO detection confidence threshold.
        max_crops_per_call: Max crops sent to the VLM in a single call.
        crop_padding:       Pixels of padding around each crop.
    """
    import yaml

    cfg: dict[str, Any] = {}
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    
    qcfg = cfg.get("qwen", {})
    yolo = YOLO(yolo_weights)
    captioner: CropCaptioner | None = None
    if use_qwen:
        from vista.qwen import get_model
        qwen = get_model(cfg)
        captioner = QwenCropCaptioner(qwen, system_prompt=qcfg.get("system_prompt"))

    return MyPipeline(
        yolo_model=yolo,
        captioner=captioner,
        caption_stride=qcfg.get("every_n_frames", caption_stride),
        yolo_conf=yolo_conf,
        max_crops_per_call=max_crops_per_call,
        crop_padding=crop_padding,
    )
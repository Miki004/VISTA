"""Detection-level evaluation for the qwen_yolo baseline on a YOLO dataset.

Estimates how well the *detection* part of the pipeline performs by running the
YOLO detector over an annotated, same-domain image dataset (e.g. VistaCrash,
classes: crashed_car, person, car) and computing COCO-style mAP via
``torchmetrics`` (mAP@0.50, mAP@0.50:0.95 and per-class AP).

The baseline detector is a generic COCO model (e.g. ``yolov8s.pt``), so COCO
categories are remapped into the VistaCrash class space before scoring. A COCO
detector cannot tell a crashed car from an intact one, so ``crashed_car`` and
``car`` are merged into a single ``car`` class by default.

Track/caption submission scoring is intentionally NOT handled here: VistaCrash
is an image dataset without track-level ground truth.

Install once:  pip install torchmetrics pycocotools

CLI:
    python -m vista.eval_detection --data data/VistaCrash/data.yaml \
        --split val --weights yolo12m.pt --conf 0.25
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import yaml

# COCO category name -> VistaCrash class name. Anything not listed is ignored.
DEFAULT_PRED_TO_VISTA: dict[str, str] = {
    "person": "person",
    "car": "car",
    "truck": "car",
    "bus": "car",
}


def _load_names(data_yaml: Path) -> list[str]:
    """Return dataset class names in dataset-index order."""
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):
        return [names[i] for i in sorted(names)]
    return list(names)


def _resolve_split_images(data_yaml: Path, split: str) -> list[Path]:
    """Resolve the list of image paths for ``split`` from a YOLO data.yaml."""
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()

    rel = cfg[split]
    rel = rel[0] if isinstance(rel, list) else rel
    target = (root / rel).resolve()

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if target.is_dir():
        return sorted(p for p in target.rglob("*") if p.suffix.lower() in exts)
    # target is a .txt file listing image paths (also a valid YOLO layout)
    with target.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return [(root / ln).resolve() if not Path(ln).is_absolute() else Path(ln) for ln in lines]


def _label_path_for(image_path: Path) -> Path:
    """Map an image path to its YOLO label .txt path (images/ -> labels/)."""
    parts = list(image_path.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def _build_class_space(
    names: list[str], merge_car: bool
) -> tuple[list[str], dict[str, int], Callable[[int], int | None]]:
    """Build the evaluation class space and a GT-index remapper.

    Returns (eval_names, name_to_idx, gt_index_map) where gt_index_map turns a
    dataset GT class index into an evaluation index (or None to drop it).
    """
    if merge_car and "crashed_car" in names and "car" in names:
        eval_names = [n for n in names if n != "crashed_car"]
    else:
        eval_names = list(names)
    name_to_idx = {n: i for i, n in enumerate(eval_names)}

    def gt_index_map(orig_idx: int) -> int | None:
        name = names[orig_idx]
        if merge_car and name == "crashed_car":
            name = "car"
        return name_to_idx.get(name)

    return eval_names, name_to_idx, gt_index_map


def _load_gt(label_path: Path, img_w: int, img_h: int, gt_index_map):
    """Load YOLO GT for one image as absolute-pixel xyxy boxes + eval labels."""
    boxes: list[list[float]] = []
    labels: list[int] = []
    if not label_path.exists():
        return boxes, labels
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            eval_cls = gt_index_map(cls)
            if eval_cls is None:
                continue
            cx, cy, w, h = (float(v) for v in parts[1:5])
            x1 = (cx - w / 2.0) * img_w
            y1 = (cy - h / 2.0) * img_h
            x2 = (cx + w / 2.0) * img_w
            y2 = (cy + h / 2.0) * img_h
            boxes.append([x1, y1, x2, y2])
            labels.append(eval_cls)
    return boxes, labels


def evaluate_detector(
    weights: str,
    data_yaml: str | Path,
    split: str = "val",
    conf: float = 0.25,
    device: str | None = None,
    merge_car: bool = True,
    pred_to_vista: dict[str, str] | None = None,
    imgsz: int = 640,
):
    """Run a YOLO detector over a split and return torchmetrics mAP results.

    Args:
        weights:       Path/name of the ultralytics YOLO weights (e.g. yolov8s.pt).
        data_yaml:     Path to the YOLO dataset data.yaml (e.g. VistaCrash).
        split:         Split key in data.yaml ("train"/"val"/"test").
        conf:          Detector confidence threshold.
        device:        torch device string ("cuda:0", "cpu"); None = auto.
        merge_car:     Merge crashed_car + car into a single "car" class.
        pred_to_vista: COCO-name -> VistaCrash-name mapping (defaults provided).
        imgsz:         Inference image size.

    Returns:
        (metrics_dict, eval_names): metrics_dict is the dict returned by
        torchmetrics MeanAveragePrecision.compute() with tensors converted to
        floats/lists; eval_names is the ordered list of evaluated class names.
    """
    import torch
    from PIL import Image
    from ultralytics import YOLO
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    pred_to_vista = pred_to_vista or DEFAULT_PRED_TO_VISTA
    data_yaml = Path(data_yaml)
    names = _load_names(data_yaml)
    eval_names, name_to_idx, gt_index_map = _build_class_space(names, merge_car)

    images = _resolve_split_images(data_yaml, split)
    if not images:
        raise RuntimeError(f"No images found for split '{split}' in {data_yaml}")

    model = YOLO(weights)
    metric = MeanAveragePrecision(box_format="xyxy", class_metrics=True)

    n_gt = n_pred = 0
    for img_path in images:
        with Image.open(img_path) as im:
            img_w, img_h = im.size

        gt_boxes, gt_labels = _load_gt(
            _label_path_for(img_path), img_w, img_h, gt_index_map
        )
        n_gt += len(gt_boxes)

        result = model.predict(
            source=str(img_path), conf=conf, imgsz=imgsz,
            device=device, verbose=False,
        )[0]

        p_boxes: list[list[float]] = []
        p_scores: list[float] = []
        p_labels: list[int] = []
        coco_names = result.names
        for box, score, cls in zip(
            result.boxes.xyxy, result.boxes.conf, result.boxes.cls
        ):
            coco_name = coco_names.get(int(cls.item()), "")
            vista_name = pred_to_vista.get(coco_name)
            eval_cls = name_to_idx.get(vista_name) if vista_name else None
            if eval_cls is None:
                continue
            p_boxes.append(box.cpu().tolist())
            p_scores.append(float(score.item()))
            p_labels.append(eval_cls)
        n_pred += len(p_boxes)

        metric.update(
            preds=[{
                "boxes": torch.tensor(p_boxes, dtype=torch.float32).reshape(-1, 4),
                "scores": torch.tensor(p_scores, dtype=torch.float32),
                "labels": torch.tensor(p_labels, dtype=torch.long),
            }],
            target=[{
                "boxes": torch.tensor(gt_boxes, dtype=torch.float32).reshape(-1, 4),
                "labels": torch.tensor(gt_labels, dtype=torch.long),
            }],
        )

    raw = metric.compute()
    metrics = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in raw.items()}
    metrics["num_images"] = len(images)
    metrics["num_gt_boxes"] = n_gt
    metrics["num_pred_boxes"] = n_pred
    return metrics, eval_names


def print_summary(metrics: dict, eval_names: list[str]) -> None:
    """Pretty-print the torchmetrics mAP results."""
    print(
        f"Images: {metrics.get('num_images')} | "
        f"GT boxes: {metrics.get('num_gt_boxes')} | "
        f"Pred boxes: {metrics.get('num_pred_boxes')}"
    )
    print(f"  mAP@0.50      = {float(metrics.get('map_50', -1)):.4f}")
    print(f"  mAP@0.50:0.95 = {float(metrics.get('map', -1)):.4f}")

    per_class = metrics.get("map_per_class")
    classes = metrics.get("classes")
    if per_class is not None and classes is not None:
        per_class = per_class if isinstance(per_class, list) else [per_class]
        classes = classes if isinstance(classes, list) else [classes]
        print("\nPer-class mAP@0.50:0.95:")
        for cid, ap in zip(classes, per_class):
            name = eval_names[cid] if 0 <= cid < len(eval_names) else str(cid)
            print(f"  {name:<12} = {float(ap):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    parser.add_argument("--split", default="val", help="Split key (train/val/test)")
    parser.add_argument("--weights", default="yolov8s.pt", help="YOLO weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--device", default=None, help="torch device (e.g. cuda:0)")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--no-merge-car", action="store_true",
                        help="Keep crashed_car and car as separate classes")
    args = parser.parse_args()

    metrics, eval_names = evaluate_detector(
        weights=args.weights,
        data_yaml=args.data,
        split=args.split,
        conf=args.conf,
        device=args.device,
        merge_car=not args.no_merge_car,
        imgsz=args.imgsz,
    )
    print_summary(metrics, eval_names)


if __name__ == "__main__":
    main()

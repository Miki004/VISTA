"""BERTScore-F1 evaluation of the qwen_yolo baseline on CapERA Traffic Collision.

CapERA provides 5 reference captions per video and no bounding-box / track
ground truth, so this script computes only caption quality via BERTScore-F1
(MOTA / IDF1 / mAP are not computable on CapERA — they need per-frame box and
track labels).

Each video has multiple references; BERTScore is computed against all of them
and the best (max) F1 per video is kept, then averaged — the standard
multi-reference protocol.

Predicted captions come from one of:
  * a submission.csv (video_id, track_id, frame_start, frame_end, caption):
    the per-track captions of a video are aggregated into one candidate string;
  * a JSON mapping {video_id: "predicted caption"}.

Install once:  pip install bert-score

CLI:
    python -m vista.eval_era --references CapERA_DATASET_train.json \
        --submission workspace/vista_output/submission.csv
    python -m vista.eval_era --references CapERA_DATASET_train.json \
        --predictions preds.json --out era_bertscore.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path


def _norm_id(video_id: str) -> str:
    """Normalize a video id for matching (drop extension, lower-case)."""
    return Path(str(video_id)).stem.lower()


def load_capera_references(
    path: str | Path, prefix: str | None = "TrafficCollision"
) -> dict[str, list[str]]:
    """Load CapERA references as {normalized_video_id: [captions...]}.

    Handles a top-level list of records, a dict wrapping such a list, or a dict
    keyed by video id. Each record exposes ``annotation.English_caption``.
    Filtered to ``prefix`` (e.g. only TrafficCollision_*) when given.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    records: list[dict] = []
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        list_vals = [v for v in data.values() if isinstance(v, list)]
        if list_vals:
            records = max(list_vals, key=len)
        else:  # dict keyed by video id
            records = [{"video_id": k, **v} for k, v in data.items()
                       if isinstance(v, dict)]

    refs: dict[str, list[str]] = {}
    for rec in records:
        vid = rec.get("video_id")
        if vid is None:
            continue
        if prefix and not str(vid).startswith(prefix):
            continue
        caps = (rec.get("annotation") or {}).get("English_caption") or []
        caps = [c.strip() for c in caps if isinstance(c, str) and c.strip()]
        if caps:
            refs[_norm_id(vid)] = caps
    return refs


def load_candidates_from_submission(path: str | Path) -> dict[str, str]:
    """Aggregate per-track captions of each video into one candidate string."""
    by_video: "OrderedDict[str, list[str]]" = OrderedDict()
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            vid = _norm_id(row.get("video_id", ""))
            cap = (row.get("caption") or "").strip()
            if not vid:
                continue
            by_video.setdefault(vid, [])
            if cap and cap not in by_video[vid]:
                by_video[vid].append(cap)
    return {vid: ", ".join(caps) for vid, caps in by_video.items()}


def load_candidates_from_json(path: str | Path) -> dict[str, str]:
    """Load {video_id: caption} predictions, normalizing the ids."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {_norm_id(k): str(v) for k, v in data.items()}


def compute_bertscore(
    candidates: dict[str, str],
    references: dict[str, list[str]],
    lang: str = "en",
    model_type: str | None = None,
    rescale_with_baseline: bool = True,
) -> dict:
    """Compute multi-reference BERTScore-F1 over the matched videos.

    Returns a dict with the corpus mean P/R/F1 and per-video F1 scores. Only
    videos present in both candidates and references are scored.
    """
    from bert_score import score as bert_score

    matched = [vid for vid in references if vid in candidates]
    if not matched:
        raise RuntimeError(
            "No overlapping video ids between predictions and references. "
            "Check that submission video ids match CapERA (e.g. TrafficCollision_001)."
        )

    cands = [candidates[vid] for vid in matched]
    refs = [references[vid] for vid in matched]  # list[list[str]] -> multi-ref

    P, R, F1 = bert_score(
        cands, refs, lang=lang, model_type=model_type,
        rescale_with_baseline=rescale_with_baseline, verbose=False,
    )

    per_video = {vid: float(f) for vid, f in zip(matched, F1.tolist())}
    return {
        "bertscore_f1": float(F1.mean()),
        "bertscore_precision": float(P.mean()),
        "bertscore_recall": float(R.mean()),
        "num_scored": len(matched),
        "num_references": len(references),
        "num_candidates": len(candidates),
        "num_unmatched_references": len([v for v in references if v not in candidates]),
        "per_video_f1": per_video,
    }


def print_summary(result: dict) -> None:
    print(
        f"Scored {result['num_scored']}/{result['num_references']} videos "
        f"(predictions available: {result['num_candidates']}, "
        f"unmatched refs: {result['num_unmatched_references']})"
    )
    print(f"  BERTScore-F1        = {result['bertscore_f1']:.4f}")
    print(f"  BERTScore-Precision = {result['bertscore_precision']:.4f}")
    print(f"  BERTScore-Recall    = {result['bertscore_recall']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--references", required=True, help="CapERA *.json with English_caption")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--submission", help="submission.csv with a 'caption' column")
    src.add_argument("--predictions", help="JSON mapping {video_id: caption}")
    parser.add_argument("--prefix", default="TrafficCollision",
                        help="Filter references by video-id prefix ('' = no filter)")
    parser.add_argument("--lang", default="en", help="BERTScore language")
    parser.add_argument("--model-type", default=None,
                        help="Override BERTScore model (e.g. microsoft/deberta-xlarge-mnli)")
    parser.add_argument("--no-rescale", action="store_true",
                        help="Disable rescale_with_baseline")
    parser.add_argument("--out", default=None, help="Write full results JSON here")
    args = parser.parse_args()

    references = load_capera_references(args.references, prefix=args.prefix or None)
    if args.submission:
        candidates = load_candidates_from_submission(args.submission)
    else:
        candidates = load_candidates_from_json(args.predictions)

    result = compute_bertscore(
        candidates, references,
        lang=args.lang, model_type=args.model_type,
        rescale_with_baseline=not args.no_rescale,
    )
    print_summary(result)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()

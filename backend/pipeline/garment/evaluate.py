"""평가. case별·난이도별로 따로 집계한다.

정답 파일 (데이터셋 뿌리의 `labels.json`)
  {
    "case2/hard/couple.jpg": {
      "case": "case2", "difficulty": "hard",
      "items": ["상의", "하의", "신발"],
      "masks": {"상의": "case2/hard/masks/couple_top.png"}   # 선택
    }
  }

눈 검수 (`<out>/review.csv`) — 없으면 검수 항목은 비워 둔다
  file,item,pass(1/0),note
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.garment.store import PhotoStore
from pipeline.garment.types import PhotoResult

TARGET_REVIEW_PASS = 0.90
TARGET_STAGE1_4_S = 3.0


def load_labels(path: Path) -> dict[str, dict]:
    return json.loads(Path(path).read_text(encoding="utf-8")) if Path(path).exists() else {}


def mask_iou(pred: np.ndarray, truth: np.ndarray) -> float:
    if pred.shape != truth.shape:
        truth = np.asarray(Image.fromarray(truth.astype(np.uint8) * 255)
                           .resize((pred.shape[1], pred.shape[0]), Image.NEAREST)) > 127
    union = np.logical_or(pred, truth).sum()
    return float(np.logical_and(pred, truth).sum() / union) if union else 1.0


def boundary_f1(pred: np.ndarray, truth: np.ndarray, tol: int = 3) -> float:
    """경계 정확도. 경계 픽셀이 서로 tol 안에 있는 비율(F1)."""
    from pipeline.garment.refine import dilate

    def edge(m):
        return m ^ _erode(m)

    pe, te = edge(pred), edge(truth)
    if not pe.any() and not te.any():
        return 1.0
    if not pe.any() or not te.any():
        return 0.0
    precision = float((pe & dilate(te, tol)).sum() / pe.sum())
    recall = float((te & dilate(pe, tol)).sum() / te.sum())
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _erode(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage

        return ndimage.binary_erosion(mask)
    except ImportError:
        out = mask.copy()
        out[1:] &= mask[:-1]
        out[:-1] &= mask[1:]
        out[:, 1:] &= mask[:, :-1]
        out[:, :-1] &= mask[:, 1:]
        return out


def count_scores(predicted: list[str], expected: list[str]) -> dict:
    """개수·카테고리 정확도와 누락·중복 비율."""
    pc, ec = Counter(predicted), Counter(expected)
    matched = sum((pc & ec).values())
    n_exp = max(len(expected), 1)
    return {
        "count_exact": float(len(predicted) == len(expected)),
        "count_abs_error": abs(len(predicted) - len(expected)),
        "category_accuracy": matched / n_exp,
        "missing_ratio": max(0, len(expected) - len(predicted)) / n_exp,
        "extra_ratio": max(0, len(predicted) - len(expected)) / n_exp,
    }


def load_review(path: Path) -> dict[tuple[str, str], float]:
    if not Path(path).exists():
        return {}
    out = {}
    with Path(path).open(encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            verdict = (row.get("pass(1/0)") or "").strip()
            if verdict in ("0", "1"):
                out[(row["file"], row.get("item", ""))] = float(verdict)
    return out


def evaluate(out_root: Path, dataset_root: Path | None = None,
             labels_path: Path | None = None) -> dict:
    out_root = Path(out_root)
    labels = load_labels(labels_path or (Path(dataset_root or ".") / "labels.json"))
    review = load_review(out_root / "review.csv")

    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    rows = []
    for state in sorted(out_root.glob("*/state.json")):
        result = PhotoResult.from_dict(json.loads(state.read_text(encoding="utf-8")))
        src = Path(result.source)
        key = _label_key(src, dataset_root, labels)
        truth = labels.get(key, {})
        case = truth.get("case") or result.scene.case
        difficulty = truth.get("difficulty", "unknown")

        row = {"file": src.name, "case": case, "difficulty": difficulty,
               "predicted_items": len(result.items),
               "predicted_categories": [i.category for i in result.items],
               "scene_case": result.scene.case,
               "scene_correct": float(result.scene.case == case) if truth.get("case") else None,
               "timings": result.timings,
               "stage1_4_s": round(sum(v for k, v in result.timings.items()
                                       if k in ("scene", "segment", "refine", "export")), 3),
               "gpu_peak_gb": result.gpu_peak_gb}
        if truth.get("items"):
            row.update(count_scores([i.category for i in result.items], truth["items"]))
        ious = _mask_scores(result, truth, out_root, src)
        if ious:
            row["mean_iou"] = round(float(np.mean([v[0] for v in ious])), 4)
            row["mean_boundary_f1"] = round(float(np.mean([v[1] for v in ious])), 4)
        verdicts = [review[(src.name, i.id)] for i in result.items if (src.name, i.id) in review]
        if verdicts:
            row["review_pass"] = sum(verdicts) / len(verdicts)
        rows.append(row)
        buckets[(case, difficulty)].append(row)

    report = {
        "images": len(rows),
        "targets": {"stage1_4_seconds": TARGET_STAGE1_4_S, "review_pass": TARGET_REVIEW_PASS},
        "by_case_difficulty": {f"{c}/{d}": _aggregate(v) for (c, d), v in sorted(buckets.items())},
        "overall": _aggregate(rows),
        "per_image": rows,
    }
    (out_root / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_root / "evaluation.md").write_text(markdown(report), encoding="utf-8")
    return report


def _label_key(src: Path, dataset_root: Path | None, labels: dict) -> str:
    if dataset_root:
        try:
            return str(src.resolve().relative_to(Path(dataset_root).resolve()))
        except ValueError:
            pass
    return next((k for k in labels if Path(k).name == src.name), src.name)


def _mask_scores(result: PhotoResult, truth: dict, out_root: Path, src: Path):
    store = PhotoStore(out_root, src)
    scores = []
    for category, rel in (truth.get("masks") or {}).items():
        gt_path = Path(rel)
        if not gt_path.is_absolute():
            gt_path = src.parent / rel if (src.parent / rel).exists() else Path(rel)
        if not gt_path.exists():
            continue
        item = next((i for i in result.items if i.category == category), None)
        pred_path = store.abs(item.refined_mask_path or item.mask_path) if item else None
        if not pred_path or not Path(pred_path).exists():
            scores.append((0.0, 0.0))
            continue
        pred = np.asarray(Image.open(pred_path).convert("L")) > 127
        gt = np.asarray(Image.open(gt_path).convert("L")) > 127
        scores.append((mask_iou(pred, gt), boundary_f1(pred, gt)))
    return scores


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}

    def mean(field):
        vals = [r[field] for r in rows if r.get(field) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    return {
        "images": len(rows),
        "scene_case_accuracy": mean("scene_correct"),
        "count_exact": mean("count_exact"),
        "count_abs_error": mean("count_abs_error"),
        "category_accuracy": mean("category_accuracy"),
        "missing_ratio": mean("missing_ratio"),
        "extra_ratio": mean("extra_ratio"),
        "mean_iou": mean("mean_iou"),
        "mean_boundary_f1": mean("mean_boundary_f1"),
        "review_pass": mean("review_pass"),
        "stage1_4_s": mean("stage1_4_s"),
    }


def markdown(report: dict) -> str:
    cols = ["images", "scene_case_accuracy", "count_exact", "category_accuracy",
            "missing_ratio", "extra_ratio", "mean_iou", "mean_boundary_f1",
            "review_pass", "stage1_4_s"]
    head = "| case/난이도 | " + " | ".join(cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = ["# 파이프라인 평가", "", head, sep]
    for key, agg in list(report["by_case_difficulty"].items()) + [("전체", report["overall"])]:
        lines.append("| " + key + " | " + " | ".join(
            "-" if agg.get(c) is None else str(agg.get(c)) for c in cols) + " |")
    lines += ["", f"- 1~4단계 목표: 사진당 {TARGET_STAGE1_4_S}초 이내",
              f"- 눈 검수 목표 합격률: {TARGET_REVIEW_PASS:.0%}"]
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="파이프라인 평가")
    p.add_argument("--out", required=True, help="파이프라인 결과 폴더")
    p.add_argument("--dataset", help="테스트 사진 뿌리 폴더")
    p.add_argument("--labels", help="정답 JSON (기본: <dataset>/labels.json)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    rep = evaluate(Path(args.out), Path(args.dataset) if args.dataset else None,
                   Path(args.labels) if args.labels else None)
    print(markdown(rep))

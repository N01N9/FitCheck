"""T01. 옷 누끼(배경 제거) 테스트.

옷 사진 폴더를 받아 배경을 지운 PNG를 만들고, 속도·메모리·품질을 리포트로 남긴다.

모델 래퍼는 `pipeline.garment.refine` 에 있다(파이프라인 3단계와 같은 코드를 쓴다).
  birefnet : BiRefNet (MIT). 실제 후보. GPU(DGX Spark)에서 돌린다.
  ben2     : BEN2 Base (MIT). 비교 후보.
  border   : 가장자리 색을 배경으로 보고 지우는 고전 방식. 비교 기준선이자,
             GPU·모델 없이 파이프라인을 점검하는 용도.

실행 예 (backend 폴더에서)
  python -m phase0.t01_bg_removal --images data/garments --out results/t01
  python -m phase0.t01_bg_removal --images data/garments --model border --out results/t01_border

정답 마스크가 있으면(--masks, 같은 파일 이름의 흑백 PNG) IoU를 계산한다.
없으면 contact_sheet.jpg를 눈으로 보고 review.csv에 합격/불합격을 적는다.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.garment.refine import BiRefNetRemover, BorderRemover, build_remover  # noqa: F401
from phase0.common import (
    GpuPeak,
    Stopwatch,
    Timings,
    contact_sheet,
    environment,
    list_images,
    load_rgb,
    write_report,
)

# 합격 기준 (기획서 Phase 0)
TARGET_P50_S = 1.0  # 한 장당 1초 이내
TARGET_REVIEW_PASS = 0.90  # 눈 검수 합격률 90% 이상
TARGET_IOU = 0.95  # 정답 마스크가 있을 때


def iou_and_mae(pred: Image.Image, truth: Image.Image) -> tuple[float, float]:
    p = np.asarray(pred.convert("L"), dtype=np.float32) / 255
    t = np.asarray(truth.convert("L").resize(pred.size), dtype=np.float32) / 255
    pb, tb = p > 0.5, t > 0.5
    union = np.logical_or(pb, tb).sum()
    iou = float(np.logical_and(pb, tb).sum() / union) if union else 1.0
    return iou, float(np.abs(p - t).mean())


def build_model(args) -> object:
    """파이프라인 3단계(refine)와 같은 모델 래퍼를 쓴다."""
    if args.model == "border":
        return build_remover("border")
    if args.model == "ben2":
        return build_remover("ben2")
    return build_remover("birefnet", variant=args.model_id, resolution=args.resolution)


def run(args) -> dict:
    images = list_images(Path(args.images))
    out = Path(args.out)
    (out / "cutouts").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    gpu = GpuPeak()
    with Stopwatch() as load:
        model = build_model(args)
    gpu.reset()

    timings = Timings(warmup=min(args.warmup, max(len(images) - 1, 0)))
    per_image, sheet = [], []
    masks_dir = Path(args.masks) if args.masks else None
    for path in images:
        img = load_rgb(path)
        with Stopwatch() as sw:
            mask = model.predict_mask(img)
            gpu.sync()
        timings.add(sw.seconds)

        cutout = img.convert("RGBA")
        cutout.putalpha(mask)
        cutout.save(out / "cutouts" / f"{path.stem}.png")
        mask.save(out / "masks" / f"{path.stem}.png")
        sheet.append((path.name, img, cutout))

        row = {"file": path.name, "seconds": round(sw.seconds, 3), "size": list(img.size)}
        truth = masks_dir / f"{path.stem}.png" if masks_dir else None
        if truth and truth.exists():
            row["iou"], row["mae"] = (round(v, 4) for v in iou_and_mae(mask, Image.open(truth)))
        per_image.append(row)

    contact_sheet(sheet, out / "contact_sheet.jpg")
    review = out / "review.csv"
    if not review.exists():  # 이미 적어 둔 검수 결과는 덮어쓰지 않는다
        with review.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "pass(1/0)", "note"])
            for row in per_image:
                w.writerow([row["file"], "", ""])

    speed = timings.summary()
    ious = [r["iou"] for r in per_image if "iou" in r]
    data = {
        "test": "T01 옷 누끼",
        "model": model.name,
        "model_id": getattr(model, "model_id", None),
        "environment": environment(),
        "load_seconds": round(load.seconds, 2),
        "images": len(images),
        "speed": speed,
        "peak_gpu_gb": gpu.peak_gb(),
        "mean_iou": round(sum(ious) / len(ious), 4) if ious else None,
        "targets": {"p50_s": TARGET_P50_S, "review_pass": TARGET_REVIEW_PASS, "iou": TARGET_IOU},
        "per_image": per_image,
    }
    write_report(out, "T01 옷 누끼 결과", data, markdown(data))
    return data


def markdown(d: dict) -> str:
    s = d["speed"]
    lines = [
        f"- 모델: `{d['model']}` {d['model_id'] or ''}",
        f"- 환경: {d['environment']}",
        f"- 모델 로딩: {d['load_seconds']}초",
        f"- 이미지: {d['images']}장",
        f"- 속도: 중앙값 {s.get('p50_s')}초 / 95% {s.get('p95_s')}초 (목표 중앙값 ≤ {TARGET_P50_S}초)",
        f"- 최대 GPU 메모리: {d['peak_gpu_gb']} GB",
    ]
    if d["mean_iou"] is not None:
        lines.append(f"- 평균 IoU: {d['mean_iou']} (목표 ≥ {TARGET_IOU})")
    lines += [
        "",
        "## 눈 검수",
        f"`contact_sheet.jpg`를 보고 `review.csv`에 장마다 1(합격)/0(불합격)을 적어 주세요. 목표 합격률 {int(TARGET_REVIEW_PASS * 100)}%.",
        "불합격 기준 예: 옷 일부가 잘림, 배경·옷걸이·손이 남음, 끈·레이스 같은 얇은 부분이 사라짐.",
    ]
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="T01 옷 누끼 테스트")
    p.add_argument("--images", required=True, help="옷 사진 폴더")
    p.add_argument("--out", required=True, help="결과 폴더")
    p.add_argument("--model", choices=["birefnet", "ben2", "border"], default="birefnet")
    p.add_argument("--model-id", default="ZhengPeng7/BiRefNet")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--masks", help="정답 마스크 폴더(선택)")
    p.add_argument("--warmup", type=int, default=2, help="속도 통계에서 뺄 앞쪽 장수")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args)
    print((Path(args.out) / "report.md").read_text(encoding="utf-8"))

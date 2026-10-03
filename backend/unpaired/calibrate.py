"""검사기(gates) 보정: 정답을 아는 이너 교체 쌍으로 양성·음성 대조군을 만들어 거부율과 AUROC 를 잰다.

승인된 swap 쌍(입력 X 의 이너 = 은행 상품 P)마다 "결과물" 후보를 만든다.
  pos        P 자체 (맞는 결과)
  other      같은 종류의 다른 은행 상품 (그럴듯하지만 틀린 옷)
  outer      X 에서 겉옷만 잘라 흰 배경에 놓은 것 (실험 0 의 "재킷 반환" 실패)
  strip      X 에서 보이는 이너 띠만 잘라 놓은 것 (펴지 않고 잘라 붙인 결과)
  merge      P 위에 X 의 겉옷 색을 덧칠한 것 (실험 0 의 "블레이저+티 합본")
  dark       P 의 채도·밝기를 낮춘 것 (남색 → 검정)
  hue        P 의 색상을 돌린 것
보고서 기준: 음성 대조군 90% 이상 거부, 가시율 구간별 AUROC 0.75 이상인 구간에서만 검사기를 보상으로 쓴다.

사용
  python -m unpaired.calibrate --swap data/unpaired/engine/swap_pilot --out results/unpaired/calibration.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from unpaired import gates
from unpaired.bank import load_usable
from unpaired.engine import prepare, product_pixels, stem_of
from unpaired.layered import Annotations
from unpaired.score import masked_crop

NEGATIVES = ("other", "outer", "strip", "merge", "dark", "hue")


def auroc(pos: list[float], neg: list[float]) -> float:
    """점수가 작을수록 좋다고 볼 때, 양성이 음성보다 작을 확률(동점 0.5)."""
    if not pos or not neg:
        return float("nan")
    p, n = np.asarray(pos)[:, None], np.asarray(neg)[None, :]
    return float(((p < n).sum() + 0.5 * (p == n).sum()) / (p.size * n.size))


def as_result(img: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """흰 배경 결과물과 전경 마스크(흰색이 아닌 곳)."""
    a = np.asarray(img.convert("RGB").resize((768, 768)))
    return a, (a.astype(int).sum(-1) < 740)


def shift(img: np.ndarray, mask: np.ndarray, hue: float = 0.0, sat: float = 1.0, val: float = 1.0) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + hue) % 180
    hsv[..., 1] *= sat
    hsv[..., 2] *= val
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    return np.where(mask[..., None], out, img)


def candidates(x: np.ndarray, inner: np.ndarray, outer: np.ndarray, prod: np.ndarray, prod_mask: np.ndarray,
               other: np.ndarray, other_mask: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {"pos": (prod, prod_mask), "other": (other, other_mask)}
    out["outer"] = as_result(masked_crop(x, outer & ~inner))
    out["strip"] = as_result(masked_crop(x, inner))
    merged = prod.copy()
    jacket = x[outer & ~inner]
    if len(jacket):
        # P 의 양옆(가운데 띠 바깥)을 겉옷 대표색으로 덮는다
        h, w = prod_mask.shape
        side = np.zeros_like(prod_mask)
        side[:, : int(w * 0.38)] = True
        side[:, int(w * 0.62):] = True
        merged[side & prod_mask] = np.median(jacket, axis=0).astype(np.uint8)
    out["merge"] = (merged, prod_mask)
    out["dark"] = (shift(prod, prod_mask, sat=0.35, val=0.45), prod_mask)
    out["hue"] = (shift(prod, prod_mask, hue=60), prod_mask)
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="검사기 보정 (양성·음성 대조군)")
    p.add_argument("--swap", required=True)
    p.add_argument("--bank", default="data/unpaired/bank")
    p.add_argument("--fashionpedia", default="data/fashionpedia")
    p.add_argument("--index", default="data/unpaired/index/layered.jsonl")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    bank = Path(args.bank)
    by_cat = defaultdict(list)
    for r in load_usable(bank):
        by_cat[r["category"]].append(r["item"])
    index = {json.loads(line)["file"]: json.loads(line) for line in Path(args.index).read_text().splitlines()}
    ann = Annotations(Path(args.fashionpedia))
    swap = Path(args.swap)
    scores: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    passed: dict[str, list[bool]] = defaultdict(list)
    for rec in map(json.loads, (swap / "attempts.jsonl").read_text().splitlines()):
        if not rec["approved"]:
            continue
        row = index[rec["file"]]
        photo = np.array(Image.open(ann.image_dir / rec["file"]).convert("RGB"))
        _, (inner, outer) = prepare(photo, [ann.mask(rec["file"], row["inner"]["ann_id"]),
                                            ann.mask(rec["file"], row["outer"]["ann_id"])])
        x = np.array(Image.open(swap / "input" / f"{stem_of(rec)}.jpg").convert("RGB"))
        prod, prod_mask = product_pixels(bank, "front", rec["product"])
        pool = [i for i in by_cat[rec["product_category"]] if i != rec["product"]]
        other_id = sorted(pool, key=lambda i: hashlib.sha1((rec["file"] + i).encode()).hexdigest())[0]
        other, other_mask = product_pixels(bank, "front", other_id)
        for name, (img, fg) in candidates(x, inner, outer, prod, prod_mask, other, other_mask).items():
            g = gates.check(x, inner, outer, img, fg, category=rec["product_category"])
            for b in (rec.get("share_bin", "?"), "all"):
                scores[b][name].append(gates.score(g))
            passed[name].append(g.passed)
    report = {"pairs": len(passed["pos"]),
              "pass_rate": {k: round(float(np.mean(v)), 3) for k, v in passed.items()},
              "reject_rate_negatives": {k: round(1 - float(np.mean(passed[k])), 3) for k in NEGATIVES},
              "auroc": {b: {k: round(auroc(s["pos"], s[k]), 3) for k in NEGATIVES} for b, s in scores.items()}}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

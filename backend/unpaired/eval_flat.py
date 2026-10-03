"""E-실제바닥 평가: 침대·바닥에 놓거나 옷걸이에 건 옷 한 벌을 폰으로 찍은 실제 사진(alexeygrigorev
clothing-dataset, CC0)에서 상품 사진을 만든다. 정답 상품 사진은 없으므로 결과물이 사진에 보이는 옷과
맞는지(형식·색·대상에 없던 색·표시 자국, gates 와 score 의 기준)만 잰다.

대상 표시는 BiRefNet 누끼(가장 큰 덩어리)로 만든다. 같은 사진에 학습한 LoRA 와 학습 없는 klein 을 돌려 비교한다.

사용 (컨테이너 안)
  python -m unpaired.eval_flat --lora data/unpaired/runs/lora_v1/step03000 --tag lora_v1 --n 200 --out results/unpaired/eval_flat
  python -m unpaired.eval_flat --tag klein_zs --n 200 --out results/unpaired/eval_flat
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from unpaired import gates
from unpaired.score import faithful, marker_fraction, right_item

# 데이터셋 라벨 → 우리 종류 이름. 신발·모자·아동복·"모름"은 뺀다
LABELS = {"T-Shirt": "t-shirt", "Longsleeve": "top", "Shirt": "shirt", "Polo": "t-shirt", "Undershirt": "tank top",
          "Top": "top", "Blouse": "blouse", "Hoodie": "hoodie", "Blazer": "blazer", "Outwear": "jacket",
          "Pants": "trousers", "Shorts": "shorts", "Skirt": "skirt", "Dress": "dress"}
ZS_PROMPT = ("Create a store product photo of the {cat} in this photo: the garment alone, laid flat and neatly "
             "smoothed, front view, centered on a plain white background. Keep its exact colors, pattern, print and "
             "details. Show only this one garment: no hanger, no hands, no bed, no other objects.")


def select(root: Path, n: int) -> list[dict]:
    rows = [r for r in csv.DictReader((root / "images.csv").open()) if r["label"] in LABELS and r["kids"] == "False"]
    rows.sort(key=lambda r: hashlib.sha1(r["image"].encode()).hexdigest())
    # 종류가 고르게 들어가게 라벨마다 돌아가며 뽑는다
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["label"], []).append(r)
    out, i = [], 0
    while len(out) < n and any(by.values()):
        lab = sorted(by)[i % len(by)]
        if by[lab]:
            out.append(by[lab].pop(0))
        i += 1
    return out


def largest(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask
    return lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="E-실제바닥 평가")
    p.add_argument("--root", default="data/clothing_cc0")
    p.add_argument("--lora")
    p.add_argument("--tag", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--out", required=True)
    p.add_argument("--base", action="store_true", help="klein-base-4B 본체로 CFG 다단계 추론")
    p.add_argument("--prompt-style", default="struct", help="학습 쌍과 같은 지시문 형식: struct / natural")
    args = p.parse_args(argv)

    from phase0.exp0_edit_models import disable_broken_cudnn
    from pipeline.garment.refine import BiRefNetRemover
    from unpaired import editors
    from unpaired.pairs import extract_prompt, fit, pointer_refs

    disable_broken_cudnn()
    root, out = Path(args.root), Path(args.out) / args.tag
    out.mkdir(parents=True, exist_ok=True)
    cases = select(root, args.n)
    remover = BiRefNetRemover()
    model = editors.load("klein", args.lora, base=args.base)
    rows = []
    for c in cases:
        photo = np.array(Image.open(root / "images" / f"{c['image']}.jpg").convert("RGB"))
        mask = largest(np.asarray(remover.predict_mask(Image.fromarray(photo))) > 127)
        if mask.sum() < 0.02 * mask.size:
            continue
        cat = LABELS[c["label"]]
        if args.lora:
            refs, prompt = pointer_refs(photo, mask), extract_prompt(cat, "flat", args.prompt_style)
        else:
            refs, prompt = [fit(photo, 768)], ZS_PROMPT.format(cat=cat)
        results = model(refs, prompt, list(range(args.k)), (768, 768))
        for k, res in enumerate(results):
            res.save(out / f"{c['image']}_k{k}.png")
            r = np.array(res.convert("RGB"))
            fg = np.asarray(remover.predict_mask(res)) > 127
            g = gates.check(photo, mask, None, r, fg, category=cat)
            row = {"image": c["image"], "label": c["label"], "k": k, "passed": g.passed, "reasons": g.reasons,
                   "scores": g.scores, "margin": 1.0, "marker_frac": round(marker_fraction(r, fg), 4)}
            row["faithful"] = faithful(row) and right_item(row)
            rows.append(row)
        print(f"{c['image']} {[x['reasons'] for x in rows[-args.k:]]}", flush=True)
    with (out / "scores.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    first = [r for r in rows if r["k"] == 0]
    by_img: dict[str, list[dict]] = {}
    for r in rows:
        by_img.setdefault(r["image"], []).append(r)
    summary = {"cases": len(first), "faithful_single": round(float(np.mean([r["faithful"] for r in first])), 3),
               "faithful_oracle": round(float(np.mean([any(x["faithful"] for x in v) for v in by_img.values()])), 3),
               "gate_pass": round(float(np.mean([r["passed"] for r in rows])), 3),
               "marker_drawn": round(float(np.mean([r["marker_frac"] > 0.01 for r in rows])), 3),
               "reasons": dict(Counter(x for r in rows for x in r["reasons"]).most_common()),
               "median_palette_dist": round(float(np.median([r["scores"].get("palette_dist", 99) for r in first])), 2)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

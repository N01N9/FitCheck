"""E-보이는옷 평가: 사진에 잘 보이는 옷(가장 바깥에 입은 옷) 한 벌 → 상품 사진.

사용자 방침(2026-10-06): 거의 가려진 이너는 맞추려 하지 않는다(정보가 없어 틀리면 신뢰만 깎인다). 서비스의 주력은
잘 보이는 바깥 옷이다. 그래서 그런 옷만으로 엄격 평가 세트를 만든다.

대상(Fashionpedia 상업 부분, 학습에 쓰지 않은 사진만 — 두 색인 모두 report 구간인 bucket < 20):
  top     겉옷 없이 입은 상의(shirt/t-shirt/sweater/top)
  outer   가장 바깥 겉옷(jacket/coat/cardigan/vest)
  bottom  pants/shorts/skirt
  dress   dress/jumpsuit
보이는 정도: 마스크가 사진의 4% 이상, 볼록 껍질 대비 면적(solidity) 0.7 이상(팔·가방에 크게 가리지 않음).

사용 (컨테이너 안)
  python -m unpaired.eval_visible build --out results/unpaired/eval_visible/cases.json --per 60
  python -m unpaired.eval_visible run --cases results/unpaired/eval_visible/cases.json --lora data/unpaired/runs/lora_v4/step09000 --tag lora_v4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from unpaired.layered import Annotations, bucket

GROUPS = {"top": {"shirt, blouse", "top, t-shirt, sweatshirt", "sweater"},
          "outer": {"jacket", "coat", "cardigan", "vest"},
          "bottom": {"pants", "shorts", "skirt"},
          "dress": {"dress", "jumpsuit"}}
SIMPLE = {"shirt, blouse": "shirt", "top, t-shirt, sweatshirt": "top", "sweater": "sweater", "jacket": "jacket",
          "coat": "coat", "cardigan": "cardigan", "vest": "vest", "pants": "trousers", "shorts": "shorts",
          "skirt": "skirt", "dress": "dress", "jumpsuit": "dress"}
MIN_AREA, MIN_SOLIDITY = 0.04, 0.7


def solidity(m: np.ndarray) -> float:
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return 0.0
    hull = cv2.convexHull(np.concatenate(cs))
    return float(m.sum() / max(cv2.contourArea(hull), 1))


def build(ann: Annotations, per: int) -> list[dict]:
    out = {g: [] for g in GROUPS}
    for file in sorted(ann.by_file, key=lambda f: bucket("vis" + f)):
        if bucket(file) >= 20:  # 학습 구간(layered train ≥30, solo train ≥35)과 겹치지 않게
            continue
        im = ann.images[file]
        area = im["width"] * im["height"]
        garments = ann.garments(file)
        outers = [a for a in garments if a["category"] in GROUPS["outer"]]
        for a in garments:
            g = next((k for k, v in GROUPS.items() if a["category"] in v), None)
            if g is None or len(out[g]) >= per or a.get("iscrowd"):
                continue
            if g == "top" and outers:  # 겉옷 아래 이너는 뺀다
                continue
            m = ann.mask(file, a["id"])
            if m.sum() < MIN_AREA * area or solidity(m) < MIN_SOLIDITY:
                continue
            out[g].append({"file": file, "ann_id": a["id"], "group": g, "category": SIMPLE[a["category"]],
                           "area_frac": round(float(m.sum() / area), 3)})
            break  # 사진 하나에 한 벌만
        if all(len(v) >= per for v in out.values()):
            break
    return [r for g in GROUPS for r in out[g]]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="E-보이는옷 평가")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--per", type=int, default=60)
    r = sub.add_parser("run")
    r.add_argument("--cases", required=True)
    r.add_argument("--lora", required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--k", type=int, default=1)
    r.add_argument("--style", default="dim", help="dim(넓은 이름) / dimfine(세부 종류)")
    r.add_argument("--groups", default="top,outer,bottom,dress")
    for q in (b, r):
        q.add_argument("--fashionpedia", default="data/fashionpedia")
    args = p.parse_args(argv)
    ann = Annotations(Path(args.fashionpedia))
    if args.cmd == "build":
        rows = build(ann, args.per)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=1))
        print({g: sum(r["group"] == g for r in rows) for g in GROUPS})
        return

    from phase0.exp0_edit_models import disable_broken_cudnn
    from unpaired import editors
    from unpaired.pairs import extract_prompt, pointer_refs

    disable_broken_cudnn()
    rows = [r for r in json.loads(Path(args.cases).read_text()) if r["group"] in args.groups.split(",")]
    out = Path(args.cases).parent / args.tag
    out.mkdir(parents=True, exist_ok=True)
    model = editors.load("klein", None if args.lora == "none" else args.lora)
    for c in rows:
        st = f"{Path(c['file']).stem}_{c['ann_id']}"
        if (out / f"{st}_k{args.k - 1}.png").exists():
            continue
        photo = np.array(Image.open(ann.image_dir / c["file"]).convert("RGB"))
        m = ann.mask(c["file"], c["ann_id"])
        layer = "outer" if c["group"] == "outer" else "single"
        res = model(pointer_refs(photo, m, "dim"), extract_prompt(c["category"], layer, args.style), list(range(args.k)), (768, 768))
        for k, im in enumerate(res):
            im.save(out / f"{st}_k{k}.png")
        print(st, flush=True)


if __name__ == "__main__":
    main()

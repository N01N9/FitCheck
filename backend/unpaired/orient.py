"""은행 상품 사진의 방향 이상을 찾는다(사람 검수 없이).

bank.py 는 촬영대 배치에 맞춰 모든 사진을 반시계 90° 돌린다. 대부분은 그것으로 바로 서지만,
옷을 다른 방향으로 놓은 품목은 옆으로 눕거나 거꾸로 선다. 대다수가 바로 서 있다는 점을 이용한다.

  1) 종류별로 DINOv2 특징의 평균(바로 선 모습의 기준)을 구한다
  2) 사진마다 0/90/180/270° 로 돌린 특징을 기준과 비교한다
  3) 0° 가 아닌 방향이 MARGIN 이상 더 가까우면 그 방향을 "의심"으로 기록한다

진단용이다. 2026-10-03 표본에서 의심의 대부분이 실제로는 바로 선 그래픽 옷이었으므로(정밀도 낮음)
bank.load_usable 은 기본적으로 이 결과로 품목을 빼지 않는다.

사용 (컨테이너 안)
  python -m unpaired.orient --bank data/unpaired/bank
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

MARGIN = 0.03
ANGLES = (0, 90, 180, 270)


def choose(emb: np.ndarray, centroid: np.ndarray, margin: float = MARGIN) -> tuple[int, list[float]]:
    """emb: (4, D) 회전별 특징(정규화됨), centroid: (D,). (의심 각도 또는 0, 회전별 유사도)."""
    c = centroid / max(np.linalg.norm(centroid), 1e-12)
    sims = emb @ c
    best = int(np.argmax(sims))
    if best != 0 and sims[best] - sims[0] >= margin:
        return ANGLES[best], [round(float(x), 4) for x in sims]
    return 0, [round(float(x), 4) for x in sims]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="은행 사진 방향 이상 찾기")
    p.add_argument("--bank", default="data/unpaired/bank")
    p.add_argument("--view", default="front")
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args(argv)

    from phase0.exp0_edit_models import disable_broken_cudnn
    from unpaired.score import Dino

    disable_broken_cudnn()
    bank = Path(args.bank)
    rows = [r for r in map(json.loads, (bank / "bank.jsonl").read_text().splitlines())
            if r["accepted"] and r["view"] == args.view]
    dino = Dino()
    embs = {}
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        ims = [Image.open(bank / args.view / f"{r['item']}.jpg").convert("RGB") for r in chunk]
        rotated = [im.rotate(a, expand=True, fillcolor=(255, 255, 255)) for im in ims for a in ANGLES]
        e = dino(rotated).reshape(len(chunk), len(ANGLES), -1)
        for r, x in zip(chunk, e):
            embs[r["item"]] = x
    centroids = defaultdict(list)
    for r in rows:
        centroids[r["category"]].append(embs[r["item"]][0])
    centroids = {k: np.mean(v, axis=0) for k, v in centroids.items()}
    out = []
    for r in rows:
        angle, sims = choose(embs[r["item"]], centroids[r["category"]])
        out.append({"item": r["item"], "view": args.view, "category": r["category"], "suspect_angle": angle,
                    "sims": sims})
    with (bank / f"orient_{args.view}.jsonl").open("w") as fh:
        for o in out:
            fh.write(json.dumps(o) + "\n")
    summary = {"total": len(out), "suspect": dict(Counter(o["suspect_angle"] for o in out)),
               "suspect_by_category": dict(Counter(o["category"] for o in out if o["suspect_angle"]))}
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

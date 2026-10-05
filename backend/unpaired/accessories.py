"""옷 위에 걸친 장신구(넥타이·벨트·스카프·가방끈)를 상품 사진에 덧붙인 학습 쌍을 만든다.

v2 엄격 판정에서 셔츠 위 넥타이, 블라우스 위 벨트, 목걸이를 상품 사진에 같이 그렸다(84건 중 6건). 입력 상품 위에
Fashionpedia 실제 사진의 장신구 조각을 얹고, 외곽선은 옷에서 장신구를 뺀 영역에 그리며, 정답은 깨끗한 원래 상품이다.
그러면 "옷 위에 있지만 옷이 아닌 것은 그리지 않는다" 를 배운다. 정답은 실제 상품 사진 그대로다.

조각은 평가에 쓰는 사진과 겹치지 않게 두 색인 모두에서 train 분할인 사진에서만 꺼낸다.

사용 (컨테이너 안)
  python -m unpaired.accessories collect --out data/unpaired/accessories --per-cat 400
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from unpaired.layered import LAYERED_SPLITS, SOLO_SPLITS, Annotations, split_of
from unpaired.masks import bbox

CATS = {"tie": "tie", "belt": "belt", "scarf": "scarf", "bag, wallet": "bag"}
MIN_AREA = {"tie": 1500, "belt": 1200, "scarf": 4000, "bag": 6000}


def collect(fashionpedia: Path, out: Path, per_cat: int) -> dict:
    ann = Annotations(fashionpedia)
    out.mkdir(parents=True, exist_ok=True)
    counts = {v: 0 for v in CATS.values()}
    rows = []
    for (file, aid), a in sorted(ann.anns.items()):
        kind = CATS.get(a["category"])
        if kind is None or counts[kind] >= per_cat or a.get("iscrowd"):
            continue
        if split_of(file, LAYERED_SPLITS) != "train" or split_of(file, SOLO_SPLITS) != "train":
            continue
        if a["area"] < MIN_AREA[kind]:
            continue
        m = ann.mask(file, aid)
        x0, y0, x1, y1 = bbox(m)
        photo = np.array(Image.open(ann.image_dir / file).convert("RGB"))
        rgba = np.dstack([photo[y0:y1, x0:x1], (m[y0:y1, x0:x1] * 255).astype(np.uint8)])
        name = f"{kind}_{Path(file).stem}_{aid}.png"
        Image.fromarray(rgba).save(out / name)
        rows.append({"file": name, "kind": kind, "size": [x1 - x0, y1 - y0]})
        counts[kind] += 1
        if all(c >= per_cat for c in counts.values()):
            break
    (out / "pieces.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return counts


def load(root: Path) -> list[dict]:
    return [json.loads(l) for l in (root / "pieces.jsonl").read_text().splitlines()]


def place(img: np.ndarray, mask: np.ndarray, piece: Image.Image, kind: str, rng) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """상품 img(배경 흰색)와 옷 mask 위에 장신구를 얹는다. 결과 이미지, 장신구 영역, 그중 옷을 덮은 영역을 돌려준다."""
    x0, y0, x1, y1 = bbox(mask)
    w, h = x1 - x0, y1 - y0
    p = piece
    if kind == "tie":  # 목 가운데에서 아래로
        scale = rng.uniform(0.45, 0.7) * h / p.height
        cx, top = x0 + w * rng.uniform(0.46, 0.54), y0 + h * rng.uniform(0.02, 0.08)
    elif kind == "belt":  # 허리 높이 가로로
        if p.height > p.width:
            p = p.rotate(90, expand=True)
        scale = rng.uniform(0.7, 1.0) * w / p.width
        cx, top = x0 + w / 2, y0 + h * rng.uniform(0.6, 0.8)
    elif kind == "scarf":  # 목 둘레에서 늘어뜨림
        scale = rng.uniform(0.4, 0.65) * w / p.width
        cx, top = x0 + w * rng.uniform(0.4, 0.6), y0 + h * rng.uniform(0.0, 0.05)
    else:  # 가방: 몸통 옆이나 앞
        scale = rng.uniform(0.3, 0.5) * h / p.height
        cx, top = x0 + w * rng.uniform(0.25, 0.75), y0 + h * rng.uniform(0.3, 0.55)
    p = p.resize((max(4, int(p.width * scale)), max(4, int(p.height * scale))), Image.LANCZOS)
    left, top = int(cx - p.width / 2), int(top)
    canvas = Image.fromarray(img).convert("RGBA")
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    layer.paste(p, (left, top), p)
    alpha = np.asarray(layer)[..., 3]
    # 얹은 조각이 옷 밖으로 크게 나가면(가방 등) 쓸 만하지만, 옷을 거의 안 덮으면 의미가 없다
    covered = (alpha > 127) & mask
    composed = Image.alpha_composite(canvas, layer).convert("RGB")
    return np.asarray(composed), cv2.dilate((alpha > 127).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool), covered


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="장신구 조각 모으기")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--fashionpedia", default="data/fashionpedia")
    c.add_argument("--out", default="data/unpaired/accessories")
    c.add_argument("--per-cat", type=int, default=400)
    args = p.parse_args(argv)
    print(json.dumps(collect(Path(args.fashionpedia), Path(args.out), args.per_cat)))


if __name__ == "__main__":
    main()

"""데이터 소스 E: 바닥·침대에 널브러진 옷 사진을 CPU 로 합성한다. 정답은 각 옷의 실제 상품 사진이다.

은행 상품(누끼) 1~4벌을 이불·러그·마루 사진 위에 돌리고, 부드럽게 휘고(탄성 변형), 겹쳐 놓고,
그림자를 깐다. 어떤 옷이 어떤 옷을 얼마나 가리는지 정확히 알기 때문에 가려진 부분의 정답도 정확하다.
생성 모델을 쓰지 않으므로 지어낸 픽셀이 정답에 섞이지 않는다(입력만 합성).

배경은 Commons 수집분 중 옷이 찍히지 않은 범주만 쓴다(빨랫줄·옷걸이·마네킹·빨래 범주 제외).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from unpaired.bank import load_usable

BACKGROUND_SEEDS = {"Category:Beds", "Category:Quilts", "Category:Blankets", "Category:Bedding", "Category:Duvets",
                    "Category:Rugs", "search:hardwood floor", "search:couch living room", "Category:Cushions"}
MIN_TARGET_VISIBLE = 0.4  # 정답으로 쓸 옷은 이만큼은 보여야 한다
# 이 범주들에는 박물관 소장 그림·태피스트리·판화가 많이 섞여 있어 제목으로 거른다
ART_WORDS = ("painting", "drawing", "engraving", "illustration", "tapestry", "museum", "statue", "sculpture",
             "poster", "lithograph", "sketch", "fresco", "mosaic", "icon", "plan", "diagram", "woodcut", "etching",
             "manuscript", "miniature", "portrait", "gemälde", "schilderij", "musée", "museo", "art ")
MIN_SATURATION = 0.12     # 흑백·세피아 사진 제외(HSV 채도 평균)


@dataclass
class Placed:
    item: str
    category: str
    full: np.ndarray      # 가림 없이 놓였을 때 차지하는 영역
    visible: np.ndarray   # 위에 놓인 옷에 가려지고 남은 영역
    order: int            # 0 이 맨 아래


def backgrounds(crawl_dir: Path) -> list[Path]:
    """옷이 없는 범주의 실제 사진만. 그림·판화(제목)와 흑백 사진(채도)은 뺀다."""
    out = []
    for r in csv.DictReader((crawl_dir / "sources.csv").open(encoding="utf-8")):
        if r["seed"] not in BACKGROUND_SEEDS or any(w in r["title"].lower() for w in ART_WORDS):
            continue
        path = crawl_dir / "raw" / r["file"]
        small = np.asarray(Image.open(path).convert("RGB").resize((64, 64)))
        if cv2.cvtColor(small, cv2.COLOR_RGB2HSV)[..., 1].mean() / 255 >= MIN_SATURATION:
            out.append(path)
    return out


def background_crop(path: Path, size: int, rng: np.random.Generator) -> np.ndarray:
    """배경의 일부를 확대해 잘라 질감 위주로 쓴다(방 전체 사진에 옷이 떠 보이지 않게)."""
    img = np.array(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    side = int(min(h, w) * rng.uniform(0.45, 0.9))
    y, x = rng.integers(0, h - side + 1), rng.integers(0, w - side + 1)
    return cv2.resize(img[y:y + side, x:x + side], (size, size), interpolation=cv2.INTER_AREA)


def elastic(img: np.ndarray, mask: np.ndarray, rng: np.random.Generator, amp: float = 0.025, smooth: float = 0.12):
    """부드러운 무작위 변위장으로 옷을 살짝 휜다(구김 흉내)."""
    h, w = mask.shape
    sigma = smooth * max(h, w)
    dx = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma)
    dy = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (0, 0), sigma)
    scale = amp * max(h, w) / max(np.abs(dx).max(), np.abs(dy).max(), 1e-6)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = gx + dx * scale, gy + dy * scale
    warped = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    wmask = cv2.remap(mask.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderValue=0)
    return warped, wmask


def place(canvas: np.ndarray, product: np.ndarray, alpha: np.ndarray, rng: np.random.Generator):
    """상품을 돌려·줄여 캔버스 위 무작위 위치에 놓을 변환과 결과(놓인 색, 알파)를 만든다."""
    size = canvas.shape[0]
    ys, xs = np.nonzero(alpha > 0.5)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    long_side = max(xs.max() - xs.min(), ys.max() - ys.min())
    scale = rng.uniform(0.38, 0.7) * size / long_side
    angle = rng.uniform(-35, 35) + (180 if rng.random() < 0.1 else 0)
    tx, ty = rng.uniform(0.25, 0.75) * size, rng.uniform(0.25, 0.75) * size
    m = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    m[:, 2] += (tx - cx, ty - cy)
    color = cv2.warpAffine(product, m, (size, size), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    a = cv2.warpAffine(alpha, m, (size, size), flags=cv2.INTER_LINEAR, borderValue=0)
    return color, np.clip(a, 0, 1)


def shade(color: np.ndarray, canvas: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """배경 조명에 맞춰 밝기·색을 조금 옮긴다."""
    gain = rng.uniform(0.82, 1.03)
    tint = (canvas.reshape(-1, 3).mean(0) / 255 - 0.5) * rng.uniform(0.0, 0.15)
    return np.clip(color.astype(np.float32) * gain * (1 + tint), 0, 255)


def compose(products: list[tuple[str, str, np.ndarray, np.ndarray]], bg: np.ndarray, rng: np.random.Generator):
    """products: (item, category, 1024 상품 사진, 0~1 알파). (합성 사진, Placed 목록)."""
    canvas = bg.astype(np.float32)
    size = canvas.shape[0]
    placed: list[Placed] = []
    for order, (item, cat, img, alpha) in enumerate(products):
        warped, walpha = elastic(img, alpha, rng)
        color, a = place(canvas, warped, walpha, rng)
        # 그림자: 알파를 조금 밀고 흐리게 해서 배경을 어둡게 한다
        off = rng.uniform(0.004, 0.012) * size
        sh = cv2.warpAffine(a, np.float32([[1, 0, off], [0, 1, off]]), (size, size))
        sh = cv2.GaussianBlur(sh, (0, 0), 0.01 * size)[..., None] * rng.uniform(0.15, 0.35)
        canvas = canvas * (1 - sh)
        a3 = a[..., None]
        canvas = canvas * (1 - a3) + shade(color, bg, rng) * a3
        full = a > 0.5
        for p in placed:
            p.visible &= ~full
        placed.append(Placed(item, cat, full, full.copy(), order))
    return canvas.round().astype(np.uint8), placed


def targets(placed: list[Placed]) -> list[Placed]:
    """정답으로 쓸 만큼 보이는 옷(보이는 비율 MIN_TARGET_VISIBLE 이상)."""
    return [p for p in placed if p.full.sum() and p.visible.sum() / p.full.sum() >= MIN_TARGET_VISIBLE]


def n_items(rng: np.random.Generator) -> int:
    return int(rng.choice([1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.15]))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="바닥·침대 옷 사진 합성 (데이터 소스 E)")
    p.add_argument("--bank", default="data/unpaired/bank")
    p.add_argument("--backgrounds", default="data/crawl/commons_background")
    p.add_argument("--n", type=int, default=200, help="만들 장면 수")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--out", default="data/unpaired/flatlay/pilot")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    bank = Path(args.bank)
    rows = load_usable(bank)
    bgs = backgrounds(Path(args.backgrounds))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "scenes.jsonl").open("w") as fh:
        for i in range(args.n):
            rng = np.random.default_rng([args.seed, i])
            chosen = [rows[j] for j in rng.choice(len(rows), n_items(rng), replace=False)]
            products = []
            for r in chosen:
                img = np.array(Image.open(bank / "front" / f"{r['item']}.jpg").convert("RGB"))
                alpha = np.asarray(Image.open(bank / "front_mask" / f"{r['item']}.png"), np.float32) / 255
                products.append((r["item"], r["category"], img, alpha))
            bg_path = bgs[int(rng.integers(len(bgs)))]
            scene, placed = compose(products, background_crop(bg_path, args.size, rng), rng)
            labels = np.zeros(scene.shape[:2], np.uint8)
            for p_ in placed:
                labels[p_.visible] = p_.order + 1
            Image.fromarray(scene).save(out / f"scene{i:05d}.jpg", quality=93)
            Image.fromarray(labels).save(out / f"scene{i:05d}_labels.png")
            keep = {id(t) for t in targets(placed)}
            fh.write(json.dumps({"scene": f"scene{i:05d}", "background": bg_path.name, "items": [
                {"item": p_.item, "category": p_.category, "label": p_.order + 1,
                 "visible_frac": round(float(p_.visible.sum() / max(p_.full.sum(), 1)), 3), "target": id(p_) in keep}
                for p_ in placed]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

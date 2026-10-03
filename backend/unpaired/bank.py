"""Second-Hand 의류 데이터셋(CC BY 4.0)을 "정답 상품 사진 은행"으로 만든다.

원본은 격자·줄자가 있는 작업대 위 사진이다. BiRefNet(MIT)으로 옷만 따서 흰 정사각형 가운데 놓는다.
세 촬영대 모두 옷의 위쪽(목·허리)이 사진 오른쪽을 향하게 놓여 있어서 반시계 방향으로 90° 돌려 세운다.
생성 모델로 고치거나 꾸미지 않는다(정답은 실제 픽셀이어야 한다). 품질 기준에 못 미치면 버리고 이유를 남긴다.

  잘림     옷이 사진 가장자리에 닿음(일부가 화면 밖)
  작음     옷의 긴 변이 MIN_LONG_SIDE px 미만
  여러 덩어리  큰 덩어리가 둘 이상(다른 옷·물건이 같이 찍힘)
  모양     볼록 껍질 대비 면적이 너무 작음(구겨짐·뒤엉킴 의심)

출력 (out/)
  front/<item>.jpg, front_mask/<item>.png   1024px 정규화 사진과 마스크 (뒷면은 back/)
  bank.jsonl                                품목별 종류·색·무늬, 품질 수치, 통과 여부

사용 (컨테이너 안)
  python -m unpaired.bank --src data/secondhand --out data/unpaired/bank --views front
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# 라벨 type → (묶음, 영어 이름). 묶음: inner(이너 상의) / outer(겉옷) / bottom / onepiece
CATEGORY = {
    "Top": ("inner", "top"), "T-shirt": ("inner", "t-shirt"), "Shirt": ("inner", "shirt"),
    "Blouse": ("inner", "blouse"), "Sweater": ("inner", "sweater"), "Tank top": ("inner", "tank top"),
    "Hoodie": ("inner", "hoodie"), "Training top": ("inner", "training top"), "Tunic": ("inner", "tunic"),
    "Jacket": ("outer", "jacket"), "Jacker": ("outer", "jacket"), "Blazer": ("outer", "blazer"),
    "Outerwear": ("outer", "coat"), "Winter jacket": ("outer", "winter jacket"), "Cardigan": ("outer", "cardigan"),
    "Vest": ("outer", "vest"), "Rain jacket": ("outer", "rain jacket"), "Denim jacket": ("outer", "denim jacket"),
    "Trousers": ("bottom", "trousers"), "Jeans": ("bottom", "jeans"), "Shorts": ("bottom", "shorts"),
    "Skirt": ("bottom", "skirt"), "Rain trousers": ("bottom", "rain trousers"),
    "Winter trousers": ("bottom", "winter trousers"),
    "Dress": ("onepiece", "dress"),
}

SIZE = 1024
ROTATE = 1             # np.rot90 횟수(반시계 90°). 촬영대 배치 때문에 옷 위쪽이 오른쪽에 있다
FILL = 0.86            # 정규화 사진에서 옷의 긴 변이 차지하는 비율
MIN_LONG_SIDE = 400    # 원본에서 옷의 긴 변(px)
MIN_SOLIDITY = 0.55
BORDER_TOUCH = 0.01    # 한 변에서 옷이 닿은 길이 비율이 이보다 크면 잘린 것으로 본다


def items(src: Path):
    for lab in sorted(src.glob("station*/**/labels_*.json")):
        item = lab.stem[len("labels_"):]
        yield {"item": item, "station": lab.relative_to(src).parts[0], "labels": lab,
               "front": lab.with_name(f"front_{item}.jpg"), "back": lab.with_name(f"back_{item}.jpg")}


def category(labels: dict) -> tuple[str, str] | None:
    """잠옷·스타킹 등은 None. 라벨 값의 끝 공백·소문자 표기 차이는 맞춰 준다."""
    t = str(labels.get("type") or "").strip()
    return CATEGORY.get(t) or CATEGORY.get(t.capitalize())


def quality(mask: np.ndarray) -> dict:
    h, w = mask.shape
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA] if n > 1 else np.array([0])
    total = max(int(mask.sum()), 1)
    big = int((areas >= 0.05 * total).sum())
    main = (lab == 1 + int(np.argmax(areas))) if n > 1 else mask
    ys, xs = np.nonzero(main)
    long_side = int(max(xs.max() - xs.min(), ys.max() - ys.min()) + 1) if len(xs) else 0
    pts = cv2.findNonZero(main.astype(np.uint8))
    hull = cv2.contourArea(cv2.convexHull(pts)) if pts is not None else 0
    touch = max(main[0].mean(), main[-1].mean(), main[:, 0].mean(), main[:, -1].mean())
    return {"area_frac": round(total / (h * w), 4), "components": big, "long_side": long_side,
            "solidity": round(float(main.sum() / max(hull, 1)), 3), "border_touch": round(float(touch), 4)}


def verdict(q: dict) -> list[str]:
    reasons = []
    if q["border_touch"] > BORDER_TOUCH:
        reasons.append("잘림")
    if q["long_side"] < MIN_LONG_SIDE:
        reasons.append("작음")
    if q["components"] > 1:
        reasons.append("여러 덩어리")
    if q["solidity"] < MIN_SOLIDITY:
        reasons.append("모양")
    return reasons


def normalize(img: np.ndarray, alpha: np.ndarray, size: int = SIZE, fill: float = FILL) -> tuple[Image.Image, Image.Image]:
    """알파의 bbox 로 자르고, 긴 변이 size*fill 이 되게 키워 흰 정사각형 가운데 합성한다."""
    ys, xs = np.nonzero(alpha > 0.5)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop, a = img[y0:y1, x0:x1].astype(np.float32), alpha[y0:y1, x0:x1, None]
    scale = size * fill / max(x1 - x0, y1 - y0)
    nw, nh = max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale))
    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LANCZOS4 if scale > 1 else cv2.INTER_AREA)
    a = cv2.resize(a[..., 0], (nw, nh), interpolation=cv2.INTER_LINEAR)[..., None]
    canvas = np.full((size, size, 3), 255, np.float32)
    mask = np.zeros((size, size), np.float32)
    ox, oy = (size - nw) // 2, (size - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = a * np.clip(crop, 0, 255) + (1 - a) * 255
    mask[oy:oy + nh, ox:ox + nw] = a[..., 0]
    return Image.fromarray(canvas.round().astype(np.uint8)), Image.fromarray((mask * 255).round().astype(np.uint8))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Second-Hand 데이터셋 → 정답 상품 사진 은행")
    p.add_argument("--src", default="data/secondhand")
    p.add_argument("--out", default="data/unpaired/bank")
    p.add_argument("--views", default="front", help="front 또는 front,back")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--stride", type=int, default=1, help="N 개마다 하나씩(시범용 표본)")
    args = p.parse_args(argv)

    from phase0.exp0_edit_models import disable_broken_cudnn
    from pipeline.garment.refine import BiRefNetRemover

    disable_broken_cudnn()
    remover = BiRefNetRemover()
    out = Path(args.out)
    views = args.views.split(",")
    for v in views:
        (out / v).mkdir(parents=True, exist_ok=True)
        (out / f"{v}_mask").mkdir(parents=True, exist_ok=True)
    done = set()
    manifest = out / "bank.jsonl"
    if manifest.exists():
        done = {(r["item"], r["view"]) for r in map(json.loads, manifest.read_text().splitlines())}
    log = manifest.open("a")
    start, n = time.perf_counter(), 0
    for idx, it in enumerate(items(Path(args.src))):
        if idx % args.stride:
            continue
        labels = json.loads(it["labels"].read_text())
        cat = category(labels)
        for v in views:
            if (it["item"], v) in done:
                continue
            img = np.array(Image.open(it[v]).convert("RGB"))
            alpha = np.asarray(remover.predict_mask(Image.fromarray(img)), np.float32) / 255
            q = quality(alpha > 0.5) if (alpha > 0.5).any() else {"area_frac": 0}
            reasons = verdict(q) if q["area_frac"] > 0 else ["옷을 찾지 못함"]
            if cat is None:
                reasons.append("종류 제외")
            if not reasons:
                im, m = normalize(np.ascontiguousarray(np.rot90(img, ROTATE)), np.ascontiguousarray(np.rot90(alpha, ROTATE)))
                im.save(out / v / f"{it['item']}.jpg", quality=95)
                m.save(out / f"{v}_mask" / f"{it['item']}.png")
            row = {"item": it["item"], "view": v, "station": it["station"], "source": str(it[v].relative_to(args.src)),
                   "group": cat[0] if cat else None, "category": cat[1] if cat else labels.get("type"),
                   "colors": labels.get("colors"), "pattern": labels.get("pattern"), "size": img.shape[1::-1],
                   "rotation": 90 * ROTATE, **q, "accepted": not reasons, "reasons": reasons}
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
        if n and n % 500 == 0:
            log.flush()
            print(f"{n} done, {(time.perf_counter() - start) / n:.2f}s/img", flush=True)
        if args.limit and n >= args.limit:
            break
    log.close()


if __name__ == "__main__":
    main()

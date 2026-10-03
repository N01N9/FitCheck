"""데이터 엔진: 정답은 실제 데이터로 고정하고, 입력 사진을 거꾸로 만든다(짝 데이터 없이 학습 쌍 만들기).

  swap   겹쳐 입은 실제 사진(Fashionpedia)의 이너만 은행 상품 P 로 바꾼다.
         정답 = P(실제 상품 사진). 겉옷·사람·배경은 실제 그대로
  layer  상의가 다 보이는 실제 사진에 은행 겉옷 O 를 덧입힌다.
         정답 = 원본 사진(겉옷 벗기기) / 원본 상의(이너 추출) / O(겉옷 추출)

생성 모델(klein-4B)은 사진 전체를 다시 그리므로, 바꾸려는 곳만 생성 결과에서 가져오고 나머지는
원본 픽셀로 되돌린다. 그 뒤 자동 검사를 통과한 것만 승인한다. 시도·승인 수와 시간을 남겨 수율을 잰다.

사용 (컨테이너 안)
  python -m unpaired.engine swap --n 200 --out data/unpaired/engine/swap_pilot
  python -m unpaired.engine layer --n 200 --out data/unpaired/engine/layer_pilot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from core.color import srgb_to_lab
from unpaired.color import delta_e2000, nearest, palette, palette_distance
from unpaired.gates import erode, fill_holes
from unpaired.layered import Annotations, load
from unpaired.masks import bbox

SWAP_PROMPT = ("Image 1 shows a person wearing a {cat} under a {outer}. Replace only the {cat} with the garment from "
               "image 2: the same garment with its exact colors, pattern and print. Keep the {outer}, the person, face, "
               "hair, hands, pose, the other clothes and the background exactly the same.")
LAYER_PROMPT = ("Dress the person in image 1 in the {outer} from image 2, worn open and unbuttoned over their {cat} so "
                "the {cat} stays visible down the middle. Keep the person, face, hair, pose, the {cat}, the other "
                "clothes and the background exactly the same.")

# 사진 속 이너 종류 → 바꿔 넣을 은행 상품 종류
SWAP_CATEGORIES = {
    "t-shirt": ("t-shirt", "top", "tank top"),
    "shirt": ("shirt", "blouse"),
    "sweater": ("sweater", "top"),
}
LAYER_CATEGORIES = ("jacket", "blazer", "cardigan", "denim jacket", "coat")

T = {
    "swap_changed_min": 6.0,     # 이너 자리 ΔE00 중앙값. 작으면 안 바뀐 것
    "swap_palette_max": 12.0,    # 새 이너 색 → 상품 색
    "swap_leak_max": 0.2,        # 옛 이너 색에 더 가까운 픽셀 비율
    "swap_ring_max": 12.0,       # 이너 둘레(겉옷 가장자리) ΔE00 중앙값
    "layer_cover_min": 0.3,      # 상의 중 새 겉옷에 덮인 비율
    "layer_cover_max": 0.92,
    "layer_face_max": 0.05,      # 상의 위쪽(얼굴 쪽)으로 번진 변화 비율
    "layer_palette_max": 15.0,   # 새 겉옷 색 → 은행 겉옷 색
}


def h01(text: str) -> float:
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def prepare(photo: np.ndarray, masks: list[np.ndarray], max_side: int = 1024, multiple: int = 16):
    """사진과 마스크를 모델 입력 크기(긴 변 max_side, multiple 배수)로 같이 줄인다."""
    h, w = photo.shape[:2]
    s = min(1.0, max_side / max(h, w))
    nw, nh = int(w * s) // multiple * multiple, int(h * s) // multiple * multiple
    sw, sh = int(round(w * s)), int(round(h * s))
    img = cv2.resize(photo, (sw, sh), interpolation=cv2.INTER_AREA)[:nh, :nw]
    out = [cv2.resize(m.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)[:nh, :nw].astype(bool) for m in masks]
    return img, out


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def feather(mask: np.ndarray, px: int) -> np.ndarray:
    a = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(px, 1) / 2)
    return np.clip(a, 0, 1)[..., None]


def composite(orig: np.ndarray, gen: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    return (orig.astype(np.float32) * (1 - alpha) + gen.astype(np.float32) * alpha).round().astype(np.uint8)


def changed(orig_lab: np.ndarray, gen_lab: np.ndarray, threshold: float = 14.0, min_frac: float = 0.002) -> np.ndarray:
    """원본과 생성 결과가 크게 다른 픽셀(덩어리 단위로 정리)."""
    d = delta_e2000(cv2.GaussianBlur(orig_lab.astype(np.float32), (0, 0), 1.5),
                    cv2.GaussianBlur(gen_lab.astype(np.float32), (0, 0), 1.5))
    m = (d > threshold).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k, iterations=2)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(m.shape, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_frac * m.size:
            keep |= lab == i
    return keep


def product_pixels(bank: Path, view: str, item: str):
    img = np.array(Image.open(bank / view / f"{item}.jpg").convert("RGB"))
    mask = np.asarray(Image.open(bank / f"{view}_mask" / f"{item}.png")) > 127
    return img, mask


def swap_check(orig: np.ndarray, gen: np.ndarray, inner: np.ndarray, outer: np.ndarray, prod_pal) -> dict:
    o_lab, g_lab = srgb_to_lab(orig), srgb_to_lab(gen)
    margin = max(1, round(0.004 * max(orig.shape[:2])))
    core = erode(inner, margin)
    s = {"changed": float(np.median(delta_e2000(o_lab[core], g_lab[core])))}
    new_pal = palette(g_lab[core])
    old_pal = palette(o_lab[core])
    s["palette_dist"] = palette_distance(new_pal, prod_pal)
    s["old_vs_prod"] = palette_distance(old_pal, prod_pal)
    px = g_lab[core]
    s["leak"] = float(np.mean(nearest(px, old_pal[0]) + 3.0 < nearest(px, prod_pal[0])))
    ring = dilate(inner, 3 * margin) & ~dilate(inner, margin) & outer
    s["ring"] = float(np.median(delta_e2000(o_lab[ring], g_lab[ring]))) if ring.sum() > 20 else 0.0
    reasons = []
    same = s["old_vs_prod"] < 8
    if not same and s["changed"] < T["swap_changed_min"]:
        reasons.append("안 바뀜")
    if s["palette_dist"] > T["swap_palette_max"]:
        reasons.append("상품 색이 아님")
    if not same and s["leak"] > T["swap_leak_max"]:
        reasons.append("옛 이너가 남음")
    if s["ring"] > T["swap_ring_max"]:
        reasons.append("겉옷이 바뀜")
    return {"scores": s, "reasons": reasons}


def layer_check(orig: np.ndarray, gen: np.ndarray, top: np.ndarray, prod_pal) -> tuple[dict, np.ndarray]:
    o_lab, g_lab = srgb_to_lab(orig), srgb_to_lab(gen)
    diff = changed(o_lab, g_lab)
    near = dilate(top, max(3, round(0.08 * max(top.shape))))
    n, lab, _, _ = cv2.connectedComponentsWithStats(diff.astype(np.uint8), connectivity=8)
    jacket = np.zeros(top.shape, bool)
    for i in range(1, n):
        comp = lab == i
        if (comp & near).sum() >= 0.5 * comp.sum():
            jacket |= comp
    jacket = fill_holes(jacket)
    s = {"cover": float((jacket & top).sum() / max(top.sum(), 1))}
    visible = top & ~jacket
    s["inner_share"] = float(visible.sum() / max(visible.sum() + jacket.sum(), 1))
    y0 = bbox(top)[1]
    above = np.zeros(top.shape, bool)
    above[: max(0, y0 - round(0.03 * top.shape[0]))] = True
    s["face"] = float((jacket & above).sum() / max(jacket.sum(), 1))
    reasons = []
    if not T["layer_cover_min"] <= s["cover"] <= T["layer_cover_max"]:
        reasons.append("겉옷이 안 덮음" if s["cover"] < T["layer_cover_min"] else "상의가 다 가려짐")
    if s["face"] > T["layer_face_max"]:
        reasons.append("얼굴 쪽이 바뀜")
    if jacket.sum() > 50:
        s["palette_dist"] = palette_distance(palette(g_lab[erode(jacket, 2)]), prod_pal)
        if s["palette_dist"] > T["layer_palette_max"]:
            reasons.append("은행 겉옷 색이 아님")
    else:
        reasons.append("겉옷 없음")
    return {"scores": s, "reasons": reasons}, jacket


def pick_products(bank_rows: list[dict], categories, key: str, n: int = 1) -> list[dict]:
    pool = sorted((r for r in bank_rows if r["category"] in categories), key=lambda r: h01(key + r["item"]))
    return pool[:n]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="짝 데이터 엔진 (이너 교체 / 겉옷 덧입히기)")
    p.add_argument("mode", choices=["swap", "layer"])
    p.add_argument("--n", type=int, default=200, help="시도 횟수")
    p.add_argument("--index-dir", default="data/unpaired/index")
    p.add_argument("--bank", default="data/unpaired/bank")
    p.add_argument("--fashionpedia", default="data/fashionpedia")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import torch

    from unpaired import editors

    out = Path(args.out)
    (out / "input").mkdir(parents=True, exist_ok=True)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    bank = Path(args.bank)
    bank_rows = [r for r in map(json.loads, (bank / "bank.jsonl").read_text().splitlines()) if r["accepted"]]
    ann = Annotations(Path(args.fashionpedia))
    if args.mode == "swap":
        rows = load(Path(args.index_dir) / "layered.jsonl", "train")
    else:
        rows = load(Path(args.index_dir) / "solo_tops.jsonl", "train")
    rows.sort(key=lambda r: h01(args.mode + r["file"]))
    rows = rows[: args.n]
    model = editors.load("klein")
    log = (out / "attempts.jsonl").open("a")
    done = {json.loads(line)["file"] for line in (out / "attempts.jsonl").read_text().splitlines() if line}
    for i, row in enumerate(rows):
        if row["file"] in done:
            continue
        photo = np.array(Image.open(ann.image_dir / row["file"]).convert("RGB"))
        cat = row["inner"]["category"]
        if args.mode == "swap":
            masks = [ann.mask(row["file"], row["inner"]["ann_id"]),
                     ann.mask(row["file"], row["outer"]["ann_id"])]
            cats = SWAP_CATEGORIES[cat]
        else:
            masks = [ann.mask(row["file"], row["inner"]["ann_id"])]
            cats = LAYER_CATEGORIES
        orig, masks = prepare(photo, masks)
        prod = pick_products(bank_rows, cats, row["file"])
        if not prod:
            continue
        prod = prod[0]
        prod_img, prod_mask = product_pixels(bank, prod["view"], prod["item"])
        prod_pal = palette(srgb_to_lab(prod_img)[erode(prod_mask, 4)])
        outer_name = row["outer"]["category"] if args.mode == "swap" else prod["category"]
        prompt = (SWAP_PROMPT if args.mode == "swap" else LAYER_PROMPT).format(cat=cat, outer=outer_name)
        torch.cuda.synchronize()
        start = time.perf_counter()
        gen = model([Image.fromarray(orig), Image.fromarray(prod_img)], prompt, [args.seed], orig.shape[1::-1])[0]
        torch.cuda.synchronize()
        sec = time.perf_counter() - start
        gen = np.array(gen.convert("RGB").resize(orig.shape[1::-1]))
        stem = Path(row["file"]).stem
        Image.fromarray(gen).save(out / "raw" / f"{stem}.jpg", quality=92)
        if args.mode == "swap":
            inner, outer = masks
            px = max(2, round(0.01 * max(orig.shape[:2])))
            region = dilate(inner, px)
            x = composite(orig, gen, feather(region, px))
            check = swap_check(orig, x, inner, outer, prod_pal)
            extra = {"inner_share": row["inner_share"], "share_bin": row["share_bin"]}
        else:
            check, jacket = layer_check(orig, gen, masks[0], prod_pal)
            x = composite(orig, gen, feather(dilate(jacket, 2), 3))
            Image.fromarray((jacket * 255).astype(np.uint8)).save(out / "input" / f"{stem}_jacket.png")
            extra = {}
        Image.fromarray(x).save(out / "input" / f"{stem}.jpg", quality=95)
        rec = {"file": row["file"], "mode": args.mode, "inner_category": cat, "product": prod["item"],
               "product_category": prod["category"], "seconds": round(sec, 1), "size": orig.shape[1::-1],
               "approved": not check["reasons"], **check, **extra}
        log.write(json.dumps(rec, ensure_ascii=False, default=float) + "\n")
        log.flush()
        print(f"[{i + 1}/{len(rows)}] {row['file']} approved={rec['approved']} {check['reasons']}", flush=True)


if __name__ == "__main__":
    main()

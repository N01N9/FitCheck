"""평가셋 E-실제숨김: 상의가 다 보이는 실제 사진에 실제 겉옷 조각을 붙여 일부러 가린다.

가려진 부분의 정답은 원본 사진의 실제 픽셀이다(사람이 만든 정답이 아님).
겉옷 조각은 평가용 분할(report)의 겹쳐 입은 사진에서만 가져온다. 학습 데이터에는 쓰지 않는다.

조각 맞추기: 원본 겉옷+이너 영역의 bbox(몸통 상자)를 대상 상의 bbox 에 맞춘다.
가로는 상의 폭의 WIDEN 배로 하되 세로가 상의 높이의 MAX_TALL 배를 넘지 않게 줄이고,
위쪽 끝과 가운데를 맞춘다. 겉옷 마스크는 이너 자리가 비어 있으므로(열린 재킷), 붙이면 가운데로 상의가 보인다.
사람이 입은 사진만 쓴다(하의·신발 주석이 같이 있는 사진). 상품 컷에 재킷을 띄우면 평가가 왜곡된다.
겉옷 조각은 재킷·카디건만 쓴다(긴 코트는 상의 폭에 맞추면 다리까지 덮는다).

출력 (out/)
  <id>_input.jpg     가린 사진 (모델 입력)
  <id>_orig.jpg      원본 (겉옷 벗기기 정답)
  <id>_masks.png     채널 R=붙인 겉옷, G=원래 상의, B=새로 가려진 상의
  cases.jsonl        사진·겉옷 출처, 가시율, 분할
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from unpaired.layered import Annotations, load, share_bin
from unpaired.masks import bbox

WIDEN = (1.05, 1.3)
MAX_TALL = 1.35                      # 붙인 몸통 상자 높이 / 상의 높이
WORN_HINTS = {"pants", "shorts", "skirt", "shoe", "tights, stockings"}
MIN_HIDDEN, MAX_HIDDEN = 0.3, 0.95   # 상의 중 새로 가려진 비율
MIN_VISIBLE_SHARE = 0.05


def h01(text: str) -> float:
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def paste_occluder(photo: np.ndarray, top: np.ndarray, src: np.ndarray, src_outer: np.ndarray,
                   src_inner: np.ndarray, widen: float) -> tuple[np.ndarray, np.ndarray]:
    """src 사진의 겉옷을 photo 의 상의 위에 맞춰 붙인다. (가린 사진, 붙인 겉옷 마스크)."""
    sx0, sy0, sx1, sy1 = bbox(src_outer | src_inner)
    tx0, ty0, tx1, ty1 = bbox(top)
    s = min(widen * (tx1 - tx0) / max(sx1 - sx0, 1), MAX_TALL * (ty1 - ty0) / max(sy1 - sy0, 1))
    cx_s, cx_t = (sx0 + sx1) / 2, (tx0 + tx1) / 2
    m = np.float32([[s, 0, cx_t - s * cx_s], [0, s, ty0 - s * sy0]])
    h, w = top.shape
    warped = cv2.warpAffine(src, m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    alpha = cv2.warpAffine(src_outer.astype(np.float32), m, (w, h), flags=cv2.INTER_LINEAR, borderValue=0)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.8)[..., None]
    out = (photo.astype(np.float32) * (1 - alpha) + warped.astype(np.float32) * alpha).round().astype(np.uint8)
    return out, alpha[..., 0] > 0.5


def build(index_dir: Path, fashionpedia: Path, out: Path, split: str) -> dict:
    ann = Annotations(fashionpedia)
    tops = [r for r in load(index_dir / "solo_tops.jsonl", split)
            if WORN_HINTS & {a["category"] for a in ann.by_file.get(r["file"], [])}]
    donors = load(index_dir / "layered.jsonl", "report")
    donors = [d for d in donors if d["outer"]["category"] in ("jacket", "cardigan")]
    out.mkdir(parents=True, exist_ok=True)
    kept, rejected = [], 0
    with (out / "cases.jsonl").open("w") as fh:
        for row in tops:
            photo = np.array(Image.open(ann.image_dir / row["file"]).convert("RGB"))
            top = ann.mask(row["file"], row["inner"]["ann_id"])
            # 같은 사진이면 언제 돌려도 같은 겉옷 조각·폭이 뽑힌다
            order = sorted(donors, key=lambda d: h01(row["file"] + d["file"]))
            done = False
            for donor in order[:5]:
                src = np.array(Image.open(ann.image_dir / donor["file"]).convert("RGB"))
                widen = WIDEN[0] + (WIDEN[1] - WIDEN[0]) * h01("w" + row["file"] + donor["file"])
                occluded, jacket = paste_occluder(photo, top, src, ann.mask(donor["file"], donor["outer"]["ann_id"]),
                                                  ann.mask(donor["file"], donor["inner"]["ann_id"]), widen)
                hidden = top & jacket
                visible = top & ~jacket
                hidden_frac = hidden.sum() / top.sum()
                share = visible.sum() / max(visible.sum() + jacket.sum(), 1)
                if not (MIN_HIDDEN <= hidden_frac <= MAX_HIDDEN and share >= MIN_VISIBLE_SHARE):
                    continue
                cid = Path(row["file"]).stem
                Image.fromarray(occluded).save(out / f"{cid}_input.jpg", quality=95)
                Image.fromarray(photo).save(out / f"{cid}_orig.jpg", quality=95)
                masks = np.stack([jacket, top, hidden], axis=-1).astype(np.uint8) * 255
                Image.fromarray(masks).save(out / f"{cid}_masks.png")
                rec = {"id": cid, "file": row["file"], "category": row["inner"]["category"],
                       "attributes": row["inner"]["attributes"], "donor": donor["file"],
                       "donor_category": donor["outer"]["category"], "widen": round(widen, 3),
                       "hidden_frac": round(float(hidden_frac), 3), "inner_share": round(float(share), 3),
                       "share_bin": share_bin(float(share)), "split": split}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept.append(rec)
                done = True
                break
            rejected += not done
    summary = {"split": split, "kept": len(kept), "rejected": rejected,
               "share_bin": dict(Counter(r["share_bin"] for r in kept)),
               "category": dict(Counter(r["category"] for r in kept))}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="E-실제숨김 평가셋 만들기")
    p.add_argument("--index-dir", default="data/unpaired/index")
    p.add_argument("--fashionpedia", default="data/fashionpedia")
    p.add_argument("--split", default="hidden_report", choices=["hidden_report", "hidden_tune"])
    p.add_argument("--out", default="data/unpaired/eval/e_real_hidden")
    args = p.parse_args(argv)
    print(json.dumps(build(Path(args.index_dir), Path(args.fashionpedia), Path(args.out), args.split),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

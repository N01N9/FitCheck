"""학습한 검사기: "결과물이 완전한 옷 한 벌의 상품 사진인가" 를 DINOv2 특징 + 로지스틱 회귀로 가린다.

검사기 보정 v1(calibrate)에서 색·모양 규칙이 "보이는 띠 조각"·"겉옷+이너 합본" 을 충분히 못 걸렀다.
사람 라벨 없이 학습한다. 양성 = 은행 상품 사진(실제), 음성 = 은행 상품으로 자동으로 만든 실패 사례:
  strip   상품의 가운데 세로 띠만 남김 (펴지 않고 잘라 붙인 결과)
  half    상품의 위·아래·왼쪽·오른쪽 절반만 남김 (잘린 결과)
  merge   이너 상품 위에 겉옷 상품의 양옆을 덮어 씌움 (블레이저+티 합본)
  pair    상품 두 벌을 한 장에 나란히 놓음
  marker  상품 외곽에 표시용 초록 선을 그림 (zero-shot 의 표시 자국)
  grey    상품 일부를 회색 덩어리로 덮음 (회색 덮기 자국)
선택(best-of-K)과 짝 승인에만 쓰고, 평가(보고)에는 쓰지 않는다.

2026-10-03 결과: 따로 둔 품목에서는 AUROC 0.995(띠 100%, 합본 95% 거부)였지만 klein 시범 출력 320장에서는
평가 라벨(맞는 옷)과의 AUROC 0.63 이었다. 실제 사진으로만 배워 생성 이미지 질감에 약하다.
그래서 아직 검사기(gates)에 넣지 않는다. 양성·음성을 같은 생성기로 한 번 거친 사진으로 맞추는 것이 다음 개선안이다.

사용 (컨테이너 안)
  python -m unpaired.verifier train --bank data/unpaired/bank --out data/unpaired/verifier
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from unpaired.bank import load_usable
from unpaired.pointer import GREY_RGB, MARK_RGB

NEGATIVES = ("strip", "half", "merge", "pair", "marker", "grey")
INNER_GROUP, OUTER_GROUP = "inner", "outer"


def on_white(img: np.ndarray, mask: np.ndarray, size: int = 448) -> Image.Image:
    """마스크 영역만 흰 정사각형 가운데 앉힌다(결과물 형식과 맞춘다)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return Image.new("RGB", (size, size), "white")
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = np.where(mask[y0:y1, x0:x1, None], img[y0:y1, x0:x1], 255).astype(np.uint8)
    side = int(max(x1 - x0, y1 - y0) / 0.86)
    canvas = np.full((side, side, 3), 255, np.uint8)
    oy, ox = (side - (y1 - y0)) // 2, (side - (x1 - x0)) // 2
    canvas[oy:oy + y1 - y0, ox:ox + x1 - x0] = crop
    return Image.fromarray(canvas).resize((size, size), Image.BICUBIC)


def negative(kind: str, img: np.ndarray, mask: np.ndarray, other: tuple[np.ndarray, np.ndarray],
             rng: np.random.Generator) -> Image.Image:
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    if kind == "strip":
        frac = rng.uniform(0.12, 0.35)
        cx = (x0 + x1) / 2 + rng.uniform(-0.05, 0.05) * (x1 - x0)
        keep = np.zeros_like(mask)
        keep[:, int(cx - frac * (x1 - x0) / 2): int(cx + frac * (x1 - x0) / 2)] = True
        return on_white(img, mask & keep)
    if kind == "half":
        keep = np.zeros_like(mask)
        side = rng.integers(4)
        if side == 0:
            keep[: (y0 + y1) // 2] = True
        elif side == 1:
            keep[(y0 + y1) // 2:] = True
        elif side == 2:
            keep[:, : (x0 + x1) // 2] = True
        else:
            keep[:, (x0 + x1) // 2:] = True
        return on_white(img, mask & keep)
    if kind == "merge":
        o_img, o_mask = other
        out, m = img.copy(), mask.copy()
        cover = o_mask.copy()
        cx = w // 2
        gap = int(rng.uniform(0.1, 0.3) * w / 2)
        cover[:, cx - gap: cx + gap] = False  # 열린 앞섶으로 이너가 보인다
        out[cover] = o_img[cover]
        return on_white(out, m | cover)
    if kind == "pair":
        o_img, o_mask = other
        a, b = on_white(img, mask, 448), on_white(o_img, o_mask, 448)
        canvas = Image.new("RGB", (896, 448), "white")
        canvas.paste(a, (0, 0))
        canvas.paste(b, (448, 0))
        return canvas.resize((448, 448)) if rng.random() < 0.5 else canvas
    if kind == "marker":
        out = img.copy()
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, MARK_RGB, max(3, int(0.008 * max(h, w))), lineType=cv2.LINE_AA)
        dil = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
        return on_white(out, dil)
    if kind == "grey":
        out = img.copy()
        gx = int(rng.uniform(x0, x1 - (x1 - x0) * 0.3))
        gw = int(rng.uniform(0.2, 0.45) * (x1 - x0))
        patch = np.zeros_like(mask)
        patch[y0:y1, gx:gx + gw] = True
        out[mask & patch] = GREY_RGB
        return on_white(out, mask)
    raise ValueError(kind)


class Verifier:
    def __init__(self, model, dino):
        self.model, self.dino = model, dino

    @classmethod
    def load(cls, path: Path, dino=None):
        from unpaired.score import Dino

        with (path / "verifier.pkl").open("rb") as fh:
            model = pickle.load(fh)
        return cls(model, dino or Dino())

    def prob_good(self, images: list[Image.Image]) -> np.ndarray:
        return self.model.predict_proba(self.dino(images))[:, 1]


def build_dataset(bank: Path, n_items: int, seed: int = 0):
    rows = load_usable(bank)
    rng = np.random.default_rng(seed)
    rows = [rows[i] for i in rng.permutation(len(rows))[:n_items]]
    outers = [r for r in rows if r["group"] == OUTER_GROUP] or rows
    images, labels, kinds, items = [], [], [], []
    for r in rows:
        img = np.array(Image.open(bank / "front" / f"{r['item']}.jpg").convert("RGB"))
        mask = np.asarray(Image.open(bank / "front_mask" / f"{r['item']}.png")) > 127
        images.append(on_white(img, mask))
        labels.append(1)
        kinds.append("pos")
        items.append(r["item"])
        kind = NEGATIVES[int(rng.integers(len(NEGATIVES)))]
        o = outers[int(rng.integers(len(outers)))] if kind == "merge" else rows[int(rng.integers(len(rows)))]
        o_img = np.array(Image.open(bank / "front" / f"{o['item']}.jpg").convert("RGB"))
        o_mask = np.asarray(Image.open(bank / "front_mask" / f"{o['item']}.png")) > 127
        images.append(negative(kind, img, mask, (o_img, o_mask), rng))
        labels.append(0)
        kinds.append(kind)
        items.append(r["item"])
    return images, np.array(labels), kinds, items


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="완전한 옷 한 벌 판별기")
    p.add_argument("cmd", choices=["train"])
    p.add_argument("--bank", default="data/unpaired/bank")
    p.add_argument("--out", default="data/unpaired/verifier")
    p.add_argument("--items", type=int, default=1600)
    args = p.parse_args(argv)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    from phase0.exp0_edit_models import disable_broken_cudnn
    from unpaired.score import Dino

    disable_broken_cudnn()
    images, y, kinds, items = build_dataset(Path(args.bank), args.items)
    dino = Dino()
    feats = np.concatenate([dino(images[i:i + 32]) for i in range(0, len(images), 32)])
    # 같은 품목의 양성·음성이 학습/검증에 나뉘지 않게 품목 단위로 나눈다
    uniq = sorted(set(items))
    held = set(uniq[:: 5])
    test = np.array([it in held for it in items])
    model = LogisticRegression(C=1.0, max_iter=2000).fit(feats[~test], y[~test])
    prob = model.predict_proba(feats[test])[:, 1]
    yt, kt = y[test], np.array(kinds)[test]
    report = {"train": int((~test).sum()), "test": int(test.sum()), "auroc": round(float(roc_auc_score(yt, prob)), 4),
              "pos_pass@0.5": round(float((prob[yt == 1] >= 0.5).mean()), 3),
              "neg_reject@0.5": {k: round(float((prob[kt == k] < 0.5).mean()), 3) for k in NEGATIVES if (kt == k).any()}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "verifier.pkl").open("wb") as fh:
        pickle.dump(model, fh)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

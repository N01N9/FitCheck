"""정답 토큰별 손실 가중치: 입력에서 안 보였던 곳의 무늬·글자는 채점하지 않는다(보고서 원칙 ④).

모델이 가려져 안 보인 로고·글자를 "지어내는 법" 을 배우지 않게 한다. 정답 이미지는 고치지 않는다.
  1.0   입력에서 보였던 곳, 배경(깨끗한 흰 배경은 늘 배워야 한다)
  0.3   가려졌지만 무지(이어 그려도 되는 곳)
  0.0   가려졌고 무늬·글자·그래픽이 있는 곳

이너 추출(겉옷 아래 이너): 겉옷 앞섶 사이로 보이는 것은 상품 가운데 세로 띠라고 본다. 띠 폭 =
상품 옷 폭 × (사진에서 보이는 이너 비율, visible_width_ratio) + 여유. 정확한 대응은 모르므로 근사다.
겉옷 벗기기: 겉옷에 가려졌던 영역은 입력과 정답이 같은 좌표라서 정확하다.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from core.color import srgb_to_lab

TOKEN = 16
PLAIN_WEIGHT = 0.3
STD_DETAIL = 8.0    # 토큰 안 밝기(L*) 표준편차. 넘으면 무늬·글자로 본다
EDGE_DETAIL = 6.0   # 토큰 안 라플라시안 절댓값 평균
BAND_MARGIN = 0.06  # 보이는 띠 양옆 여유(상품 옷 폭 대비)


def detail_tokens(img: np.ndarray, mask: np.ndarray | None = None, inset: int = 4) -> np.ndarray:
    """(H/16, W/16) 불리언: 무늬·글자처럼 잔무늬가 있는 토큰.

    옷 테두리(배경과의 경계)는 무늬로 치지 않는다. 테두리까지 0 으로 두면 가려진 소매·밑단 모양을 못 배운다.
    그래서 mask 를 inset 만큼 깎은 안쪽 픽셀로만 판정하고, 안쪽 픽셀이 토큰의 30% 미만이면 무늬가 아니라고 본다.
    """
    lab_l = srgb_to_lab(img)[..., 0].astype(np.float32)
    h, w = lab_l.shape
    gh, gw = h // TOKEN, w // TOKEN
    if mask is None:
        mask = np.ones((h, w), bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1))
    inner = cv2.erode(mask.astype(np.uint8), k).astype(bool)[: gh * TOKEN, : gw * TOKEN]
    lap = np.abs(cv2.Laplacian(lab_l, cv2.CV_32F, ksize=3))[: gh * TOKEN, : gw * TOKEN]
    val = lab_l[: gh * TOKEN, : gw * TOKEN]
    shape = (gh, TOKEN, gw, TOKEN)
    m = inner.reshape(shape).astype(np.float32)
    n = m.sum(axis=(1, 3))
    safe = np.maximum(n, 1)
    mean = (val.reshape(shape) * m).sum(axis=(1, 3)) / safe
    var = ((val.reshape(shape) - mean[:, None, :, None]) ** 2 * m).sum(axis=(1, 3)) / safe
    edge = (lap.reshape(shape) * m).sum(axis=(1, 3)) / safe
    enough = n >= 0.3 * TOKEN * TOKEN
    return enough & ((np.sqrt(var) > STD_DETAIL) | (edge > EDGE_DETAIL))


def to_tokens(mask: np.ndarray, frac: float = 0.5) -> np.ndarray:
    h, w = mask.shape
    gh, gw = h // TOKEN, w // TOKEN
    return mask[: gh * TOKEN, : gw * TOKEN].reshape(gh, TOKEN, gw, TOKEN).mean(axis=(1, 3)) >= frac


def weight_image(tokens: np.ndarray) -> Image.Image:
    """토큰 가중치(0~1) → 토큰 하나가 16px 인 L 이미지(train_lora.weight_tokens 가 다시 줄인다)."""
    big = np.kron(tokens, np.ones((TOKEN, TOKEN), np.float32))
    return Image.fromarray((big * 255).round().astype(np.uint8), "L")


def visible_width_ratio(visible: np.ndarray, torso: np.ndarray) -> float:
    """사진에서 보이는 이너의 비율: 상의 높이 범위 안에서 (보이는 이너 면적 / 이너+겉옷 면적).

    bbox 폭으로 재면 옆구리·밑단의 작은 조각까지 합쳐 거의 다 보인다고 잘못 잰다(2026-10-03 미리보기).
    겉옷 소매까지 분모에 들어가 실제보다 덜 보인다고 잡는 쪽(보수적, 지어내기 방지 쪽)으로 틀린다.
    """
    rows = np.nonzero(visible.any(axis=1))[0]
    if len(rows) == 0:
        return 0.0
    band = torso[rows.min(): rows.max() + 1]
    return float(visible[rows.min(): rows.max() + 1].sum() / max(band.sum(), 1))


def extract_weights(product: np.ndarray, product_mask: np.ndarray, ratio: float,
                    size: tuple[int, int] = (768, 768)) -> Image.Image:
    """겉옷 아래 이너 정답(은행 상품)의 토큰 가중치. ratio = visible_width_ratio."""
    img = np.asarray(Image.fromarray(product).resize(size, Image.BILINEAR))
    mask = np.asarray(Image.fromarray(product_mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127
    cols = np.nonzero(mask.any(axis=0))[0]
    band = np.zeros_like(mask)
    if len(cols):
        gx0, gx1 = cols.min(), cols.max() + 1
        half = (gx1 - gx0) * (min(1.0, ratio) / 2 + BAND_MARGIN)
        cx = (gx0 + gx1) / 2
        band[:, max(0, int(cx - half)): int(cx + half)] = True
    garment = to_tokens(mask, 0.25)
    hidden = garment & ~to_tokens(band, 0.5)
    detail = detail_tokens(img, mask)
    w = np.ones(garment.shape, np.float32)
    w[hidden] = PLAIN_WEIGHT
    w[hidden & detail] = 0.0
    return weight_image(w)


def peel_weights(target: np.ndarray, hidden: np.ndarray) -> Image.Image:
    """겉옷 벗기기 정답(원본 사진)의 토큰 가중치. hidden = 겉옷에 가려졌던 영역(정답과 같은 좌표)."""
    tok_hidden = to_tokens(hidden, 0.25)
    detail = detail_tokens(target, hidden)
    w = np.ones(tok_hidden.shape, np.float32)
    w[tok_hidden] = PLAIN_WEIGHT
    w[tok_hidden & detail] = 0.0
    return weight_image(w)

"""대상 옷을 말 대신 사진 위 표시로 가리킨다(set-of-marks).

실험 0 에서 "재킷 안의 티셔츠" 처럼 말로만 지정하면 모델이 겉옷을 내놓거나 둘을 합쳤다.
그래서 사진 전체를 그대로 주되, 대상 옷의 보이는 영역에 외곽선이나 색을 입혀 가리킨다.

  outline    대상 외곽선만 그린다
  fill       대상을 반투명 색으로 칠하고 외곽선을 그린다
  grey_outer 겉옷 픽셀을 회색으로 덮는다 (보고서 P4)
  dim_crop   대상 주변을 잘라 내고 대상 밖은 어둡게 한 보조 참조 이미지 (보고서 P2)
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from unpaired.masks import bbox

MARK_RGB = (0, 255, 0)  # 선명한 초록. 옷 색으로는 드물어 표시와 옷이 헷갈리지 않는다
GREY_RGB = (128, 128, 128)


def _line_width(shape) -> int:
    return max(2, round(0.006 * max(shape[:2])))


def outline(img: np.ndarray, mask: np.ndarray, color=MARK_RGB) -> np.ndarray:
    out = img.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, color, _line_width(img.shape), lineType=cv2.LINE_AA)
    return out


def fill(img: np.ndarray, mask: np.ndarray, color=MARK_RGB, alpha: float = 0.35) -> np.ndarray:
    out = img.astype(np.float32)
    out[mask] = (1 - alpha) * out[mask] + alpha * np.asarray(color, np.float32)
    return outline(out.round().astype(np.uint8), mask, color)


def grey_outer(img: np.ndarray, outer: np.ndarray, target: np.ndarray, color=MARK_RGB) -> np.ndarray:
    out = img.copy()
    out[outer & ~target] = GREY_RGB
    return outline(out, target, color)


def dim_crop(img: np.ndarray, target: np.ndarray, pad: float = 0.15, factor: float = 0.3,
             min_aspect: float = 0.4) -> np.ndarray:
    """대상 bbox 를 pad 만큼 넓혀 자르고, 대상 밖 픽셀은 factor 배로 어둡게 한다.

    겉옷 사이로 아주 조금 보이는 이너는 크롭이 가는 띠(예: 48×384)가 되어 klein 입력 제한(가로·세로 64px 이상)에
    걸린다. 그래서 가로/세로 비율이 [min_aspect, 1/min_aspect] 안에 들도록 짧은 쪽을 주변(어둡게 된 영역)으로 넓힌다.
    """
    box = bbox(target)
    if box is None:
        raise ValueError("빈 마스크")
    x0, y0, x1, y1 = box
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    h, w = target.shape
    x0, y0, x1, y1 = max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)
    cw, ch = x1 - x0, y1 - y0
    if cw < min_aspect * ch:
        grow = int(min_aspect * ch) - cw
        x0, x1 = max(0, x0 - grow // 2), min(w, x1 + grow - grow // 2)
    elif ch < min_aspect * cw:
        grow = int(min_aspect * cw) - ch
        y0, y1 = max(0, y0 - grow // 2), min(h, y1 + grow - grow // 2)
    out = img.astype(np.float32)
    out[~target] *= factor
    return out[y0:y1, x0:x1].round().astype(np.uint8)


def to_pil(arr: np.ndarray, multiple: int = 16, max_side: int = 1024) -> Image.Image:
    """모델 입력 크기(긴 변 max_side, multiple 의 배수)로 맞춘 PIL 이미지."""
    im = Image.fromarray(arr)
    im.thumbnail((max_side, max_side), Image.LANCZOS)
    w, h = (im.width // multiple) * multiple, (im.height // multiple) * multiple
    return im.crop((0, 0, max(w, multiple), max(h, multiple)))


def render(variant: str, img: np.ndarray, target: np.ndarray, outer: np.ndarray | None) -> list[Image.Image]:
    """변형 이름에 맞는 참조 이미지 목록을 만든다. 첫 장은 항상 사진 전체다."""
    if variant == "text":
        return [to_pil(img)]
    if variant == "outline":
        return [to_pil(outline(img, target))]
    if variant == "fill":
        return [to_pil(fill(img, target))]
    if variant == "grey_outer":
        if outer is None:
            raise ValueError("grey_outer 는 겉옷 마스크가 필요합니다")
        return [to_pil(grey_outer(img, outer, target))]
    if variant == "outline_dimcrop":
        return [to_pil(outline(img, target)), to_pil(dim_crop(img, target))]
    raise ValueError(f"모르는 변형: {variant}")


VARIANTS = ("text", "outline", "fill", "grey_outer", "outline_dimcrop")

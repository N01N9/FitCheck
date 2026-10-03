"""COCO 마스크(폴리곤, RLE)를 numpy 불리언 배열로 바꾼다. pycocotools 없이 동작한다."""

from __future__ import annotations

import cv2
import numpy as np


def rle_counts_from_string(s: str) -> list[int]:
    """COCO 압축 RLE 문자열을 run 길이 목록으로 푼다(maskApi.c 의 rleFrString 과 같은 규칙)."""
    counts: list[int] = []
    p = 0
    while p < len(s):
        x, k, more = 0, 0, True
        while more:
            c = ord(s[p]) - 48
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            p += 1
            k += 1
            if not more and (c & 0x10):
                x |= -1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def rle_decode(counts: list[int], h: int, w: int) -> np.ndarray:
    flat = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for c in counts:
        if val:
            flat[pos:pos + c] = True
        pos += c
        val = not val
    # COCO RLE 는 열 우선(column-major) 순서다
    return flat.reshape(w, h).T


def decode(seg, h: int, w: int) -> np.ndarray:
    if isinstance(seg, list):
        mask = np.zeros((h, w), dtype=np.uint8)
        polys = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in seg if len(p) >= 6]
        if polys:
            cv2.fillPoly(mask, [np.round(p).astype(np.int32) for p in polys], 1)
        return mask.astype(bool)
    counts = seg["counts"]
    if isinstance(counts, str):
        counts = rle_counts_from_string(counts)
    sh, sw = seg.get("size", (h, w))
    return rle_decode(list(counts), sh, sw)


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1), x1·y1 은 포함하지 않는 끝."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def convex_hull(mask: np.ndarray) -> np.ndarray:
    pts = cv2.findNonZero(mask.astype(np.uint8))
    out = np.zeros(mask.shape, dtype=np.uint8)
    if pts is not None:
        cv2.fillConvexPoly(out, cv2.convexHull(pts), 1)
    return out.astype(bool)

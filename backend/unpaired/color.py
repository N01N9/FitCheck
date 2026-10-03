"""CIELAB 색차(CIEDE2000)를 픽셀 배열 단위로 계산하고, 옷의 색 묶음(팔레트)을 뽑는다.

core.color.delta_e2000 은 색 하나씩 계산하므로, 여기서는 같은 공식을 numpy 로 벡터화했다.
kL 을 1 보다 크게 주면 밝기 차이를 덜 셈한다. 실내 착용 사진과 스튜디오 상품 사진은 조명이 달라
밝기만 다른 경우가 많아서, 검사에서는 kL=2 를 "조명 보정 ΔE00" 으로 쓴다.
"""

from __future__ import annotations

import cv2
import numpy as np


def delta_e2000(lab1, lab2, kL: float = 1.0, kC: float = 1.0, kH: float = 1.0) -> np.ndarray:
    lab1 = np.asarray(lab1, dtype=np.float64)
    lab2 = np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    c_bar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(c_bar**7 / (c_bar**7 + 25.0**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    zero = c1p * c2p == 0
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(zero, 0.0, dh)
    dL, dC = L2 - L1, c2p - c1p
    dH = 2 * np.sqrt(c1p * c2p) * np.sin(np.radians(dh / 2))
    L_bar, C_bar = (L1 + L2) / 2, (c1p + c2p) / 2
    h_sum = h1p + h2p
    h_bar = np.where(np.abs(h1p - h2p) > 180, np.where(h_sum < 360, (h_sum + 360) / 2, (h_sum - 360) / 2), h_sum / 2)
    h_bar = np.where(zero, h_sum, h_bar)
    t = (1 - 0.17 * np.cos(np.radians(h_bar - 30)) + 0.24 * np.cos(np.radians(2 * h_bar))
         + 0.32 * np.cos(np.radians(3 * h_bar + 6)) - 0.20 * np.cos(np.radians(4 * h_bar - 63)))
    d_theta = 30 * np.exp(-(((h_bar - 275) / 25) ** 2))
    r_c = 2 * np.sqrt(C_bar**7 / (C_bar**7 + 25.0**7))
    s_l = 1 + 0.015 * (L_bar - 50) ** 2 / np.sqrt(20 + (L_bar - 50) ** 2)
    s_c = 1 + 0.045 * C_bar
    s_h = 1 + 0.015 * C_bar * t
    r_t = -np.sin(np.radians(2 * d_theta)) * r_c
    tl, tc, th = dL / (kL * s_l), dC / (kC * s_c), dH / (kH * s_h)
    return np.sqrt(np.maximum(tl**2 + tc**2 + th**2 + r_t * tc * th, 0.0))


def chroma(lab) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    return np.hypot(lab[..., 1], lab[..., 2])


def palette(lab_pixels: np.ndarray, k: int = 4, max_samples: int = 20000, seed: int = 0):
    """Lab 픽셀(N,3)을 k 개 색으로 묶는다. (중심 (k',3), 비율 (k',)) — 비율 큰 순서, 빈 묶음은 뺀다."""
    px = np.asarray(lab_pixels, dtype=np.float32).reshape(-1, 3)
    if len(px) == 0:
        raise ValueError("픽셀이 없습니다")
    if len(px) > max_samples:
        px = px[np.random.default_rng(seed).choice(len(px), max_samples, replace=False)]
    k = max(1, min(k, len(np.unique(px.round(1), axis=0))))
    cv2.setRNGSeed(seed)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, centers = cv2.kmeans(px, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    weights = np.bincount(labels.ravel(), minlength=k) / len(px)
    order = np.argsort(-weights)
    keep = weights[order] > 0
    return centers[order][keep].astype(np.float64), weights[order][keep]


def nearest(lab_pixels: np.ndarray, centers: np.ndarray, kL: float = 2.0) -> np.ndarray:
    """픽셀마다 팔레트 중 가장 가까운 색까지의 ΔE00."""
    px = np.asarray(lab_pixels, dtype=np.float64).reshape(-1, 1, 3)
    return delta_e2000(px, centers.reshape(1, -1, 3), kL=kL).min(axis=1)


def palette_distance(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray], kL: float = 2.0) -> float:
    """a 의 각 색이 b 에서 얼마나 가까운 색을 찾는지(비율 가중 평균 ΔE00). 방향이 있다."""
    centers_a, weights_a = a
    return float((nearest(centers_a, b[0], kL) * weights_a).sum())

"""옷 누끼 이미지에서 대표색을 뽑고 한국어 색 이름을 붙인다. 색 차이는 CIELAB ΔE로 잰다.

색 이름은 phase0.garment_schema.COLORS 와 같은 목록을 쓴다(태깅 결과와 비교 가능하게).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

# 색 이름별 기준 sRGB. 새 색을 추가할 때는 garment_schema.COLORS 에도 넣는다.
REFERENCE_COLORS = {
    "블랙": (22, 22, 24),
    "화이트": (246, 246, 244),
    "아이보리": (240, 233, 212),
    "그레이": (150, 150, 152),
    "차콜": (66, 66, 70),
    "네이비": (32, 42, 70),
    "블루": (40, 85, 180),
    "스카이블루": (140, 190, 228),
    "베이지": (212, 192, 158),
    "브라운": (112, 72, 42),
    "카키": (108, 108, 68),
    "그린": (40, 120, 64),
    "민트": (160, 220, 200),
    "레드": (200, 32, 42),
    "버건디": (110, 22, 42),
    "핑크": (240, 162, 184),
    "퍼플": (112, 62, 150),
    "옐로우": (240, 210, 52),
    "오렌지": (238, 130, 40),
}


def srgb_to_lab(rgb) -> np.ndarray:
    """sRGB(0~255, (...,3)) → CIELAB(D65)."""
    c = np.asarray(rgb, dtype=np.float64) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array([[0.4124564, 0.3575761, 0.1804375],
                  [0.2126729, 0.7151522, 0.0721750],
                  [0.0193339, 0.1191920, 0.9503041]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz / (3 * (6 / 29) ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def delta_e2000(lab1, lab2) -> float:
    """CIEDE2000 색차. 사람 눈의 색 차이 인지에 가장 가까운 표준 공식."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    c_bar = (math.hypot(a1, b1) + math.hypot(a2, b2)) / 2
    g = 0.5 * (1 - math.sqrt(c_bar**7 / (c_bar**7 + 25**7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = math.hypot(a1p, b1), math.hypot(a2p, b2)
    h1p = math.degrees(math.atan2(b1, a1p)) % 360
    h2p = math.degrees(math.atan2(b2, a2p)) % 360
    dL, dC = L2 - L1, c2p - c1p
    if c1p * c2p == 0:
        dh = 0.0
    elif abs(h2p - h1p) <= 180:
        dh = h2p - h1p
    else:
        dh = h2p - h1p - 360 if h2p > h1p else h2p - h1p + 360
    dH = 2 * math.sqrt(c1p * c2p) * math.sin(math.radians(dh / 2))
    L_bar, C_bar = (L1 + L2) / 2, (c1p + c2p) / 2
    if c1p * c2p == 0:
        h_bar = h1p + h2p
    elif abs(h1p - h2p) <= 180:
        h_bar = (h1p + h2p) / 2
    else:
        h_bar = (h1p + h2p + 360) / 2 if h1p + h2p < 360 else (h1p + h2p - 360) / 2
    t = (1 - 0.17 * math.cos(math.radians(h_bar - 30)) + 0.24 * math.cos(math.radians(2 * h_bar))
         + 0.32 * math.cos(math.radians(3 * h_bar + 6)) - 0.20 * math.cos(math.radians(4 * h_bar - 63)))
    d_theta = 30 * math.exp(-(((h_bar - 275) / 25) ** 2))
    r_c = 2 * math.sqrt(C_bar**7 / (C_bar**7 + 25**7))
    s_l = 1 + 0.015 * (L_bar - 50) ** 2 / math.sqrt(20 + (L_bar - 50) ** 2)
    s_c = 1 + 0.045 * C_bar
    s_h = 1 + 0.015 * C_bar * t
    r_t = -math.sin(math.radians(2 * d_theta)) * r_c
    return math.sqrt((dL / s_l) ** 2 + (dC / s_c) ** 2 + (dH / s_h) ** 2 + r_t * (dC / s_c) * (dH / s_h))


_REF_LAB = {name: srgb_to_lab(rgb) for name, rgb in REFERENCE_COLORS.items()}


def color_name(rgb) -> str:
    lab = srgb_to_lab(rgb)
    return min(_REF_LAB, key=lambda name: delta_e2000(lab, _REF_LAB[name]))


def to_hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*(int(round(v)) for v in rgb))


@dataclass
class ColorShare:
    hex: str
    name: str
    ratio: float


def _kmeans(points: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    # k-means++ 초기화
    centers = [points[rng.integers(len(points))]]
    for _ in range(1, k):
        d = np.min(((points[:, None] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
        if d.sum() == 0:
            break
        centers.append(points[rng.choice(len(points), p=d / d.sum())])
    centers = np.array(centers)
    for _ in range(iters):
        labels = np.argmin(((points[:, None] - centers[None]) ** 2).sum(-1), axis=1)
        new = np.array([points[labels == i].mean(0) if np.any(labels == i) else centers[i] for i in range(len(centers))])
        if np.allclose(new, centers):
            break
        centers = new
    return centers, labels


def dominant_colors(img: Image.Image, k: int = 4, min_ratio: float = 0.08, merge_de: float = 8.0,
                    max_pixels: int = 20000) -> list[ColorShare]:
    """투명 배경을 뺀 옷 픽셀의 대표색. 비율이 큰 순서. 비슷한 색(ΔE < merge_de)은 합친다."""
    rgba = np.asarray(img.convert("RGBA"))
    pixels = rgba[rgba[..., 3] > 128][:, :3].astype(np.float64)
    if len(pixels) == 0:
        return []
    if len(pixels) > max_pixels:
        pixels = pixels[np.random.default_rng(0).choice(len(pixels), max_pixels, replace=False)]
    lab = srgb_to_lab(pixels)
    centers, labels = _kmeans(lab, min(k, len(np.unique(pixels, axis=0))))
    groups = []  # [lab_center, count, rgb_sum]
    for i, center in enumerate(centers):
        mask = labels == i
        count = int(mask.sum())
        if not count:
            continue
        for g in groups:
            if delta_e2000(g[0], center) < merge_de:
                g[0] = (g[0] * g[1] + center * count) / (g[1] + count)
                g[1] += count
                g[2] += pixels[mask].sum(0)
                break
        else:
            groups.append([center, count, pixels[mask].sum(0)])
    total = len(pixels)
    result = []
    for _, count, rgb_sum in sorted(groups, key=lambda g: -g[1]):
        ratio = count / total
        if ratio < min_ratio:
            continue
        rgb = rgb_sum / count
        result.append(ColorShare(to_hex(rgb), color_name(rgb), round(ratio, 3)))
    return result


def primary_and_secondary(img: Image.Image) -> tuple[str | None, list[str]]:
    """태깅 스키마의 primary_color / secondary_colors 형태로 돌려준다.
    대표색 3개 이상이 각각 20% 넘게 섞여 있으면 멀티로 본다."""
    shares = dominant_colors(img)
    if not shares:
        return None, []
    if sum(1 for s in shares if s.ratio >= 0.2) >= 3:
        return "멀티", []
    names = []
    for s in shares:
        if s.name not in names:
            names.append(s.name)
    return names[0], names[1:3]

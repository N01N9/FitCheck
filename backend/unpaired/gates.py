"""생성 결과(상품 사진)를 사람 없이 검사한다.

입력 사진의 대상 옷(보이는 부분)과 겉옷의 실제 픽셀을 기준으로, 결과물이
  1) 형식: 단색 배경 위에 옷 한 벌만 있는가
  2) 색: 대상 옷의 보이는 색과 맞는가 (조명 보정 ΔE00, 채도 비율 — 남색→검정 같은 오류를 잡는다)
  3) 겉옷 섞임: 결과물 픽셀 중 겉옷 색에 더 가까운 비율이 얼마인가
를 수치로 낸다. 임계값은 tune 분할과 음성 대조군으로 정하고, 보고용 분할에서는 바꾸지 않는다.

배경 분리: 결과물의 전경 마스크가 주어지면(예: BiRefNet) 그것을 쓰고, 없으면 테두리 색과의
차이로 분리한다. 흰 옷이 흰 배경 위에 있으면 색 방식이 약하므로 전경 마스크를 주는 편이 낫다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from core.color import srgb_to_lab
from unpaired.color import chroma, delta_e2000, nearest, palette, palette_distance

# 초기값. tune 분할로 다시 정한다
THRESHOLDS = {
    "border_std_max": 6.0,      # 테두리 Lab 표준편차(채널 평균). 크면 단색 배경이 아니다
    "fg_frac_min": 0.06,
    "fg_frac_max": 0.92,
    "components_max": 2,        # 전경 면적 3% 이상인 덩어리 수
    "palette_dist_max": 12.0,   # 대상 색 → 결과물 색 (kL=2)
    "extra_color_max": 15.0,    # 결과물 색 → 대상 색. 대상에 없던 색이 많이 생기면 커진다
    "chroma_ratio_min": 0.5,    # 대상이 유채색(C*>8)일 때만 본다
    "chroma_ratio_max": 2.0,
    "outer_frac_max": 0.15,
    "same_color_sep": 10.0,     # 대상·겉옷 팔레트가 이보다 가까우면 겉옷 섞임은 판정하지 않는다
}


@dataclass
class GateResult:
    passed: bool
    scores: dict = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)


def border_band(shape, frac: float = 0.03) -> np.ndarray:
    h, w = shape[:2]
    b = max(2, round(frac * min(h, w)))
    band = np.zeros((h, w), dtype=bool)
    band[:b], band[-b:], band[:, :b], band[:, -b:] = True, True, True, True
    return band


def foreground_by_color(lab: np.ndarray, threshold: float = 8.0) -> np.ndarray:
    band = border_band(lab.shape)
    bg = np.median(lab[band], axis=0)
    fg = delta_e2000(lab, bg) > threshold
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, k)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    return fill_holes(fg.astype(bool))


def fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    flood = np.zeros((h + 2, w + 2), dtype=np.uint8)
    flood[1:-1, 1:-1] = mask
    cv2.floodFill(flood, None, (0, 0), 2)
    return (flood[1:-1, 1:-1] != 2)


def components(mask: np.ndarray, min_frac: float = 0.03) -> int:
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    total = max(int(mask.sum()), 1)
    return int(sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_frac * total))


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    eroded = cv2.erode(mask.astype(np.uint8), k).astype(bool)
    # 너무 얇아 다 깎이면 원래 마스크를 쓴다
    return eroded if eroded.sum() >= 50 else mask


def check(photo: np.ndarray, target: np.ndarray, outer: np.ndarray | None, result: np.ndarray,
          result_fg: np.ndarray | None = None, thresholds: dict | None = None) -> GateResult:
    """photo: 원본 사진 RGB, target/outer: 원본 기준 마스크, result: 결과물 RGB."""
    t = {**THRESHOLDS, **(thresholds or {})}
    s: dict = {}
    reasons: list[str] = []
    res_lab = srgb_to_lab(result)
    fg = result_fg if result_fg is not None else foreground_by_color(res_lab)
    band = border_band(result.shape)
    s["border_std"] = float(res_lab[band].std(axis=0).mean())
    s["fg_frac"] = float(fg.mean())
    s["components"] = components(fg)
    if s["border_std"] > t["border_std_max"]:
        reasons.append("배경이 단색이 아님")
    if not t["fg_frac_min"] <= s["fg_frac"] <= t["fg_frac_max"]:
        reasons.append("옷 크기가 이상함")
    if s["components"] > t["components_max"]:
        reasons.append("옷이 여러 덩어리")
    if fg.sum() < 50:
        return GateResult(False, s, reasons or ["옷을 찾지 못함"])

    margin = max(1, round(0.004 * max(photo.shape[:2])))
    photo_lab = srgb_to_lab(photo)
    tgt_px = photo_lab[erode(target, margin)]
    out_px = res_lab[erode(fg, margin)]
    tgt_pal = palette(tgt_px)
    out_pal = palette(out_px)
    s["palette_dist"] = palette_distance(tgt_pal, out_pal)
    s["extra_color"] = palette_distance(out_pal, tgt_pal)
    tgt_c, out_c = float(np.median(chroma(tgt_px))), float(np.median(chroma(out_px)))
    s["target_chroma"], s["result_chroma"] = tgt_c, out_c
    s["chroma_ratio"] = out_c / max(tgt_c, 1e-6)
    if s["palette_dist"] > t["palette_dist_max"]:
        reasons.append("대상 옷 색이 결과물에 없음")
    if s["extra_color"] > t["extra_color_max"]:
        reasons.append("대상에 없던 색이 많음")
    if tgt_c > 8 and not t["chroma_ratio_min"] <= s["chroma_ratio"] <= t["chroma_ratio_max"]:
        reasons.append("채도가 크게 달라짐")

    if outer is not None and outer.sum() >= 50:
        out_pal_outer = palette(photo_lab[erode(outer & ~target, margin)])
        s["palette_sep"] = palette_distance(tgt_pal, out_pal_outer)
        sample = out_px if len(out_px) <= 20000 else out_px[np.random.default_rng(0).choice(len(out_px), 20000, replace=False)]
        d_t = nearest(sample, tgt_pal[0])
        d_o = nearest(sample, out_pal_outer[0])
        s["outer_frac"] = float(np.mean(d_o + 3.0 < d_t))
        s["same_color"] = s["palette_sep"] < t["same_color_sep"]
        if not s["same_color"] and s["outer_frac"] > t["outer_frac_max"]:
            reasons.append("겉옷 색이 섞임")
    return GateResult(not reasons, s, reasons)


def score(result: GateResult) -> float:
    """여러 장 중 하나를 고를 때 쓰는 점수(작을수록 좋음). 통과한 것이 항상 앞선다."""
    s = result.scores
    value = s.get("palette_dist", 99) + 0.5 * s.get("extra_color", 99) + 40 * s.get("outer_frac", 0)
    return value + (0 if result.passed else 1000)

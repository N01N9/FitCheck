"""결과물 옷의 색을 사진 속 보이는 대상 옷의 색에 맞춘다(그림은 다시 그리지 않고 색만 옮긴다).

v2 엄격 판정 실패의 가장 큰 덩어리가 색(진홍→쨍한 빨강, 연한 데님→남색, 회색→검정)이었다. 편집 모델이 매번 새로
그리는 한 색은 흔들리므로, 만든 뒤에 사진 쪽 색으로 끌어당긴다.

방법: 두 쪽 옷 픽셀을 Lab 로 바꿔 k 개 색 묶음(palette)을 구하고, 결과물 묶음마다 가장 가까운 사진 묶음을 짝지어
(묶음 중심 차이)를 그 묶음에 속한 픽셀에 부드럽게 더한다. L(밝기)은 묶음 평균만 옮기고 픽셀 사이의 명암(주름·그림자)은
그대로 둔다. 배경은 건드리지 않는다.
"""

from __future__ import annotations

import cv2
import numpy as np



def lab_to_srgb(lab: np.ndarray) -> np.ndarray:
    """to_lab 의 역: OpenCV 의 Lab(8비트 스케일) 변환을 거친다."""
    l8 = np.empty_like(lab, dtype=np.float32)
    l8[..., 0] = lab[..., 0] * 255 / 100
    l8[..., 1:] = lab[..., 1:] + 128
    return cv2.cvtColor(np.clip(l8, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def to_lab(img: np.ndarray) -> np.ndarray:
    l8 = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    l8[..., 0] *= 100 / 255
    l8[..., 1:] -= 128
    return l8


def fix(photo: np.ndarray, target: np.ndarray, result: np.ndarray, result_fg: np.ndarray, k: int = 4,
        strength: float = 1.0, max_shift: float = 40.0) -> tuple[np.ndarray, dict]:
    """result 의 옷(result_fg) 색을 photo 의 대상(target) 색에 맞춘 이미지와 진단값을 돌려준다."""
    core = cv2.erode(target.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    src = to_lab(photo)[core if core.sum() >= 200 else target]  # 경계 픽셀은 겉옷 색이 섞여 있어 깎는다
    lab = to_lab(result)
    dst = lab[result_fg]
    if len(src) < 50 or len(dst) < 50:
        return result, {"skipped": True}
    ps = palette_centers(src, k)
    pd = palette_centers(dst, k)
    # 결과물 묶음마다 가장 가까운 사진 묶음(색상·채도 위주, 밝기는 절반 무게)
    w = np.array([0.5, 1.0, 1.0])
    pair = np.argmin((((pd[:, None] - ps[None]) * w) ** 2).sum(-1), axis=1)
    shift = np.clip(ps[pair] - pd, -max_shift, max_shift) * strength
    # 픽셀마다 결과물 묶음에 부드럽게 속하게 한다(가까운 묶음일수록 큰 무게)
    d = (((dst[:, None] - pd[None]) * w) ** 2).sum(-1)
    soft = np.exp(-(d - d.min(1, keepdims=True)) / 50.0)
    soft /= soft.sum(1, keepdims=True)
    lab2 = lab.copy()
    lab2[result_fg] = dst + soft @ shift
    out = result.copy()
    out[result_fg] = lab_to_srgb(lab2)[result_fg]
    return out, {"shift": np.round(shift, 1).tolist(), "src": np.round(ps, 1).tolist(), "dst": np.round(pd, 1).tolist()}


def palette_centers(lab_px: np.ndarray, k: int) -> np.ndarray:
    px = lab_px[np.random.default_rng(0).choice(len(lab_px), min(len(lab_px), 20000), replace=False)].astype(np.float32)
    k = min(k, max(1, len(px) // 50))
    _, _, centers = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5), 3,
                                    cv2.KMEANS_PP_CENTERS)
    return centers

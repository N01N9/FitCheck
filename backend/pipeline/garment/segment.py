"""2단계. 아이템 찾기·분할(segment).

사진에서 패션 아이템마다 거친 마스크를 만든다. 사람 몸·손·머리카락·옷걸이·배경은 뺀다.

backend
  sam3      : SAM 3 에 텍스트 프롬프트를 넣어 개념 단위로 분할한다 (1순위)
  heuristic : 배경색 추정 + 연결 요소. SAM 3 가 없을 때 파이프라인을 끝까지 돌려 보기 위한 임시.
  fake      : 테스트용. 미리 정한 사각형을 돌려준다.

공통 후처리(`postprocess`)는 어떤 backend 를 쓰든 똑같이 적용한다.
  - 너무 작은 조각은 버린다
  - 겹치는 후보는 합친다 (IoU 또는 포함 관계)
  - 신발 한 켤레는 한 아이템으로 묶는다
  - case1 힌트가 오면 하나로 합쳐 여러 개로 쪼개지지 않게 한다
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from pipeline.garment.types import CATEGORIES, UNKNOWN, SceneInfo

# SAM 3 에 넣을 텍스트 프롬프트. 한국어 분류 -> 영어 개념어.
PROMPT_MAP = {
    "상의": "top, shirt, t-shirt, blouse, sweater",
    "하의": "pants, trousers, jeans, skirt, shorts",
    "아우터": "jacket, coat, outerwear, blazer, cardigan",
    "원피스": "dress, one-piece dress",
    "신발": "shoes, sneakers, boots",
    "가방": "bag, handbag, backpack",
    "모자": "hat, cap, beanie",
    "액세서리": "scarf, belt, glasses, watch, necklace",
}
# 마스크에서 빼야 할 것들 (SAM 3 로 따로 뽑아 뺀다)
NEGATIVE_MAP = {
    "person": "person, human body, arm, leg, face",
    "hand": "hand, fingers",
    "hair": "hair",
    "hanger": "clothes hanger, coat hook",
}
PAIRED = {"신발"}  # 좌우 한 켤레를 한 아이템으로 묶는 분류


@dataclass
class Candidate:
    category: str
    mask: np.ndarray               # bool (H, W)
    score: float = 0.0
    source: str = ""
    parts: int = 1                 # 합쳐진 조각 수 (신발 한 켤레면 2)


@dataclass
class SegmentResult:
    candidates: list[Candidate] = field(default_factory=list)
    backend: str = ""
    model: str | None = None
    seconds: float = 0.0
    dropped_small: int = 0
    merged: int = 0


# --------------------------------------------------------------------------
# 공통 후처리
# --------------------------------------------------------------------------
def iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def containment(a: np.ndarray, b: np.ndarray) -> float:
    """a 가 b 안에 얼마나 들어가 있나 (작은 쪽 기준)."""
    sa = a.sum()
    return float(np.logical_and(a, b).sum() / sa) if sa else 0.0


def postprocess(cands: list[Candidate], scene: SceneInfo | None = None, *,
                min_area_ratio: float = 0.004, merge_iou: float = 0.55,
                merge_containment: float = 0.80, max_items: int = 12) -> tuple[list[Candidate], dict]:
    """작은 조각 버리기 -> 겹침 합치기 -> 켤레 묶기 -> case1 이면 하나로."""
    stats = {"dropped_small": 0, "merged": 0}
    if not cands:
        return [], stats
    total = cands[0].mask.size

    kept = []
    for c in cands:
        if c.mask.sum() / total < min_area_ratio:
            stats["dropped_small"] += 1
        else:  # 아래에서 마스크를 합치므로 원본을 건드리지 않게 복사해 둔다
            kept.append(Candidate(c.category, c.mask.copy(), c.score, c.source, c.parts))
    kept.sort(key=lambda c: -c.mask.sum())

    # 겹치는 후보 합치기
    merged: list[Candidate] = []
    for c in kept:
        for m in merged:
            if iou(c.mask, m.mask) >= merge_iou or containment(c.mask, m.mask) >= merge_containment:
                m.mask = np.logical_or(m.mask, c.mask)
                m.score = max(m.score, c.score)
                m.parts += c.parts
                if m.category == UNKNOWN and c.category != UNKNOWN:
                    m.category = c.category
                stats["merged"] += 1
                break
        else:
            merged.append(c)

    # 켤레(신발)는 같은 분류끼리 하나로
    grouped: list[Candidate] = []
    for c in merged:
        if c.category in PAIRED:
            same = next((g for g in grouped if g.category == c.category), None)
            if same is not None:
                same.mask = np.logical_or(same.mask, c.mask)
                same.parts += c.parts
                stats["merged"] += 1
                continue
        grouped.append(c)

    # case1(상품 사진)은 한 벌을 여러 조각으로 쪼개면 안 된다.
    # 같은 분류(또는 분류를 모르는 조각)끼리만 합치고, 분류가 다른 아이템은 그대로 둔다.
    # (상·하의를 함께 찍은 상품 사진도 있으므로 무조건 1개로 줄이지는 않는다)
    if scene is not None and scene.case == "case1" and len(grouped) > 1:

        def absorb(target: Candidate, c: Candidate) -> None:
            target.mask = np.logical_or(target.mask, c.mask)
            target.parts += c.parts
            target.score = max(target.score, c.score)
            stats["merged"] += 1

        # 순서에 상관없이: 아는 분류끼리 먼저 묶고, 모르는 조각은 가장 큰 아이템에 붙인다.
        # 분류를 하나도 모르면(예: SAM 3 없이 돌릴 때) 통째로 한 아이템으로 본다.
        collapsed: list[Candidate] = []
        for c in grouped:
            if c.category == UNKNOWN:
                continue
            target = next((g for g in collapsed if g.category == c.category), None)
            if target is None:
                collapsed.append(c)
            else:
                absorb(target, c)
        for c in grouped:
            if c.category != UNKNOWN:
                continue
            if collapsed:
                absorb(max(collapsed, key=lambda g: g.mask.sum()), c)
            else:
                collapsed.append(c)
        grouped = collapsed

    grouped.sort(key=lambda c: -c.mask.sum())
    return grouped[:max_items], stats


# --------------------------------------------------------------------------
# backend: SAM 3
# --------------------------------------------------------------------------
class Sam3Segmenter:
    """SAM 3 텍스트 프롬프트 분할.

    주의: `facebook/sam3` 는 HuggingFace 에서 승인이 필요한(gated) 모델이다.
    HF 계정으로 약관에 동의하고 `HF_TOKEN` 을 넣어야 내려받을 수 있다.
    """

    name = "sam3"

    def __init__(self, model_id: str = "facebook/sam3", threshold: float = 0.4,
                 categories: list[str] | None = None, subtract_negatives: bool = True):
        import torch
        from transformers import AutoModel, AutoProcessor

        from pipeline.garment.gpu import guard_cudnn

        guard_cudnn()  # GB10 에서는 cuDNN 합성곱이 틀린 값을 낸다
        self.torch = torch
        self.model_id = model_id
        self.threshold = threshold
        self.categories = categories or CATEGORIES
        self.subtract_negatives = subtract_negatives
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval().to(self.device)

    def _masks_for(self, img: Image.Image, phrase: str) -> list[tuple[np.ndarray, float]]:
        inputs = self.processor(images=img, text=phrase, return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            out = self.model(**inputs)
        post = getattr(self.processor, "post_process_instance_segmentation", None)
        if post is None:  # 버전이 다르면 이름이 다를 수 있다
            post = self.processor.post_process_grounded_object_detection
        res = post(out, threshold=self.threshold, target_sizes=[(img.height, img.width)])[0]
        masks, scores = res.get("masks"), res.get("scores")
        if masks is None:
            return []
        return [(np.asarray(m.cpu()).astype(bool), float(s))
                for m, s in zip(masks, scores if scores is not None else [1.0] * len(masks))]

    def segment(self, img: Image.Image, scene: SceneInfo | None = None) -> SegmentResult:
        start = time.perf_counter()
        # 힌트가 있으면 그 분류만 물어 본다(빠르다). 없으면 전부 물어 본다.
        wanted = [i["category"] for i in (scene.expected_items if scene else []) if i.get("category") in PROMPT_MAP]
        cats = list(dict.fromkeys(wanted)) or self.categories

        cands: list[Candidate] = []
        for cat in cats:
            for mask, score in self._masks_for(img, PROMPT_MAP[cat]):
                cands.append(Candidate(cat, mask, score, self.name))

        if self.subtract_negatives and cands:
            negative = np.zeros_like(cands[0].mask)
            for phrase in NEGATIVE_MAP.values():
                for mask, _ in self._masks_for(img, phrase):
                    negative |= mask
            for c in cands:
                c.mask = np.logical_and(c.mask, ~negative)

        return SegmentResult(cands, self.name, self.model_id, round(time.perf_counter() - start, 3))


# --------------------------------------------------------------------------
# backend: heuristic (SAM 3 없이 배선 확인용)
# --------------------------------------------------------------------------
def label_components(mask: np.ndarray) -> np.ndarray:
    """연결 요소 번호 매기기. scipy 가 있으면 쓰고 없으면 직접 센다."""
    try:
        from scipy import ndimage

        return ndimage.label(mask)[0]
    except ImportError:
        pass
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    h, w = mask.shape
    for sy in range(h):
        for sx in range(w):
            if not mask[sy, sx] or labels[sy, sx]:
                continue
            current += 1
            stack = [(sy, sx)]
            labels[sy, sx] = current
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                        labels[ny, nx] = current
                        stack.append((ny, nx))
    return labels


def fill_holes(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage

        return ndimage.binary_fill_holes(mask)
    except ImportError:
        return mask


class HeuristicSegmenter:
    """가장자리 색을 배경으로 보고 아이템 덩어리를 찾는다.

    SAM 3 를 받기 전까지 파이프라인 전체를 돌려 보기 위한 임시 구현이다.
    품질 기준선이 아니다 (분류를 못 하므로 category 는 '미상').
    """

    name = "heuristic"

    def __init__(self, work_side: int = 512, threshold: float = 34.0, max_items: int = 12):
        self.work_side, self.threshold, self.max_items = work_side, threshold, max_items

    def segment(self, img: Image.Image, scene: SceneInfo | None = None) -> SegmentResult:
        start = time.perf_counter()
        w, h = img.size
        scale = self.work_side / max(w, h)
        small = img.convert("RGB").resize((max(1, round(w * scale)), max(1, round(h * scale))))
        a = np.asarray(small, dtype=np.float32)

        border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
        bg = np.median(border, axis=0)
        fg = np.linalg.norm(a - bg, axis=-1) > self.threshold
        fg = fill_holes(fg)

        labels = label_components(fg)
        cands: list[Candidate] = []
        for i in range(1, int(labels.max()) + 1):
            piece = labels == i
            full = np.asarray(Image.fromarray(piece.astype(np.uint8) * 255)
                              .resize((w, h), Image.NEAREST)) > 127
            cands.append(Candidate(UNKNOWN, full, float(piece.mean()), self.name))
        cands.sort(key=lambda c: -c.mask.sum())
        cands = cands[: self.max_items]

        # 장면 힌트에 분류가 있으면 큰 덩어리부터 순서대로 붙여 준다
        hints = [i["category"] for i in (scene.expected_items if scene else [])]
        for c, cat in zip(cands, hints):
            c.category = cat
        return SegmentResult(cands, self.name, None, round(time.perf_counter() - start, 3))


# --------------------------------------------------------------------------
# backend: fake (단위 테스트용)
# --------------------------------------------------------------------------
class FakeSegmenter:
    name = "fake"

    def __init__(self, boxes: list[tuple[str, tuple[int, int, int, int]]]):
        self.boxes = boxes

    def segment(self, img: Image.Image, scene: SceneInfo | None = None) -> SegmentResult:
        w, h = img.size
        cands = []
        for cat, (x0, y0, x1, y1) in self.boxes:
            m = np.zeros((h, w), dtype=bool)
            m[y0:y1, x0:x1] = True
            cands.append(Candidate(cat, m, 0.9, self.name))
        return SegmentResult(cands, self.name, "fake", 0.0)


def build_segmenter(backend: str, **kw):
    if backend == "sam3":
        return Sam3Segmenter(kw.get("model_id", "facebook/sam3"))
    if backend == "fake":
        return FakeSegmenter(kw["boxes"])
    return HeuristicSegmenter()

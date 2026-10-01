"""파이프라인 단계들이 주고받는 자료형.

모든 단계는 `PhotoResult` 하나를 받아 자기 자리를 채우고 돌려준다.
JSON으로 저장·복원할 수 있어야 단계별 재실행 없이 다음 단계를 시험할 수 있다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

# 옷장에서 쓰는 아이템 분류. SAM 3 텍스트 프롬프트와 VLM 출력이 이 값으로 맞춰진다.
CATEGORIES = [
    "상의", "하의", "아우터", "원피스", "신발", "가방", "모자", "액세서리",
]
# 분류를 모를 때(예: SAM 3 없이 돌릴 때) 쓰는 값. VLM 출력 enum 에는 넣지 않는다.
UNKNOWN = "미상"

# 옷이 아니어서 마스크에서 빼야 하는 것들
NEGATIVE_PROMPTS = ["사람", "얼굴", "손", "머리카락", "옷걸이", "가구", "배경"]

CASES = ["case1", "case2", "case3"]
CASE_DESC = {
    "case1": "단색 배경 상품 사진",
    "case2": "사람이 입고 찍은 사진",
    "case3": "바닥·침대에 펼쳐 놓은 사진",
}


def _rebuild(cls, data: dict):
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class SceneInfo:
    """1단계 결과. 다음 단계가 참고만 하는 힌트다(없어도 돌아가야 한다)."""

    case: str = "case3"
    confidence: float = 0.0
    expected_items: list[dict] = field(default_factory=list)  # [{"category": ..., "note": ...}]
    backend: str = "none"
    model: str | None = None
    seconds: float = 0.0
    raw: str | None = None  # VLM 원문 (디버깅용)

    @property
    def expected_count(self) -> int:
        return len(self.expected_items)

    @classmethod
    def from_dict(cls, d: dict) -> "SceneInfo":
        return _rebuild(cls, d)


@dataclass
class Item:
    """아이템 하나. 마스크는 파일로 따로 저장하고 여기에는 경로만 둔다."""

    id: str
    category: str
    bbox: list[int]                      # 원본 안 위치 [x0, y0, x1, y1]
    score: float = 0.0                   # 분할 신뢰도
    area_ratio: float = 0.0              # 원본 면적 대비 비율
    occlusion: float = 0.0               # 가려진 비율 추정 (0~1)
    mask_path: str | None = None         # 2단계 거친 마스크
    refined_mask_path: str | None = None # 3단계 정밀 마스크(알파)
    cutout_path: str | None = None       # 4단계 투명 배경 PNG
    meta_path: str | None = None
    ai_restored: bool = False            # 5단계에서 복원했으면 True
    views: dict[str, str] = field(default_factory=dict)   # {"front": 경로, "back": ...}
    view_sources: dict[str, str] = field(default_factory=dict)  # {"back": "ai" | "photo"}
    mesh_path: str | None = None
    stage_models: dict[str, str] = field(default_factory=dict)  # 단계별 쓴 모델

    @classmethod
    def from_dict(cls, d: dict) -> "Item":
        return _rebuild(cls, d)


@dataclass
class PhotoResult:
    """사진 한 장에 대한 전체 결과."""

    source: str                       # 원본 사진 경로
    width: int = 0
    height: int = 0
    scene: SceneInfo = field(default_factory=SceneInfo)
    items: list[Item] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)   # 단계별 초
    gpu_peak_gb: dict[str, float | None] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PhotoResult":
        r = cls(source=d["source"], width=d.get("width", 0), height=d.get("height", 0))
        r.scene = SceneInfo.from_dict(d.get("scene") or {})
        r.items = [Item.from_dict(i) for i in d.get("items", [])]
        r.timings = d.get("timings", {})
        r.gpu_peak_gb = d.get("gpu_peak_gb", {})
        r.errors = d.get("errors", {})
        return r

"""파이프라인 설정. CLI 인자와 테스트가 같은 자료형을 쓴다."""

from __future__ import annotations

from dataclasses import dataclass, field

STAGES = ["scene", "segment", "refine", "export", "complete", "views", "mesh"]
STAGE1_4 = ["scene", "segment", "refine", "export"]


@dataclass
class PipelineConfig:
    # 단계별 backend 이름
    scene_backend: str = "heuristic"       # vlm | heuristic | none
    segment_backend: str = "heuristic"     # sam3 | heuristic | fake
    refine_backend: str = "border"         # birefnet | ben2 | border | fake
    complete_backend: str = "none"
    views_backend: str = "none"
    mesh_backend: str = "none"

    # 모델 지정
    refine_variant: str = "general"        # birefnet: general | hr | matting | lite
    sam3_model_id: str = "facebook/sam3"
    vlm_base_url: str = "http://localhost:8000/v1"
    vlm_model: str = "Qwen/Qwen3.6-35B-A3B"
    vlm_api_key: str = "EMPTY"

    # 후처리
    min_area_ratio: float = 0.004
    merge_iou: float = 0.55
    refine_margin: float = 0.10
    guard_px: int = 6
    max_items: int = 12

    reuse: bool = True                     # 이미 끝난 단계는 다시 돌리지 않는다
    fake_boxes: list = field(default_factory=list)   # segment_backend=fake 일 때

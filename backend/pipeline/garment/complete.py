"""5단계. 가려진 부분 복원(complete) — 인터페이스와 테스트용 가짜 구현.

case2 처럼 몸·팔·머리카락에 가려졌거나 구겨진 아이템을 "정면 상품 사진"처럼 되돌린다
(virtual try-off). 실제 backend 는 Qwen-Image-Edit(Apache-2.0)를 쓴다.

복원한 이미지에는 반드시 `ai_restored=true` 를 남기고, 원본에 없던 로고·무늬를
지어내지 않았는지 검사한다(색 분포·임베딩 유사도).
"""

from __future__ import annotations

from PIL import Image


class QwenImageEditCompleter:
    name = "qwen-image-edit"

    def __init__(self, model_id: str = "Qwen/Qwen-Image-Edit-2511"):
        raise RuntimeError(
            "Qwen-Image-Edit 가중치가 아직 없습니다. models/registry.yaml 의 status 를 확인하세요. "
            "(6·7단계와 함께 2차로 붙입니다)"
        )


class PassthroughCompleter:
    """복원하지 않고 그대로 둔다. 1~4단계만 볼 때의 기본값."""

    name = "none"
    model_id = None

    def complete(self, cutout: Image.Image, occlusion: float) -> tuple[Image.Image, bool]:
        return cutout, False


class FakeCompleter:
    """테스트용. 가려짐이 있으면 '복원했다'고만 표시한다."""

    name = "fake"
    model_id = "fake"

    def complete(self, cutout: Image.Image, occlusion: float) -> tuple[Image.Image, bool]:
        return cutout, occlusion > 0.0


def build_completer(backend: str, **kw):
    if backend == "qwen-image-edit":
        return QwenImageEditCompleter(kw.get("model_id", "Qwen/Qwen-Image-Edit-2511"))
    if backend == "fake":
        return FakeCompleter()
    return PassthroughCompleter()

"""6단계. 다각도 이미지(views) — 인터페이스와 테스트용 가짜 구현.

정면 이미지로 뒤·왼쪽·오른쪽을 만든다. 사용자가 실제 뒷면 사진을 주면 그것을 우선 쓰고,
생성한 면은 `view_sources[면] = "ai"` 로 표시한다.
"""

from __future__ import annotations

from PIL import Image

VIEWS = ["back", "left", "right"]


class QwenImageEditViews:
    name = "qwen-image-edit"

    def __init__(self, model_id: str = "Qwen/Qwen-Image-Edit-2511"):
        raise RuntimeError("Qwen-Image-Edit 가중치가 아직 없습니다. (2차로 붙입니다)")


class NoViews:
    name = "none"
    model_id = None

    def generate(self, front: Image.Image, view: str) -> Image.Image | None:
        return None


class FakeViews:
    """테스트용. 좌우 반전으로 '다른 면'을 흉내 낸다."""

    name = "fake"
    model_id = "fake"

    def generate(self, front: Image.Image, view: str) -> Image.Image:
        return front.transpose(Image.FLIP_LEFT_RIGHT) if view != "back" else front.copy()


def build_views(backend: str, **kw):
    if backend == "qwen-image-edit":
        return QwenImageEditViews(kw.get("model_id", "Qwen/Qwen-Image-Edit-2511"))
    if backend == "fake":
        return FakeViews()
    return NoViews()

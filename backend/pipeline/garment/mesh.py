"""7단계. 3D(mesh) — 인터페이스와 테스트용 가짜 구현.

앞·뒤·옆 이미지로 텍스처 있는 GLB 를 만들고 한 바퀴 도는 미리보기를 렌더링한다.
후보는 TRELLIS(MIT) 와 SAM 3D Objects(승인 필요). Hunyuan3D 계열은 라이선스상 쓰지 않는다.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image


class TrellisMesh:
    name = "trellis"

    def __init__(self, model_id: str = "microsoft/TRELLIS-image-large"):
        raise RuntimeError(
            "TRELLIS 가중치가 없고 ARM64 호환성도 확인 전입니다. (2차로 붙입니다)"
        )


class NoMesh:
    name = "none"
    model_id = None

    def build(self, views: dict[str, Image.Image], out_path: Path) -> Path | None:
        return None


class FakeMesh:
    """테스트용. 실제 3D 대신 자리표시 GLB 를 쓴다."""

    name = "fake"
    model_id = "fake"

    def build(self, views: dict[str, Image.Image], out_path: Path) -> Path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"glTF\x02\x00\x00\x00")  # 자리표시 헤더
        return out_path


def build_mesh(backend: str, **kw):
    if backend == "trellis":
        return TrellisMesh(kw.get("model_id", "microsoft/TRELLIS-image-large"))
    if backend == "fake":
        return FakeMesh()
    return NoMesh()

"""Phase 0 테스트가 함께 쓰는 도구: 이미지 목록, 시간·메모리 측정, 리포트 작성."""

from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic"}


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise SystemExit(f"이미지 폴더가 없습니다: {folder}")
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"이미지가 없습니다: {folder}")
    return images


def load_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    # 폰 사진의 회전 정보(EXIF)를 반영한다
    try:
        from PIL import ImageOps

        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    return img.convert("RGB")


def environment() -> dict:
    env = {"machine": platform.machine(), "python": platform.python_version()}
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        env["torch"] = None
    return env


class GpuPeak:
    """torch가 GPU에 잡은 최대 메모리(GB). GPU가 없으면 None."""

    def __init__(self):
        try:
            import torch

            self.torch = torch if torch.cuda.is_available() else None
        except ImportError:
            self.torch = None

    def reset(self):
        if self.torch:
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()

    def sync(self):
        if self.torch:
            self.torch.cuda.synchronize()

    def peak_gb(self) -> float | None:
        if not self.torch:
            return None
        return round(self.torch.cuda.max_memory_allocated() / 1024**3, 2)


@dataclass
class Timings:
    """이미지별 처리 시간. 앞쪽 warmup 개수는 통계에서 뺀다(첫 실행은 느리다)."""

    warmup: int = 2
    seconds: list[float] = field(default_factory=list)

    def add(self, s: float):
        self.seconds.append(s)

    def summary(self) -> dict:
        measured = self.seconds[self.warmup :] or self.seconds
        if not measured:
            return {}
        ordered = sorted(measured)
        p95 = ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]
        return {
            "count": len(measured),
            "p50_s": round(statistics.median(measured), 3),
            "p95_s": round(p95, 3),
            "mean_s": round(statistics.fmean(measured), 3),
        }


class Stopwatch:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


def checkerboard(size: tuple[int, int], cell: int = 16) -> Image.Image:
    """투명 영역을 보여주기 위한 회색 체크 배경."""
    w, h = size
    board = Image.new("RGB", size, (235, 235, 235))
    dark = Image.new("RGB", (cell, cell), (200, 200, 200))
    for y in range(0, h, cell):
        for x in range((y // cell) % 2 * cell, w, cell * 2):
            board.paste(dark, (x, y))
    return board


def contact_sheet(pairs: list[tuple[str, Image.Image, Image.Image]], out: Path, thumb: int = 320):
    """(이름, 원본, 결과) 목록을 한 장에 모아 눈으로 빠르게 검수하게 한다."""
    if not pairs:
        return
    cols, pad, label_h = 2, 8, 22
    rows = len(pairs)
    sheet = Image.new("RGB", (cols * (thumb + pad) + pad, rows * (thumb + label_h + pad) + pad), "white")
    from PIL import ImageDraw

    draw = ImageDraw.Draw(sheet)
    for r, (name, left, right) in enumerate(pairs):
        y = pad + r * (thumb + label_h + pad)
        draw.text((pad, y), name, fill="black")
        for c, img in enumerate((left, right)):
            tile = img.copy()
            tile.thumbnail((thumb, thumb))
            if tile.mode == "RGBA":
                bg = checkerboard(tile.size)
                bg.paste(tile, mask=tile.split()[-1])
                tile = bg
            sheet.paste(tile, (pad + c * (thumb + pad), y + label_h))
    sheet.save(out, quality=90)


def write_report(out_dir: Path, title: str, data: dict, markdown_body: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(f"# {title}\n\n{markdown_body}\n", encoding="utf-8")

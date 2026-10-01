"""단계별 결과 저장소.

사진마다 `<out>/<사진이름>/` 폴더를 만들고 단계가 끝날 때마다 `state.json` 과
단계 산출물(마스크 PNG 등)을 남긴다. 덕분에 `--stages refine,export` 처럼
뒷 단계만 다시 돌려 볼 수 있다.

구조
  <out>/<stem>/
    state.json            지금까지의 PhotoResult
    stages_done.json      끝난 단계 이름과 시각
    masks/                2단계 거친 마스크
    refined/              3단계 정밀 마스크
    items/                4단계 누끼 PNG·메타데이터
    views/, mesh/         6·7단계
    contact_sheet.jpg
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from pipeline.garment.types import PhotoResult

SUBDIRS = ("masks", "refined", "items", "views", "mesh")


class PhotoStore:
    def __init__(self, out_root: Path, source: Path):
        self.source = Path(source)
        self.dir = Path(out_root) / self.source.stem
        self.state_path = self.dir / "state.json"
        self.done_path = self.dir / "stages_done.json"

    def prepare(self) -> "PhotoStore":
        for sub in ("", *SUBDIRS):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        return self

    # --- 상태 ---------------------------------------------------------------
    def load(self) -> PhotoResult | None:
        if not self.state_path.exists():
            return None
        try:
            return PhotoResult.from_dict(json.loads(self.state_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, KeyError):
            return None  # 깨진 캐시는 없는 셈 친다

    def save(self, result: PhotoResult):
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # --- 어디까지 했는지 ------------------------------------------------------
    def done(self) -> dict[str, str]:
        if not self.done_path.exists():
            return {}
        try:
            return json.loads(self.done_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def mark_done(self, stage: str):
        d = self.done()
        d[stage] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.done_path.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")

    def has(self, stage: str) -> bool:
        return stage in self.done()

    # --- 경로 도우미 ----------------------------------------------------------
    def path(self, sub: str, name: str) -> Path:
        p = self.dir / sub / name
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def rel(self, path: Path | str) -> str:
        """state.json 에는 결과 폴더 기준 상대경로를 적는다(폴더째 옮겨도 살아 있게)."""
        try:
            return str(Path(path).relative_to(self.dir))
        except ValueError:
            return str(path)

    def abs(self, rel: str | None) -> Path | None:
        if not rel:
            return None
        p = Path(rel)
        return p if p.is_absolute() else self.dir / p

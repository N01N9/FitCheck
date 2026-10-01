"""4단계. 저장(export).

아이템마다 투명 배경 PNG, 마스크, 메타데이터 JSON 을 남기고
사진마다 결과를 한눈에 보는 contact sheet 를 만든다.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pipeline.garment.store import PhotoStore
from pipeline.garment.types import CASE_DESC, Item, PhotoResult

SCHEMA_VERSION = 1


def cutout(img: Image.Image, alpha: np.ndarray, crop_to_item: bool = True,
           pad_ratio: float = 0.02, min_pad: int = 4) -> Image.Image:
    """알파를 붙인 RGBA. 기본은 아이템 영역만 잘라 낸다(옷장 카드에 바로 쓰려고).

    옷이 그림 테두리에 붙지 않게 둘레에 투명 여백을 조금 둔다.
    (원본 밖으로 나간 부분은 PIL 이 투명으로 채운다)
    """
    rgba = img.convert("RGBA")
    rgba.putalpha(Image.fromarray(alpha))
    if crop_to_item:
        box = rgba.getbbox()  # 알파가 0 이 아닌 영역
        if box:
            x0, y0, x1, y1 = box
            pad = max(min_pad, round(max(x1 - x0, y1 - y0) * pad_ratio))
            rgba = rgba.crop((x0 - pad, y0 - pad, x1 + pad, y1 + pad))
    return rgba


def item_metadata(item: Item, result: PhotoResult, extra: dict | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": item.id,
        "category": item.category,
        "source_image": result.source,
        "source_size": [result.width, result.height],
        "case": result.scene.case,
        "case_desc": CASE_DESC.get(result.scene.case, ""),
        "bbox": item.bbox,
        "score": round(item.score, 4),
        "area_ratio": round(item.area_ratio, 5),
        "occlusion": round(item.occlusion, 4),
        "ai_restored": item.ai_restored,
        "views": item.views,
        "view_sources": item.view_sources,
        "mesh": item.mesh_path,
        "models": item.stage_models,
        **(extra or {}),
    }


def export_items(result: PhotoResult, img: Image.Image, alphas: dict[str, np.ndarray],
                 store: PhotoStore) -> PhotoResult:
    for item in result.items:
        alpha = alphas.get(item.id)
        if alpha is None:
            continue
        png = store.path("items", f"{item.id}.png")
        cutout(img, alpha).save(png)
        mask_png = store.path("items", f"{item.id}_mask.png")
        Image.fromarray(alpha).save(mask_png)
        meta = store.path("items", f"{item.id}.json")
        meta.write_text(json.dumps(item_metadata(item, result), ensure_ascii=False, indent=2),
                        encoding="utf-8")
        item.cutout_path = store.rel(png)
        item.refined_mask_path = store.rel(mask_png)
        item.meta_path = store.rel(meta)
    return result


def checkerboard(size: tuple[int, int], cell: int = 16) -> Image.Image:
    w, h = size
    board = Image.new("RGB", size, (235, 235, 235))
    dark = Image.new("RGB", (cell, cell), (205, 205, 205))
    for y in range(0, h, cell):
        for x in range((y // cell) % 2 * cell, w, cell * 2):
            board.paste(dark, (x, y))
    return board


def on_checker(img: Image.Image, thumb: int) -> Image.Image:
    tile = img.copy()
    tile.thumbnail((thumb, thumb))
    if tile.mode == "RGBA":
        bg = checkerboard(tile.size)
        bg.paste(tile, mask=tile.split()[-1])
        return bg
    return tile.convert("RGB")


def contact_sheet(result: PhotoResult, img: Image.Image, store: PhotoStore, thumb: int = 300) -> Path:
    """맨 왼쪽에 원본, 오른쪽에 아이템별 누끼를 늘어놓는다."""
    cols = 1 + max(1, len(result.items))
    label_h = 34
    sheet = Image.new("RGB", (cols * (thumb + 10) + 10, thumb + label_h + 20), "white")
    draw = ImageDraw.Draw(sheet)
    scene = result.scene
    draw.text((10, 6), f"{Path(result.source).name}  |  {scene.case} ({scene.backend}, conf {scene.confidence})"
                       f"  |  아이템 {len(result.items)}개", fill="black")

    sheet.paste(on_checker(img, thumb), (10, label_h))
    for i, item in enumerate(result.items):
        x = 10 + (i + 1) * (thumb + 10)
        path = store.abs(item.cutout_path)
        if path and Path(path).exists():
            sheet.paste(on_checker(Image.open(path), thumb), (x, label_h))
        draw.text((x, label_h - 16), f"{item.id} {item.category} "
                                     f"가림{item.occlusion:.0%}", fill="black")
    out = store.dir / "contact_sheet.jpg"
    sheet.save(out, quality=88)
    return out

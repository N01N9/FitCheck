"""옷 파이프라인 실행기(CLI).

  python -m pipeline.garment.run --image 사진경로_또는_폴더 --out 결과폴더 \
      [--stages segment,refine,export] [--back 뒷면사진]

단계는 scene -> segment -> refine -> export -> complete -> views -> mesh 순이다.
각 단계가 끝나면 결과 폴더에 저장되므로, 뒤 단계만 다시 돌려 볼 수 있다.
예: 먼저 전부 돌린 뒤 `--stages refine,export --refine-backend ben2` 로 경계 모델만 바꿔 비교한다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from phase0.common import GpuPeak, list_images, load_rgb
from pipeline.garment import complete as complete_mod
from pipeline.garment import export as export_mod
from pipeline.garment import mesh as mesh_mod
from pipeline.garment import refine as refine_mod
from pipeline.garment import scene as scene_mod
from pipeline.garment import segment as segment_mod
from pipeline.garment import views as views_mod
from pipeline.garment.config import STAGE1_4, STAGES, PipelineConfig
from pipeline.garment.store import PhotoStore
from pipeline.garment.types import Item, PhotoResult


# --------------------------------------------------------------------------
# 단계 실행
# --------------------------------------------------------------------------
def run_scene(img: Image.Image, result: PhotoResult, cfg: PipelineConfig, store: PhotoStore):
    backend = scene_mod.build_scene(
        cfg.scene_backend, base_url=cfg.vlm_base_url, model=cfg.vlm_model, api_key=cfg.vlm_api_key)
    result.scene = backend.analyze(img)
    return result


def run_segment(img: Image.Image, result: PhotoResult, cfg: PipelineConfig, store: PhotoStore):
    backend = segment_mod.build_segmenter(
        cfg.segment_backend, model_id=cfg.sam3_model_id, boxes=cfg.fake_boxes)
    seg = backend.segment(img, result.scene)
    cands, stats = segment_mod.postprocess(
        seg.candidates, result.scene, min_area_ratio=cfg.min_area_ratio,
        merge_iou=cfg.merge_iou, max_items=cfg.max_items)

    total = img.size[0] * img.size[1]
    items: list[Item] = []
    for i, c in enumerate(cands):
        item_id = f"item_{i:02d}"
        path = store.path("masks", f"{item_id}.png")
        Image.fromarray((c.mask * 255).astype(np.uint8)).save(path)
        x0, y0, x1, y1 = refine_mod.bbox_of(c.mask)
        items.append(Item(id=item_id, category=c.category, bbox=[x0, y0, x1, y1],
                          score=round(c.score, 4), area_ratio=round(float(c.mask.sum()) / total, 5),
                          mask_path=store.rel(path),
                          stage_models={"segment": f"{seg.backend}:{seg.model or '-'}"}))
    result.items = items
    result.errors.pop("segment", None)
    if not items:
        result.errors["segment"] = "아이템을 하나도 찾지 못했습니다"
    return result


def _load_mask(store: PhotoStore, rel: str | None, size: tuple[int, int]) -> np.ndarray | None:
    path = store.abs(rel)
    if not path or not Path(path).exists():
        return None
    return np.asarray(Image.open(path).convert("L")) > 127


def run_refine(img: Image.Image, result: PhotoResult, cfg: PipelineConfig, store: PhotoStore):
    remover = refine_mod.build_remover(
        cfg.refine_backend, variant=cfg.refine_variant)
    refiner = refine_mod.Refiner(remover, margin=cfg.refine_margin, guard_px=cfg.guard_px)
    for item in result.items:
        mask = _load_mask(store, item.mask_path, img.size)
        if mask is None:
            result.errors[f"refine:{item.id}"] = "2단계 마스크가 없습니다"
            continue
        alpha, box, _ = refiner.refine(img, mask)
        path = store.path("refined", f"{item.id}.png")
        Image.fromarray(alpha).save(path)
        item.refined_mask_path = store.rel(path)
        item.bbox = list(box)
        item.occlusion = round(refine_mod.occlusion_ratio(alpha), 4)
        item.stage_models["refine"] = f"{refiner.name}:{refiner.model_id or '-'}"
    return result


def run_export(img: Image.Image, result: PhotoResult, cfg: PipelineConfig, store: PhotoStore):
    alphas: dict[str, np.ndarray] = {}
    for item in result.items:
        path = store.abs(item.refined_mask_path) or store.abs(item.mask_path)
        if path and Path(path).exists():
            alphas[item.id] = np.asarray(Image.open(path).convert("L"))
    export_mod.export_items(result, img, alphas, store)
    export_mod.contact_sheet(result, img, store)
    return result


def run_complete(img: Image.Image, result: PhotoResult, cfg: PipelineConfig, store: PhotoStore):
    completer = complete_mod.build_completer(cfg.complete_backend)
    for item in result.items:
        path = store.abs(item.cutout_path)
        if not path or not Path(path).exists():
            continue
        out, restored = completer.complete(Image.open(path), item.occlusion)
        if restored:
            dest = store.path("items", f"{item.id}_restored.png")
            out.save(dest)
            item.views["front"] = store.rel(dest)
            item.view_sources["front"] = "ai"
            item.ai_restored = True
            item.stage_models["complete"] = f"{completer.name}:{completer.model_id or '-'}"
        else:
            item.views.setdefault("front", item.cutout_path)
            item.view_sources.setdefault("front", "photo")
    return result


def run_views(img: Image.Image, result: PhotoResult, cfg: PipelineConfig, store: PhotoStore,
              back_photo: Path | None = None):
    gen = views_mod.build_views(cfg.views_backend)
    for item in result.items:
        front_rel = item.views.get("front") or item.cutout_path
        front_path = store.abs(front_rel)
        if not front_path or not Path(front_path).exists():
            continue
        front = Image.open(front_path)
        for view in views_mod.VIEWS:
            if view == "back" and back_photo and Path(back_photo).exists():
                dest = store.path("views", f"{item.id}_back.png")
                Image.open(back_photo).convert("RGBA").save(dest)
                item.views["back"] = store.rel(dest)
                item.view_sources["back"] = "photo"   # 실제 사진이 우선이다
                continue
            made = gen.generate(front, view)
            if made is None:
                continue
            dest = store.path("views", f"{item.id}_{view}.png")
            made.save(dest)
            item.views[view] = store.rel(dest)
            item.view_sources[view] = "ai"            # 생성한 면은 "AI 추정"으로 표시
        if gen.name != "none":
            item.stage_models["views"] = f"{gen.name}:{gen.model_id or '-'}"
    return result


def run_mesh(img: Image.Image, result: PhotoResult, cfg: PipelineConfig, store: PhotoStore):
    builder = mesh_mod.build_mesh(cfg.mesh_backend)
    for item in result.items:
        views = {}
        for name, rel in item.views.items():
            path = store.abs(rel)
            if path and Path(path).exists():
                views[name] = Image.open(path)
        if not views:
            continue
        out = builder.build(views, store.path("mesh", f"{item.id}.glb"))
        if out is not None:
            item.mesh_path = store.rel(out)
            item.stage_models["mesh"] = f"{builder.name}:{builder.model_id or '-'}"
    return result


RUNNERS = {
    "scene": run_scene, "segment": run_segment, "refine": run_refine, "export": run_export,
    "complete": run_complete, "views": run_views, "mesh": run_mesh,
}


# --------------------------------------------------------------------------
# 사진 한 장
# --------------------------------------------------------------------------
def process_photo(path: Path, out_root: Path, cfg: PipelineConfig,
                  stages: list[str] | None = None, back_photo: Path | None = None) -> PhotoResult:
    stages = stages or STAGE1_4
    store = PhotoStore(out_root, path).prepare()
    img = load_rgb(path)

    result = store.load() or PhotoResult(source=str(path))
    result.source = str(path)
    result.width, result.height = img.size

    gpu = GpuPeak()
    for stage in STAGES:
        if stage not in stages:
            continue
        if cfg.reuse and store.has(stage):
            continue
        gpu.reset()
        start = time.perf_counter()
        try:
            if stage == "views":
                result = RUNNERS[stage](img, result, cfg, store, back_photo)
            else:
                result = RUNNERS[stage](img, result, cfg, store)
        except Exception as exc:  # 한 단계가 실패해도 지금까지 결과는 남긴다
            result.errors[stage] = f"{type(exc).__name__}: {exc}"
            store.save(result)
            raise
        gpu.sync()
        result.timings[stage] = round(time.perf_counter() - start, 3)
        result.gpu_peak_gb[stage] = gpu.peak_gb()
        store.save(result)
        store.mark_done(stage)
    store.save(result)
    return result


def process(images: list[Path], out_root: Path, cfg: PipelineConfig,
            stages: list[str] | None = None, back_photo: Path | None = None) -> list[PhotoResult]:
    results = []
    for path in images:
        try:
            results.append(process_photo(path, out_root, cfg, stages, back_photo))
        except Exception as exc:
            print(f"  [실패] {path.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    summary = {
        "images": len(images),
        "ok": len(results),
        "items_total": sum(len(r.items) for r in results),
        "stage_seconds_mean": _mean_timings(results),
        "per_image": [{"file": Path(r.source).name, "case": r.scene.case,
                       "items": len(r.items), "timings": r.timings} for r in results],
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    return results


def _mean_timings(results: list[PhotoResult]) -> dict[str, float]:
    out: dict[str, float] = {}
    for stage in STAGES:
        vals = [r.timings[stage] for r in results if stage in r.timings]
        if vals:
            out[stage] = round(sum(vals) / len(vals), 3)
    if out:
        out["stage1_4_total"] = round(sum(v for k, v in out.items() if k in STAGE1_4), 3)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="옷 추출 파이프라인")
    p.add_argument("--image", required=True, help="사진 파일 또는 폴더")
    p.add_argument("--out", required=True, help="결과 폴더")
    p.add_argument("--stages", default=",".join(STAGE1_4),
                   help=f"돌릴 단계 (쉼표로 구분). 가능: {','.join(STAGES)}")
    p.add_argument("--back", help="뒷면 사진 (6단계에서 생성보다 우선한다)")
    p.add_argument("--scene-backend", default="heuristic", choices=["vlm", "heuristic", "none"])
    p.add_argument("--segment-backend", default="heuristic", choices=["sam3", "heuristic", "fake"])
    p.add_argument("--refine-backend", default="border", choices=["birefnet", "ben2", "border", "fake"])
    p.add_argument("--refine-variant", default="general", choices=list(refine_mod.BIREFNET_VARIANTS))
    p.add_argument("--sam3-model-id", default="facebook/sam3")
    p.add_argument("--vlm-base-url", default="http://localhost:8000/v1")
    p.add_argument("--vlm-model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--vlm-api-key", default="EMPTY")
    p.add_argument("--complete-backend", default="none", choices=["qwen-image-edit", "none", "fake"])
    p.add_argument("--views-backend", default="none", choices=["qwen-image-edit", "none", "fake"])
    p.add_argument("--mesh-backend", default="none", choices=["trellis", "none", "fake"])
    p.add_argument("--no-reuse", action="store_true", help="이미 끝난 단계도 다시 돌린다")
    p.add_argument("--limit", type=int, help="폴더일 때 앞에서 N장만")
    return p.parse_args(argv)


def config_from_args(args) -> PipelineConfig:
    return PipelineConfig(
        scene_backend=args.scene_backend, segment_backend=args.segment_backend,
        refine_backend=args.refine_backend, refine_variant=args.refine_variant,
        complete_backend=args.complete_backend, views_backend=args.views_backend,
        mesh_backend=args.mesh_backend, sam3_model_id=args.sam3_model_id,
        vlm_base_url=args.vlm_base_url, vlm_model=args.vlm_model, vlm_api_key=args.vlm_api_key,
        reuse=not args.no_reuse)


def main(argv=None):
    args = parse_args(argv)
    target = Path(args.image)
    images = list_images(target) if target.is_dir() else [target]
    if args.limit:
        images = images[: args.limit]
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise SystemExit(f"모르는 단계: {unknown}. 가능: {STAGES}")

    out_root = Path(args.out)
    results = process(images, out_root, config_from_args(args), stages,
                      Path(args.back) if args.back else None)
    print(f"사진 {len(images)}장 중 {len(results)}장 처리, "
          f"아이템 {sum(len(r.items) for r in results)}개 -> {out_root}")
    print(json.dumps(_mean_timings(results), ensure_ascii=False))
    return results


if __name__ == "__main__":
    main()

"""unpaired.zeroshot 결과를 채점한다. 선택용 검사기와 평가용 지표를 일부러 분리한다.

  검사기(선택용)  gates.check — 형식·색·겉옷 섞임. K장 중 하나를 고르는 데만 쓴다
  평가(보고용)    right_item: DINOv2-large(학습 안 함) 검색 순위 1위 + 겉옷 색이 섞이지 않음.
                 겉옷을 내놓거나 겉옷과 합친 실패를 잡는다 (2026-10-03 눈 확인: 검색 순위만으로는
                 블레이저+셔츠 합본을 맞다고 보는 경우가 있었다)
                 faithful: right_item + 보이는 색 일치 + 대상에 없던 색 없음 + 표시(초록 선) 안 그림

지표(변형 × 가시율 구간). 뒤에 _c 가 붙은 것은 faithful 기준이다
  single  첫 장(k=0)이 맞는 옷인 비율
  oracle  K장 중 하나라도 맞는 비율 (검사기가 완벽할 때의 상한)
  picked  검사기 점수로 고른 한 장이 맞는 비율 (검사기가 색도 보므로 picked_c 는 낙관적이다)
  gate_pass, precision  검사 통과율, 통과한 것 중 맞는 비율

사용 (컨테이너 안)
  python -m unpaired.score --run results/unpaired/zs_pilot/klein4b
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from unpaired import gates
from unpaired.layered import Annotations
from unpaired.masks import bbox

DINO_ID = "facebook/dinov2-large"
DINO_SIZE = 448  # 14 의 배수
COLOR_OK = 12.0   # 대상의 보이는 색 → 결과물 팔레트 거리(ΔE00, kL=2)
EXTRA_OK = 15.0   # 결과물 색 → 대상 색. 크면 대상에 없던 색(대개 겉옷 색)이 많다
OUTER_OK = 0.2    # 결과물 중 겉옷 색에 더 가까운 픽셀 비율
MARKER_OK = 0.01  # 결과물 옷 픽셀 중 표시용 초록에 가까운 비율
# D-HARD: 그래픽·글자·무늬가 있어 "무지로 단정"하거나 지어내기 쉬운 옷(Fashionpedia 무늬 속성)
HARD_PATTERNS = {"letters, numbers", "cartoon", "abstract", "geometric", "floral", "camouflage", "animal",
                 "paisley", "plant", "stripe", "check", "dot"}


def is_hard(case: dict) -> bool:
    patterns = set(case["inner"]["attributes"].get("textile pattern", []))
    return bool(patterns & HARD_PATTERNS)


def marker_fraction(result: np.ndarray, fg: np.ndarray | None = None) -> float:
    """결과물에 표시용 초록(pointer.MARK_RGB)이 그려진 비율. fg 가 없으면 흰 배경이 아닌 픽셀 기준."""
    from core.color import srgb_to_lab
    from unpaired.color import delta_e2000
    from unpaired.pointer import MARK_RGB

    if fg is None:
        fg = (result.astype(int).sum(-1) < 735)
    if fg.sum() == 0:
        return 0.0
    d = delta_e2000(srgb_to_lab(result[fg]), srgb_to_lab(np.array(MARK_RGB, np.uint8)))
    return float((d < 20).mean())


def right_item(r: dict) -> bool:
    s = r["scores"]
    no_outer = s.get("same_color", False) or s.get("outer_frac", 0.0) <= OUTER_OK
    return r["margin"] > 0 and s.get("fg_frac", 0.0) >= 0.02 and no_outer


def faithful(r: dict) -> bool:
    s = r["scores"]
    chroma_ok = not (s.get("target_chroma", 0.0) > 8 and not 0.5 <= s.get("chroma_ratio", 0.0) <= 2.0)
    return (right_item(r) and s.get("palette_dist", 99.0) <= COLOR_OK and s.get("extra_color", 99.0) <= EXTRA_OK
            and chroma_ok and r.get("marker_frac", 0.0) <= MARKER_OK)


def masked_crop(img: np.ndarray, mask: np.ndarray, pad: float = 0.05) -> Image.Image:
    """마스크 밖은 흰색으로 지우고 bbox 로 자른 뒤 흰 정사각형에 앉힌다."""
    x0, y0, x1, y1 = bbox(mask)
    px, py = int((x1 - x0) * pad), int((y1 - y0) * pad)
    h, w = mask.shape
    x0, y0, x1, y1 = max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py)
    out = img.copy()
    out[~mask] = 255
    crop = out[y0:y1, x0:x1]
    side = max(crop.shape[:2])
    canvas = np.full((side, side, 3), 255, np.uint8)
    oy, ox = (side - crop.shape[0]) // 2, (side - crop.shape[1]) // 2
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return Image.fromarray(canvas)


class Dino:
    def __init__(self):
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.model = AutoModel.from_pretrained(DINO_ID, torch_dtype=torch.float32).eval().to("cuda")
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    def __call__(self, images: list[Image.Image]) -> np.ndarray:
        torch = self.torch
        x = np.stack([np.asarray(im.convert("RGB").resize((DINO_SIZE, DINO_SIZE), Image.BICUBIC)) for im in images])
        x = torch.from_numpy(x).to("cuda").permute(0, 3, 1, 2).float() / 255
        with torch.inference_mode():
            h = self.model(pixel_values=(x - self.mean) / self.std).last_hidden_state
        emb = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=1)
        return torch.nn.functional.normalize(emb, dim=1).cpu().numpy()


def summarize(rows: list[dict]) -> dict:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        r["correct"], r["correct_color"] = right_item(r), faithful(r)
        for b in (r["share_bin"], "all") + (("hard",) if r.get("hard") else ()):
            groups[(r["variant"], b, r["file"])].append(r)
    table: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for (variant, b, _), items in groups.items():
        items.sort(key=lambda r: r["k"])
        picked = min(items, key=lambda r: r["gate_score"])
        cell = table[variant][b]
        for key, suffix in (("correct", ""), ("correct_color", "_c")):
            cell["single" + suffix].append(items[0][key])
            cell["oracle" + suffix].append(any(r[key] for r in items))
            cell["picked" + suffix].append(picked[key])
        cell["picked_passed"].append(picked["passed"])
        cell["gate_pass"].extend(r["passed"] for r in items)
        cell["correct_any_k"].extend(r["correct"] for r in items)
        cell["precision"].extend(r["correct"] for r in items if r["passed"])
        cell["picked_palette_dist"].append(picked["scores"].get("palette_dist", np.nan))
        cell["marker_drawn"].extend(r.get("marker_frac", 0.0) > MARKER_OK for r in items)
    out = {}
    for variant, bins in table.items():
        out[variant] = {}
        for b, cell in bins.items():
            out[variant][b] = {
                "n": len(cell["single"]),
                **{k: round(float(np.mean(v)), 3) if v else None for k, v in cell.items()
                   if k != "picked_palette_dist"},
                "picked_palette_dist_median": round(float(np.nanmedian(cell["picked_palette_dist"])), 2),
            }
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="학습 없는 추출 결과 채점")
    p.add_argument("--run", required=True, help="zeroshot 결과 폴더(모델 이름까지)")
    p.add_argument("--fashionpedia", default="data/fashionpedia")
    p.add_argument("--summarize-only", action="store_true", help="저장된 scores.jsonl 로 지표만 다시 계산(CPU)")
    args = p.parse_args(argv)

    run = Path(args.run)
    if args.summarize_only:
        rows = [json.loads(line) for line in (run / "scores.jsonl").read_text().splitlines()]
        for r in rows:
            if "marker_frac" not in r:
                img = np.array(Image.open(run / r["variant"] / f"{Path(r['file']).stem}_k{r['k']}.png").convert("RGB"))
                r["marker_frac"] = round(marker_fraction(img), 4)
        write(run, rows)
        return

    from phase0.exp0_edit_models import disable_broken_cudnn
    from pipeline.garment.refine import BiRefNetRemover

    disable_broken_cudnn()
    cases = json.loads((run / "cases.json").read_text())
    ann = Annotations(Path(args.fashionpedia))
    remover = BiRefNetRemover()
    dino = Dino()
    rows = []
    for case in cases:
        stem = Path(case["file"]).stem
        photo = np.array(Image.open(ann.image_dir / case["file"]).convert("RGB"))
        inner = ann.mask(case["file"], case["inner"]["ann_id"])
        outer = ann.mask(case["file"], case["outer"]["ann_id"])
        others = [a for a in ann.garments(case["file"]) if a["id"] != case["inner"]["ann_id"]]
        crops = [masked_crop(photo, inner)] + [masked_crop(photo, ann.mask(case["file"], a["id"])) for a in others]
        ref = dino(crops)
        for vdir in sorted(d for d in run.iterdir() if d.is_dir()):
            for path in sorted(vdir.glob(f"{stem}_k*.png")):
                result = np.array(Image.open(path).convert("RGB"))
                fg = np.asarray(remover.predict_mask(Image.fromarray(result))) > 127
                g = gates.check(photo, inner, outer, result, fg, category=case["inner"]["category"])
                sims = ref @ dino([masked_crop(result, fg) if fg.sum() > 50 else Image.fromarray(result)])[0]
                margin = float(sims[0] - sims[1:].max()) if len(sims) > 1 else 1.0
                outer_pos = next((i + 1 for i, a in enumerate(others) if a["id"] == case["outer"]["ann_id"]), None)
                rows.append({"file": case["file"], "variant": vdir.name, "k": int(path.stem.rsplit("_k", 1)[1]),
                             "share_bin": case["share_bin"], "hard": is_hard(case), "passed": g.passed,
                             "reasons": g.reasons,
                             "scores": g.scores, "gate_score": gates.score(g),
                             "sim_target": round(float(sims[0]), 4),
                             "sim_outer": round(float(sims[outer_pos]), 4) if outer_pos else None,
                             "margin": round(margin, 4), "marker_frac": round(marker_fraction(result, fg), 4)})
        print(f"scored {case['file']}", flush=True)
    write(run, rows)


def write(run: Path, rows: list[dict]) -> None:
    summary = summarize(rows)
    with (run / "scores.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

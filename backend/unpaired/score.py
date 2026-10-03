"""unpaired.zeroshot 결과를 채점한다. 선택용 검사기와 평가용 지표를 일부러 분리한다.

  검사기(선택용)  gates.check — 형식·색·겉옷 섞임. K장 중 하나를 고르는 데만 쓴다
  평가(보고용)    DINOv2-large(학습 안 함) 검색 순위. 결과물이 사진 속 옷들 중 대상 옷과 가장
                 닮았으면(1위) 맞는 옷으로 본다. 겉옷을 내놓거나 겉옷과 합친 실패를 잡는다

지표(변형 × 가시율 구간). 뒤에 _c 가 붙은 것은 "맞는 옷 + 보이는 색 일치"(COLOR_OK) 기준이다
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
COLOR_OK = 12.0  # 평가용 색 일치: 대상의 보이는 색 → 결과물 팔레트 거리(ΔE00, kL=2)


def color_ok(scores: dict) -> bool:
    if scores.get("palette_dist", 99.0) > COLOR_OK:
        return False
    return not (scores.get("target_chroma", 0.0) > 8 and not 0.5 <= scores.get("chroma_ratio", 0.0) <= 2.0)


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
        for b in (r["share_bin"], "all"):
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
    args = p.parse_args(argv)

    from phase0.exp0_edit_models import disable_broken_cudnn
    from pipeline.garment.refine import BiRefNetRemover

    disable_broken_cudnn()
    run = Path(args.run)
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
                g = gates.check(photo, inner, outer, result, fg)
                sims = ref @ dino([masked_crop(result, fg) if fg.sum() > 50 else Image.fromarray(result)])[0]
                margin = float(sims[0] - sims[1:].max()) if len(sims) > 1 else 1.0
                outer_pos = next((i + 1 for i, a in enumerate(others) if a["id"] == case["outer"]["ann_id"]), None)
                rows.append({"file": case["file"], "variant": vdir.name, "k": int(path.stem.rsplit("_k", 1)[1]),
                             "share_bin": case["share_bin"], "passed": g.passed, "reasons": g.reasons,
                             "scores": g.scores, "gate_score": gates.score(g),
                             "sim_target": round(float(sims[0]), 4),
                             "sim_outer": round(float(sims[outer_pos]), 4) if outer_pos else None,
                             "margin": round(margin, 4)})
                rows[-1]["correct"] = margin > 0 and 0.02 <= g.scores["fg_frac"]
                rows[-1]["correct_color"] = rows[-1]["correct"] and color_ok(g.scores)
        print(f"scored {case['file']}", flush=True)
    with (run / "scores.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    summary = summarize(rows)
    (run / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

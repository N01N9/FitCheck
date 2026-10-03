"""E-실제숨김 평가: 실제 겉옷 조각으로 가린 사진에서 학습한 LoRA 를 잰다. 정답은 원본의 실제 픽셀이다.

  peel     "[PEEL] 재킷" 결과를 원본과 비교한다. 새로 가려졌던 곳(hidden)의 ΔE00 이 핵심이고,
           원래 보이던 곳(visible)·나머지(rest)는 바뀌지 않아야 한다
  extract  가린 사진에서 꺼낸 이너와 안 가린 원본에서 꺼낸 이너가 같은지(DINOv2 코사인, 팔레트 거리).
           원본에서 꺼내기는 쉬운 문제라서, 둘이 같으면 가림에 흔들리지 않는다는 뜻이다

사용 (컨테이너 안)
  python -m unpaired.eval_hidden --lora data/unpaired/runs/lora_v1/step03000 --per-bin 30 \\
      --out results/unpaired/eval_hidden/lora_v1
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from core.color import srgb_to_lab
from unpaired.color import delta_e2000, palette, palette_distance
from unpaired.pairs import PEEL_LONG_SIDE, REF_LONG_SIDE, extract_prompt, fit, peel_prompt, pointer_refs
from unpaired.pointer import outline


def load_case(root: Path, cid: str):
    x = np.array(Image.open(root / f"{cid}_input.jpg").convert("RGB"))
    orig = np.array(Image.open(root / f"{cid}_orig.jpg").convert("RGB"))
    m = np.asarray(Image.open(root / f"{cid}_masks.png").convert("RGB")) > 127
    return x, orig, m[..., 0], m[..., 1], m[..., 2]


def region_de(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    if mask.sum() == 0:
        return float("nan")
    return float(np.median(delta_e2000(srgb_to_lab(a[mask]), srgb_to_lab(b[mask]))))


def resize_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127


def pick(rows: list[dict], per_bin: int) -> list[dict]:
    out = []
    for b in ("lt15", "15_40", "40_70"):
        hits = sorted((r for r in rows if r["share_bin"] == b), key=lambda r: hashlib.sha1(r["id"].encode()).hexdigest())
        out += hits[:per_bin]
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="E-실제숨김 평가 (겉옷 벗기기, 가림에 흔들리지 않는 추출)")
    p.add_argument("--lora", required=True)
    p.add_argument("--cases", default="data/unpaired/eval/e_real_hidden")
    p.add_argument("--per-bin", type=int, default=30)
    p.add_argument("--out", required=True)
    p.add_argument("--base", action="store_true", help="klein-base-4B 본체로 CFG 다단계 추론")
    p.add_argument("--prompt-style", default="struct", help="학습 쌍과 같은 지시문 형식: struct / natural")
    args = p.parse_args(argv)

    from unpaired import editors
    from unpaired.score import Dino

    root, out = Path(args.cases), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = pick([json.loads(line) for line in (root / "cases.jsonl").read_text().splitlines()], args.per_bin)
    model = editors.load("klein", args.lora, base=args.base)
    dino = Dino()
    results = []
    for r in rows:
        x, orig, jacket, top, hidden = load_case(root, r["id"])
        # 겉옷 벗기기: 학습 쌍(pairs.from_layer)과 같은 입력·크기
        ref = fit(outline(x, jacket), REF_LONG_SIDE)
        target = fit(orig, PEEL_LONG_SIDE)
        peeled = np.array(model([ref], peel_prompt(r["donor_category"], args.prompt_style), [0], target.size)[0].convert("RGB"))
        size = target.size
        o = np.asarray(target)
        rec = {"id": r["id"], "share_bin": r["share_bin"], "category": r["category"],
               "peel_hidden_de": region_de(peeled, o, resize_mask(hidden, size)),
               "peel_visible_de": region_de(peeled, o, resize_mask(top & ~jacket, size)),
               "peel_rest_de": region_de(peeled, o, resize_mask(~top & ~jacket, size))}
        Image.fromarray(peeled).save(out / f"{r['id']}_peel.jpg", quality=92)
        # 추출: 가린 사진 vs 원본
        layer = f"inner under {r['donor_category']}"
        occl = model(pointer_refs(x, top & ~jacket), extract_prompt(r["category"], layer, args.prompt_style), [0], (768, 768))[0]
        clean = model(pointer_refs(orig, top), extract_prompt(r["category"], "single", args.prompt_style), [0], (768, 768))[0]
        occl.save(out / f"{r['id']}_extract_occluded.png")
        clean.save(out / f"{r['id']}_extract_clean.png")
        e = dino([occl, clean])
        a, b = np.asarray(occl.convert("RGB")), np.asarray(clean.convert("RGB"))
        fa, fb = a.astype(int).sum(-1) < 735, b.astype(int).sum(-1) < 735
        rec["extract_dino_cos"] = float(e[0] @ e[1])
        if fa.sum() > 50 and fb.sum() > 50:
            rec["extract_palette_dist"] = palette_distance(palette(srgb_to_lab(a[fa])), palette(srgb_to_lab(b[fb])))
        results.append(rec)
        print(json.dumps(rec), flush=True)
    with (out / "results.jsonl").open("w") as fh:
        for rec in results:
            fh.write(json.dumps(rec) + "\n")
    summary = defaultdict(dict)
    for b in ("lt15", "15_40", "40_70", "all"):
        sel = [rec for rec in results if b == "all" or rec["share_bin"] == b]
        for k in ("peel_hidden_de", "peel_visible_de", "peel_rest_de", "extract_dino_cos", "extract_palette_dist"):
            vals = [rec[k] for rec in sel if k in rec and rec[k] == rec[k]]
            summary[b][k] = round(float(np.median(vals)), 3) if vals else None
        summary[b]["n"] = len(sel)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

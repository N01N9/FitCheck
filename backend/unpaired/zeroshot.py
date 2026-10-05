"""학습 없는 기준선: 겹쳐 입은 실제 사진에서 이너만 상품 사진으로 꺼내기.

대상 표시 방식(pointer.VARIANTS)마다 시드 K개로 K장을 만든다. 채점은 unpaired.score 가 따로 한다.
이미 만든 결과는 건너뛰므로 끊겨도 같은 명령으로 이어서 돌릴 수 있다.

학습한 LoRA 는 --lora 로 얹는다. 이때 변형은 "lora" 하나이고, 표시·지시문을 학습 쌍(unpaired.pairs)과
똑같이 만든다(사진 전체 + 외곽선, 대상 크롭 / "[EXTRACT] t-shirt; layer=inner under jacket").

사용 (컨테이너 안)
  python -m unpaired.zeroshot --model klein --per-bin 4 --include-exp0 --k 4 --out results/unpaired/zs_pilot
  python -m unpaired.zeroshot --model klein --lora runs/lora1/step01500 --tag lora1 --per-bin 20 --out results/unpaired/eval
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from unpaired.layered import Annotations, load, sample_by_bin
from unpaired.pointer import VARIANTS, render

PRODUCT = ("Create a store product photo of {what}: the garment alone, laid flat and neatly smoothed, front view, "
           "centered on a plain white background. Keep its exact colors, pattern, print and details. "
           "Show only this one garment: no {outer}, no person, no other clothing.")
WHAT = {
    "text": "the {cat} worn under the {outer}",
    "outline": "the {cat} marked with the green outline (it is worn under the {outer})",
    "fill": "the {cat} highlighted in green (it is worn under the {outer})",
    "grey_outer": "the {cat} marked with the green outline (the {outer} over it is greyed out)",
    "outline_dimcrop": "the {cat} marked with the green outline in image 1 "
                       "(image 2 is a close-up of it; it is worn under the {outer})",
}
MARK_NOTE = " The green marks are only pointers; do not draw them."


def prompt_for(variant: str, row: dict, style: str = "struct") -> str:
    cat, outer = row["inner"]["category"], row["outer"]["category"]
    if variant == "lora":
        from unpaired.pairs import extract_prompt

        return extract_prompt(cat, f"inner under {outer}", style)
    if variant == "lora_natural":  # 증류 모델이 이미 아는 자연어 지시문(학습 없는 outline_dimcrop 과 같음)
        variant = "outline_dimcrop"
    text = PRODUCT.format(what=WHAT[variant].format(cat=cat, outer=outer), outer=outer)
    return text if variant == "text" else text + MARK_NOTE


def select(index: Path, split: str, per_bin: int, include_exp0: bool) -> list[dict]:
    rows = sample_by_bin(load(index, split), per_bin)
    if include_exp0:
        rows = load(index, "exp0") + rows
    return rows


def run(model, rows: list[dict], variants: list[str], k: int, size: int, out: Path, ann: Annotations,
        style: str = "struct") -> None:
    import torch

    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    log = (out / "runs.jsonl").open("a")
    for i, row in enumerate(rows):
        stem = Path(row["file"]).stem
        photo = np.array(Image.open(ann.image_dir / row["file"]).convert("RGB"))
        inner = ann.mask(row["file"], row["inner"]["ann_id"])
        outer = ann.mask(row["file"], row["outer"]["ann_id"])
        for variant in variants:
            vdir = out / variant
            targets = [vdir / f"{stem}_k{j}.png" for j in range(k)]
            if all(t.exists() for t in targets):
                continue
            vdir.mkdir(parents=True, exist_ok=True)
            if variant.startswith("lora"):
                from unpaired.pairs import pointer_refs

                refs = pointer_refs(photo, inner, style)
            else:
                refs = render(variant, photo, inner, outer)
            refs[0].save(vdir / f"{stem}_in.jpg", quality=90)
            torch.cuda.synchronize()
            start = time.perf_counter()
            images = model(refs, prompt_for(variant, row, style), list(range(k)), (size, size))
            torch.cuda.synchronize()
            sec = time.perf_counter() - start
            for t, im in zip(targets, images):
                im.save(t)
            rec = {"file": row["file"], "variant": variant, "k": k, "seconds": round(sec, 1),
                   "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
        print(f"[{i + 1}/{len(rows)}] {row['file']}", flush=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="학습 없는 이너 추출 기준선")
    p.add_argument("--model", choices=["klein", "qie"], required=True)
    p.add_argument("--variants", default=",".join(VARIANTS))
    p.add_argument("--index", default="data/unpaired/index/layered.jsonl")
    p.add_argument("--fashionpedia", default="data/fashionpedia")
    p.add_argument("--split", default="report")
    p.add_argument("--per-bin", type=int, default=4)
    p.add_argument("--include-exp0", action="store_true")
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--out", required=True)
    p.add_argument("--lora", help="학습한 LoRA 폴더(klein 전용). 주면 변형은 lora 하나")
    p.add_argument("--tag", help="출력 폴더 이름(기본: 모델 이름)")
    p.add_argument("--base", action="store_true", help="klein-base-4B 본체로 CFG 다단계 추론(LoRA 를 학습한 모델)")
    p.add_argument("--lora-variants", default="lora", help='lora 일 때 변형: "lora"(학습 지시문), "lora_natural"(자연어)')
    p.add_argument("--prompt-style", default="struct", help="lora 변형의 지시문 형식(학습 쌍과 같게): struct / natural")
    args = p.parse_args(argv)

    from unpaired import editors

    rows = select(Path(args.index), args.split, args.per_bin, args.include_exp0)
    ann = Annotations(Path(args.fashionpedia))
    model = editors.load(args.model, args.lora, base=args.base)
    variants = args.lora_variants.split(",") if args.lora else args.variants.split(",")
    run(model, rows, variants, args.k, args.size, Path(args.out) / (args.tag or model.name), ann, args.prompt_style)


if __name__ == "__main__":
    main()

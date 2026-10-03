"""데이터 엔진·합성기가 승인한 결과를 LoRA 학습 쌍(train_lora 의 jsonl)으로 바꾼다.

  swap      [EXTRACT] 사진 전체 + 이너 외곽선, 이너 크롭(밖은 어둡게)  → 은행 상품 P (swap_solo 도 같은 형식)
  layer     [PEEL]    사진 전체 + 겉옷 외곽선                        → 원본 사진
            [EXTRACT] 사진 전체 + 겉옷 외곽선, 겉옷 크롭              → 은행 겉옷 O
            [EXTRACT] 이너가 은행 상품이면(swap_solo 위에 덧입힘) 이너 외곽선 → 은행 상품 P
  flatlay   [EXTRACT] 장면 + 대상 외곽선, 대상 크롭                   → 은행 상품
  identity  [EXTRACT] 상품 사진 자체 + 외곽선                         → 같은 상품 사진

정답은 모두 실제 사진(은행 상품 또는 원본)이다. 참조 이미지는 refs/ 에 저장한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from unpaired.bank import load_usable
from unpaired.engine import prepare, stem_of
from unpaired.layered import Annotations
from unpaired.pointer import dim_crop, outline

EXTRACT_SIZE = (768, 768)
PEEL_LONG_SIDE = 768
REF_LONG_SIDE = 768
CROP_LONG_SIDE = 384


def extract_prompt(category: str, layer: str) -> str:
    return f"[EXTRACT] {category}; layer={layer}"


def peel_prompt(outer: str) -> str:
    return f"[PEEL] {outer}"


def fit(img: np.ndarray, long_side: int) -> Image.Image:
    im = Image.fromarray(img)
    s = long_side / max(im.size)
    w, h = max(16, int(im.width * s) // 16 * 16), max(16, int(im.height * s) // 16 * 16)
    return im.resize((w, h), Image.LANCZOS)


class Writer:
    def __init__(self, out: Path):
        self.out = out
        (out / "refs").mkdir(parents=True, exist_ok=True)
        (out / "targets").mkdir(parents=True, exist_ok=True)
        self.fh = (out / "pairs.jsonl").open("a")
        self.n = 0

    def add(self, pid: str, refs: list[Image.Image], target: str | Image.Image, prompt: str, size, meta: dict):
        paths = []
        for i, r in enumerate(refs):
            p = self.out / "refs" / f"{pid}_r{i}.jpg"
            r.save(p, quality=93)
            paths.append(str(p))
        if isinstance(target, Image.Image):
            tp = self.out / "targets" / f"{pid}.jpg"
            target.save(tp, quality=95)
            target = str(tp)
        self.fh.write(json.dumps({"id": pid, "refs": paths, "target": target, "prompt": prompt,
                                  "size": list(size), **meta}, ensure_ascii=False) + "\n")
        self.n += 1


def pointer_refs(img: np.ndarray, mask: np.ndarray) -> list[Image.Image]:
    """참조 1: 사진 전체 + 외곽선, 참조 2: 대상 주변 크롭(대상 밖 어둡게)."""
    return [fit(outline(img, mask), REF_LONG_SIDE), fit(dim_crop(img, mask), CROP_LONG_SIDE)]


def from_swap(w: Writer, engine_dir: Path, ann: Annotations, bank: Path, index: dict[str, dict]) -> None:
    for rec in map(json.loads, (engine_dir / "attempts.jsonl").read_text().splitlines()):
        if not rec["approved"]:
            continue
        row = index[rec["file"]]
        photo = np.array(Image.open(ann.image_dir / rec["file"]).convert("RGB"))
        _, (inner,) = prepare(photo, [ann.mask(rec["file"], row["inner"]["ann_id"])], rec.get("max_side", 1024))
        x = np.array(Image.open(engine_dir / "input" / f"{stem_of(rec)}.jpg").convert("RGB"))
        solo = "outer" not in row  # swap_solo: 겉옷 없이 다 보이는 상의를 바꾼 것
        layer = "single" if solo else f"inner under {row['outer']['category']}"
        source = "swap_solo" if solo else "swap"
        w.add(f"{source}_{stem_of(rec)}", pointer_refs(x, inner), str(bank / "front" / f"{rec['product']}.jpg"),
              extract_prompt(rec["product_category"], layer), EXTRACT_SIZE,
              {"source": source, "share_bin": rec.get("share_bin"), "product": rec["product"]})


def from_layer(w: Writer, engine_dir: Path, ann: Annotations, bank: Path, index: dict[str, dict]) -> None:
    for rec in map(json.loads, (engine_dir / "attempts.jsonl").read_text().splitlines()):
        if not rec["approved"]:
            continue
        stem = stem_of(rec)
        row = index[rec["file"]]
        photo = np.array(Image.open(ann.image_dir / rec["file"]).convert("RGB"))
        orig, (top,) = prepare(photo, [ann.mask(rec["file"], row["inner"]["ann_id"])], rec.get("max_side", 1024))
        if rec.get("base"):  # swap_solo 로 상의를 바꾼 사진 위에 덧입힌 경우, 그 사진이 벗기기 정답
            orig = np.array(Image.open(rec["base"]).convert("RGB"))
        x = np.array(Image.open(engine_dir / "input" / f"{stem}.jpg").convert("RGB"))
        jacket = np.asarray(Image.open(engine_dir / "input" / f"{stem}_jacket.png")) > 127
        peel_target = fit(orig, PEEL_LONG_SIDE)
        w.add(f"peel_{stem}", [fit(outline(x, jacket), REF_LONG_SIDE)], peel_target, peel_prompt(rec["product_category"]),
              peel_target.size, {"source": "layer_peel"})
        w.add(f"outer_{stem}", pointer_refs(x, jacket), str(bank / "front" / f"{rec['product']}.jpg"),
              extract_prompt(rec["product_category"], "outer"), EXTRACT_SIZE,
              {"source": "layer_outer", "product": rec["product"]})
        if rec.get("inner_product"):  # 이너도 은행 상품이므로 실제 정답이 있다
            w.add(f"inner_{stem}", pointer_refs(x, top & ~jacket), str(bank / "front" / f"{rec['inner_product']}.jpg"),
                  extract_prompt(rec["inner_product_category"], f"inner under {rec['product_category']}"),
                  EXTRACT_SIZE, {"source": "layer_inner", "product": rec["inner_product"]})


def from_flatlay(w: Writer, flat_dir: Path, bank: Path) -> None:
    for scene in map(json.loads, (flat_dir / "scenes.jsonl").read_text().splitlines()):
        img = np.array(Image.open(flat_dir / f"{scene['scene']}.jpg").convert("RGB"))
        labels = np.asarray(Image.open(flat_dir / f"{scene['scene']}_labels.png"))
        for it in scene["items"]:
            if not it["target"]:
                continue
            mask = labels == it["label"]
            w.add(f"flat_{scene['scene']}_{it['label']}", pointer_refs(img, mask),
                  str(bank / "front" / f"{it['item']}.jpg"), extract_prompt(it["category"], "flat"), EXTRACT_SIZE,
                  {"source": "flatlay", "product": it["item"], "visible_frac": it["visible_frac"]})


def from_identity(w: Writer, bank: Path, n: int, seed: int = 0) -> None:
    rows = load_usable(bank)
    rng = np.random.default_rng(seed)
    for j in rng.choice(len(rows), min(n, len(rows)), replace=False):
        r = rows[j]
        img = np.array(Image.open(bank / "front" / f"{r['item']}.jpg").convert("RGB"))
        mask = np.asarray(Image.open(bank / "front_mask" / f"{r['item']}.png")) > 127
        w.add(f"ident_{r['item']}", pointer_refs(img, mask), str(bank / "front" / f"{r['item']}.jpg"),
              extract_prompt(r["category"], "single"), EXTRACT_SIZE, {"source": "identity", "product": r["item"]})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="승인 결과 → LoRA 학습 쌍")
    p.add_argument("--out", required=True)
    p.add_argument("--swap", action="append", default=[], help="engine swap 출력 폴더(여러 번 가능)")
    p.add_argument("--layer", action="append", default=[])
    p.add_argument("--flatlay", action="append", default=[])
    p.add_argument("--identity", type=int, default=0, help="상품 그대로 쌍 개수")
    p.add_argument("--bank", default="data/unpaired/bank")
    p.add_argument("--index-dir", default="data/unpaired/index")
    p.add_argument("--fashionpedia", default="data/fashionpedia")
    args = p.parse_args(argv)

    w = Writer(Path(args.out))
    bank = Path(args.bank)
    if args.swap or args.layer:
        ann = Annotations(Path(args.fashionpedia))
        index = {}
        for name in ("layered.jsonl", "solo_tops.jsonl"):
            for line in (Path(args.index_dir) / name).read_text().splitlines():
                r = json.loads(line)
                index[r["file"]] = r
        for d in args.swap:
            from_swap(w, Path(d), ann, bank, index)
        for d in args.layer:
            from_layer(w, Path(d), ann, bank, index)
    for d in args.flatlay:
        from_flatlay(w, Path(d), bank)
    if args.identity:
        from_identity(w, bank, args.identity)
    print(json.dumps({"pairs": w.n, "out": args.out}))


if __name__ == "__main__":
    main()

"""데이터 엔진·합성기가 승인한 결과를 LoRA 학습 쌍(train_lora 의 jsonl)으로 바꾼다.

  swap      [EXTRACT] 사진 전체 + 이너 외곽선, 이너 크롭(밖은 어둡게)  → 은행 상품 P (swap_solo 도 같은 형식)
  layer     [PEEL]    사진 전체 + 겉옷 외곽선                        → 원본 사진
            [EXTRACT] 사진 전체 + 겉옷 외곽선, 겉옷 크롭              → 은행 겉옷 O
            [EXTRACT] 이너가 은행 상품이면(swap_solo 위에 덧입힘) 이너 외곽선 → 은행 상품 P
  flatlay   [EXTRACT] 장면 + 대상 외곽선, 대상 크롭                   → 은행 상품
  identity  [EXTRACT] 상품 사진 자체 + 외곽선                         → 같은 상품 사진

정답은 모두 실제 사진(은행 상품 또는 원본)이다. 참조 이미지는 refs/ 에 저장한다.
쌍 이름에 출력 폴더 이름을 넣는다. 같은 사진을 여러 실행에서 썼을 때(상품은 다름) 겹치지 않게 하려고.
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
from unpaired.weights import extract_weights, peel_weights, visible_width_ratio

EXTRACT_SIZE = (768, 768)
PEEL_LONG_SIDE = 768
REF_LONG_SIDE = 768
CROP_LONG_SIDE = 384


# 지시문 형식. struct = 새 형식 문구(v1). natural = 증류 모델이 이미 아는 자연어(학습 없는 outline_dimcrop 과 같은 문장).
# 2026-10-03 진단: base 에서 struct 로 학습한 LoRA 는 4스텝 증류 모델에서 입력을 그대로 다시 그렸다.
PROMPT_STYLES = ("struct", "natural", "generic", "dim", "dimfine")
SOLO_PRODUCT = ("Create a store product photo of {what}: the garment alone, laid flat and neatly smoothed, front view, "
                "centered on a plain white background. Keep its exact colors, pattern, print and details. "
                "Show only this one garment: no person, no other clothing, no other objects.")
NATURAL_PEEL = ("Remove the {outer} marked with the green outline from the person in image 1. Keep the person, face, "
                "hair, pose, the clothes underneath and the background exactly the same.")


# v3: 대상을 넓은 이름(top 등)으로만 부른다. v2 엄격 판정에서 캐미솔·민소매·터틀넥을 "t-shirt" 라고 부르니
# 평범한 반팔 티로 바꿔 그렸다(84건 중 10건). 실제 앱에서도 세부 종류는 모르므로 모델이 사진에서 알아내게 한다.
GENERIC = {"inner": "top", "outer": "outer garment", "bottom": "bottoms", "onepiece": "dress"}
_FINE_TO_GROUP = {fine: group for group, fine in __import__("unpaired.bank", fromlist=["CATEGORY"]).CATEGORY.values()}
_FINE_TO_GROUP.update({"t-shirt": "inner", "shirt": "inner", "sweater": "inner", "jacket": "outer", "coat": "outer",
                       "cardigan": "outer", "vest": "outer", "cape": "outer", "trousers": "bottom", "pants": "bottom"})


def generic(category: str) -> str:
    return GENERIC.get(_FINE_TO_GROUP.get(category, ""), category)


# v3 "dim": 사진에 아무 표시도 그리지 않는다. v2 는 초록 외곽선이 결과물의 목 테두리로 새어 나왔다(엄격 판정의 가장 큰
# 단일 실패 원인). 지시문에서 "초록 표시를 그리지 마라" 를 빼면 오히려 더 새어 나왔다(28건 중 6 → 16). 그래서 표시 대신
# 대상만 밝고 나머지는 어둡게 한 확대 이미지(image 2)로 가리킨다. 대상 이름은 generic 처럼 넓게 부른다.
DIM_WHAT = {"inner": "the {cat} shown bright in image 2, which is worn under the {outer} in the photo (image 1)",
            "flat": "the {cat} shown bright in image 2, which lies among other things in the photo (image 1)",
            "outer": "the {cat} shown bright in image 2, which is the outer layer in the photo (image 1)",
            "single": "the {cat} shown bright in image 2 (a close-up of the photo, image 1)"}
DIM_PRODUCT = ("Create a store product photo of {what}: the garment alone, laid flat and neatly smoothed, front view, "
               "centered on a plain white background. Keep its exact colors, pattern, print and details. Show only this "
               "one garment: no person, no other clothing, no accessories, no other objects. In image 2 everything "
               "except the garment is darkened only to point at it.")


def extract_prompt(category: str, layer: str, style: str = "struct") -> str:
    if style in ("dim", "dimfine"):  # dimfine: 넓은 이름 대신 세부 종류(trousers, skirt …)로 부른다
        cat = generic(category) if style == "dim" else category
        if layer.startswith("inner under "):
            what = DIM_WHAT["inner"].format(cat=cat, outer=layer[len("inner under "):])
        else:
            what = DIM_WHAT.get(layer, DIM_WHAT["single"]).format(cat=cat)
        return DIM_PRODUCT.format(what=what)
    if style == "generic":
        category, style = generic(category), "natural"
    if style == "struct":
        return f"[EXTRACT] {category}; layer={layer}"
    from unpaired.zeroshot import MARK_NOTE, PRODUCT, WHAT

    if layer.startswith("inner under "):
        outer = layer[len("inner under "):]
        return PRODUCT.format(what=WHAT["outline_dimcrop"].format(cat=category, outer=outer), outer=outer) + MARK_NOTE
    where = {"flat": "; it lies among other things", "outer": "; it is the outer layer"}.get(layer, "")
    what = f"the {category} marked with the green outline in image 1 (image 2 is a close-up of it{where})"
    return SOLO_PRODUCT.format(what=what) + MARK_NOTE


def peel_prompt(outer: str, style: str = "struct") -> str:
    if style == "struct":
        return f"[PEEL] {outer}"
    from unpaired.zeroshot import MARK_NOTE

    return NATURAL_PEEL.format(outer=outer) + MARK_NOTE


def fit(img: np.ndarray, long_side: int) -> Image.Image:
    im = Image.fromarray(img)
    s = long_side / max(im.size)
    w, h = max(16, int(im.width * s) // 16 * 16), max(16, int(im.height * s) // 16 * 16)
    return im.resize((w, h), Image.LANCZOS)


class Writer:
    def __init__(self, out: Path, hidden_weights: bool = False, style: str = "struct", max_palette: float | None = None):
        self.out = out
        self.max_palette = max_palette  # 엔진 결과 속 옷 색과 정답 상품 색의 팔레트 거리 상한(넘으면 버린다)
        self.skipped = 0
        self.style = style
        self.hidden_weights = hidden_weights  # 가려졌던 곳의 무늬·글자는 채점하지 않는 가중치(unpaired.weights)
        (out / "refs").mkdir(parents=True, exist_ok=True)
        (out / "targets").mkdir(parents=True, exist_ok=True)
        (out / "weights").mkdir(parents=True, exist_ok=True)
        self.fh = (out / "pairs.jsonl").open("a")
        self.n = 0
        self.seen: set[str] = set()

    def add(self, pid: str, refs: list[Image.Image], target: str | Image.Image, prompt: str, size, meta: dict):
        if pid in self.seen:  # 같은 이름이면 참조 이미지를 덮어써서 다른 쌍의 입력이 바뀐다
            raise ValueError(f"쌍 이름이 겹칩니다: {pid}")
        self.seen.add(pid)
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

    def weight(self, pid: str, image: Image.Image) -> dict:
        """가중치 이미지를 저장하고 쌍에 붙일 meta 를 돌려준다. 옵션이 꺼져 있으면 빈 dict."""
        if not self.hidden_weights:
            return {}
        path = self.out / "weights" / f"{pid}.png"
        image.save(path)
        return {"weight": str(path)}


def product_weight(w: Writer, pid: str, bank: Path, item: str, visible: np.ndarray, torso: np.ndarray) -> dict:
    if not w.hidden_weights:
        return {}
    img = np.array(Image.open(bank / "front" / f"{item}.jpg").convert("RGB"))
    mask = np.asarray(Image.open(bank / "front_mask" / f"{item}.png")) > 127
    return w.weight(pid, extract_weights(img, mask, visible_width_ratio(visible, torso), EXTRACT_SIZE))


def pointer_refs(img: np.ndarray, mask: np.ndarray, style: str = "natural") -> list[Image.Image]:
    """참조 1: 사진 전체 + 외곽선(style "dim" 이면 표시 없음), 참조 2: 대상 주변 크롭(대상 밖 어둡게)."""
    full = img if style == "dim" else outline(img, mask)
    return [fit(full, REF_LONG_SIDE), fit(dim_crop(img, mask), CROP_LONG_SIDE)]


def from_swap(w: Writer, engine_dir: Path, ann: Annotations, bank: Path, index: dict[str, dict]) -> None:
    for rec in map(json.loads, (engine_dir / "attempts.jsonl").read_text().splitlines()):
        if not rec["approved"]:
            continue
        if w.max_palette is not None and rec["scores"].get("palette_dist", 0) > w.max_palette:
            w.skipped += 1  # 사진 속 색과 정답 색이 다르면 "색을 바꿔도 된다" 고 가르친다
            continue
        row = index[rec["file"]]
        photo = np.array(Image.open(ann.image_dir / rec["file"]).convert("RGB"))
        _, (inner,) = prepare(photo, [ann.mask(rec["file"], row["inner"]["ann_id"])], rec.get("max_side", 1024))
        x = np.array(Image.open(engine_dir / "input" / f"{stem_of(rec)}.jpg").convert("RGB"))
        solo = "outer" not in row  # swap_solo: 겉옷 없이 다 보이는 상의를 바꾼 것
        layer = "single" if solo else f"inner under {row['outer']['category']}"
        source = "swap_solo" if solo else "swap"
        pid = f"{source}_{engine_dir.name}_{stem_of(rec)}"
        weight = {}
        if not solo:  # 겉옷 아래 이너만 가려진 곳이 있다
            _, (_, outer) = prepare(photo, [ann.mask(rec["file"], row["inner"]["ann_id"]),
                                            ann.mask(rec["file"], row["outer"]["ann_id"])], rec.get("max_side", 1024))
            weight = product_weight(w, pid, bank, rec["product"], inner, inner | outer)
        w.add(pid, pointer_refs(x, inner, w.style), str(bank / "front" / f"{rec['product']}.jpg"),
              extract_prompt(rec["product_category"], layer, w.style), EXTRACT_SIZE,
              {"source": source, "share_bin": rec.get("share_bin"), "product": rec["product"], **weight})


def from_layer(w: Writer, engine_dir: Path, ann: Annotations, bank: Path, index: dict[str, dict],
               base_palette: dict[str, float] | None = None) -> None:
    """base_palette: 덧입힌 바탕 사진(swap_solo 결과) 이름 → 그 교체의 팔레트 거리."""
    base_palette = base_palette or {}
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
        pid = f"peel_{engine_dir.name}_{stem}"
        weight = {}
        if w.hidden_weights:
            hidden = np.asarray(Image.fromarray(jacket.astype(np.uint8) * 255).resize(peel_target.size, Image.NEAREST)) > 127
            weight = w.weight(pid, peel_weights(np.asarray(peel_target), hidden))
        w.add(pid, [fit(outline(x, jacket), REF_LONG_SIDE)], peel_target, peel_prompt(rec["product_category"], w.style),
              peel_target.size, {"source": "layer_peel", **weight})
        w.add(f"outer_{engine_dir.name}_{stem}", pointer_refs(x, jacket, w.style), str(bank / "front" / f"{rec['product']}.jpg"),
              extract_prompt(rec["product_category"], "outer", w.style), EXTRACT_SIZE,
              {"source": "layer_outer", "product": rec["product"]})
        base_pal = base_palette.get(Path(rec["base"]).stem) if rec.get("base") else None
        if rec.get("inner_product") and (w.max_palette is None or base_pal is None or base_pal <= w.max_palette):
            # 이너도 은행 상품이므로 실제 정답이 있다
            pid = f"inner_{engine_dir.name}_{stem}"
            weight = product_weight(w, pid, bank, rec["inner_product"], top & ~jacket, top | jacket)
            w.add(pid, pointer_refs(x, top & ~jacket, w.style), str(bank / "front" / f"{rec['inner_product']}.jpg"),
                  extract_prompt(rec["inner_product_category"], f"inner under {rec['product_category']}", w.style),
                  EXTRACT_SIZE, {"source": "layer_inner", "product": rec["inner_product"], **weight})


def from_flatlay(w: Writer, flat_dir: Path, bank: Path) -> None:
    for scene in map(json.loads, (flat_dir / "scenes.jsonl").read_text().splitlines()):
        img = np.array(Image.open(flat_dir / f"{scene['scene']}.jpg").convert("RGB"))
        labels = np.asarray(Image.open(flat_dir / f"{scene['scene']}_labels.png"))
        for it in scene["items"]:
            if not it["target"]:
                continue
            mask = labels == it["label"]
            w.add(f"flat_{flat_dir.name}_{scene['scene']}_{it['label']}", pointer_refs(img, mask, w.style),
                  str(bank / "front" / f"{it['item']}.jpg"), extract_prompt(it["category"], "flat", w.style), EXTRACT_SIZE,
                  {"source": "flatlay", "product": it["item"], "visible_frac": it["visible_frac"]})


def from_identity(w: Writer, bank: Path, n: int, seed: int = 0) -> None:
    rows = load_usable(bank)
    rng = np.random.default_rng(seed)
    for j in rng.choice(len(rows), min(n, len(rows)), replace=False):
        r = rows[j]
        img = np.array(Image.open(bank / "front" / f"{r['item']}.jpg").convert("RGB"))
        mask = np.asarray(Image.open(bank / "front_mask" / f"{r['item']}.png")) > 127
        w.add(f"ident_{r['item']}", pointer_refs(img, mask, w.style), str(bank / "front" / f"{r['item']}.jpg"),
              extract_prompt(r["category"], "single", w.style), EXTRACT_SIZE, {"source": "identity", "product": r["item"]})


def from_swap_accessories(w: Writer, engine_dirs: list[Path], ann: Annotations, index: dict[str, dict], bank: Path,
                          pieces_dir: Path, n: int, seed: int = 2) -> None:
    """엔진이 상의를 바꾼 사람 사진의 이너 위에 실제 넥타이·스카프·가방 조각을 얹은 입력 → 원래 은행 상품.

    v3 엄격 판정에서도 넥타이·목걸이를 같이 그렸다(84건 중 10건). 엔진은 상의를 바꾸며 넥타이까지 지운 경우가 많아
    (넥타이 사진 160건 중 승인 55건, 대부분 넥타이 없음) "사람이 맨 넥타이가 있는 사진 → 넥타이 없는 상품" 예가 거의
    없었다. 상품 위에 얹은 장신구 쌍(from_accessories)과 달리 실제 착용 사진 위에 얹는다."""
    from unpaired import accessories

    pieces = [p for p in accessories.load(pieces_dir) if p["kind"] in ("tie", "scarf", "bag")]
    kinds = {k: [p for p in pieces if p["kind"] == k] for k in ("tie", "scarf", "bag")}
    recs = []
    for d in engine_dirs:
        for rec in map(json.loads, (d / "attempts.jsonl").read_text().splitlines()):
            if rec["approved"] and (w.max_palette is None or rec["scores"].get("palette_dist", 0) <= w.max_palette):
                recs.append((d, rec))
    rng = np.random.default_rng(seed)
    rng.shuffle(recs)
    made = 0
    for d, rec in recs:
        if made >= n:
            break
        row = index[rec["file"]]
        photo = np.array(Image.open(ann.image_dir / rec["file"]).convert("RGB"))
        _, (inner,) = prepare(photo, [ann.mask(rec["file"], row["inner"]["ann_id"])], rec.get("max_side", 1024))
        x = np.array(Image.open(d / "input" / f"{stem_of(rec)}.jpg").convert("RGB"))
        if inner.shape != x.shape[:2]:
            continue
        kind = rng.choice(["tie", "scarf", "bag"], p=[0.6, 0.25, 0.15])
        pc = kinds[kind][rng.integers(len(kinds[kind]))]
        piece = Image.open(pieces_dir / pc["file"]).convert("RGBA")
        x2, acc, covered = accessories.place(x, inner, piece, kind, rng)
        if covered.sum() < 0.04 * inner.sum() or (inner & ~acc).sum() < 0.4 * inner.sum():
            continue
        solo = "outer" not in row
        layer = "single" if solo else f"inner under {row['outer']['category']}"
        w.add(f"swapacc_{d.name}_{stem_of(rec)}", pointer_refs(x2, inner & ~acc, w.style),
              str(bank / "front" / f"{rec['product']}.jpg"), extract_prompt(rec["product_category"], layer, w.style),
              EXTRACT_SIZE, {"source": "swap_acc", "product": rec["product"], "accessory": kind})
        made += 1


def from_accessories(w: Writer, bank: Path, pieces_dir: Path, n: int, seed: int = 1) -> None:
    """상품 위에 넥타이·벨트·스카프·가방을 얹은 입력 → 깨끗한 원래 상품 (unpaired.accessories)."""
    from unpaired import accessories

    rows = [r for r in load_usable(bank) if r["group"] in ("inner", "onepiece")]
    pieces = accessories.load(pieces_dir)
    rng = np.random.default_rng(seed)
    made = tries = 0
    while made < n and tries < 3 * n:
        tries += 1
        r = rows[rng.integers(len(rows))]
        pc = pieces[rng.integers(len(pieces))]
        img = np.array(Image.open(bank / "front" / f"{r['item']}.jpg").convert("RGB"))
        mask = np.asarray(Image.open(bank / "front_mask" / f"{r['item']}.png")) > 127
        piece = Image.open(pieces_dir / pc["file"]).convert("RGBA")
        x, acc, covered = accessories.place(img, mask, piece, pc["kind"], rng)
        if covered.sum() < 0.02 * mask.sum() or (mask & ~acc).sum() < 0.5 * mask.sum():
            continue
        w.add(f"acc_{r['item']}_{made}", pointer_refs(x, mask & ~acc, w.style), str(bank / "front" / f"{r['item']}.jpg"),
              extract_prompt(r["category"], "single", w.style), EXTRACT_SIZE,
              {"source": "accessory", "product": r["item"], "accessory": pc["kind"]})
        made += 1


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
    p.add_argument("--hidden-weights", action="store_true", help="가려졌던 곳의 무늬·글자는 채점하지 않는 가중치를 붙인다")
    p.add_argument("--prompt-style", choices=PROMPT_STYLES, default="struct")
    p.add_argument("--accessories", type=int, default=0, help="장신구를 얹은 상품 쌍 개수")
    p.add_argument("--accessory-dir", default="data/unpaired/accessories")
    p.add_argument("--swap-accessories", type=int, default=0, help="엔진 교체 사진 위에 장신구를 얹은 쌍 개수(--swap 폴더 사용)")
    p.add_argument("--max-palette", type=float, help="엔진 결과 색이 정답과 이만큼 넘게 다르면 버린다 (v3: 6)")
    args = p.parse_args(argv)

    w = Writer(Path(args.out), args.hidden_weights, args.prompt_style, args.max_palette)
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
        base_palette = {}
        for d in args.swap:
            for rec in map(json.loads, (Path(d) / "attempts.jsonl").read_text().splitlines()):
                if rec["approved"]:
                    base_palette[stem_of(rec)] = rec["scores"].get("palette_dist", 0)
        for d in args.layer:
            from_layer(w, Path(d), ann, bank, index, base_palette)
        if args.swap_accessories:
            from_swap_accessories(w, [Path(d) for d in args.swap], ann, index, bank, Path(args.accessory_dir),
                                  args.swap_accessories)
    for d in args.flatlay:
        from_flatlay(w, Path(d), bank)
    if args.identity:
        from_identity(w, bank, args.identity)
    if args.accessories:
        from_accessories(w, bank, Path(args.accessory_dir), args.accessories)
    print(json.dumps({"pairs": w.n, "skipped_color": w.skipped, "out": args.out}))


if __name__ == "__main__":
    main()

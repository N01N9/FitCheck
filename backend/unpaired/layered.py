"""Fashionpedia(상업 이용 가능 부분)에서 실험에 쓸 사진을 골라 색인하고, 평가셋을 고정 분할한다.

  layered.jsonl     겉옷 1벌 + 이너 상의 1벌이 겹친 사진. 이너가 겉옷의 볼록 껍질 안에 절반 이상
                    들어 있으면 "겉옷 아래 이너"로 본다. 이너 추출 실험·데이터 엔진 A(이너 교체)의 재료
  solo_tops.jsonl   겉옷 없이 상의 1벌이 거의 다 보이는 사진. 데이터 엔진 B(겉옷 덧입히기)와
                    E-실제숨김 평가셋(실제 픽셀 정답)의 재료

가시율은 알 수 없으므로 대신 `inner_share = 이너 면적 / (이너 + 겉옷 면적)` 을 쓴다.
분할은 파일 이름 해시로 정해서 코드만으로 다시 만들 수 있다. 같은 사진은 한 분할에만 들어간다.

사용
  python -m unpaired.layered --fashionpedia data/fashionpedia --out data/unpaired/index
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from unpaired.masks import bbox, convex_hull, decode

INNER = {"shirt, blouse": "shirt", "top, t-shirt, sweatshirt": "t-shirt", "sweater": "sweater"}
OUTER = {"jacket": "jacket", "coat": "coat", "cardigan": "cardigan", "vest": "vest", "cape": "cape"}

# 실험 0 에서 쓴 사진은 학습·조정 분할에 넣지 않는다
EXP0_IMAGES = {
    "6fedbb43e23aeba31b8a70194a109b48.jpg", "200844f9646a4ae6ff3e8af37b6bf6da.jpg",
    "a29afba47d025d645c02793a7fce4c1c.jpg", "48c2c31b07d0e07db6dfba0f5d5d743a.jpg",
}

# [시작, 끝) 버킷(0~99). 임계값 조정·체크포인트 선택·최종 보고용을 서로 떼어 둔다
LAYERED_SPLITS = {"report": (0, 20), "select": (20, 25), "tune": (25, 30), "train": (30, 100)}
SOLO_SPLITS = {"hidden_report": (0, 25), "hidden_tune": (25, 35), "train": (35, 100)}

MIN_INSIDE = 0.5        # 이너가 겉옷 볼록 껍질 안에 들어 있는 비율
MIN_SOLO_AREA = 0.03    # 단독 상의가 사진에서 차지하는 최소 비율
MIN_SOLO_SOLIDITY = 0.75  # 면적 / 볼록 껍질 면적. 낮으면 팔·머리카락에 가려 구멍이 난 것


def bucket(name: str) -> int:
    return int(hashlib.sha1(name.encode()).hexdigest()[:8], 16) % 100


def split_of(name: str, splits: dict[str, tuple[int, int]]) -> str:
    if name in EXP0_IMAGES:
        return "exp0"
    b = bucket(name)
    return next(k for k, (lo, hi) in splits.items() if lo <= b < hi)


def share_bin(share: float) -> str:
    return "lt15" if share < 0.15 else "15_40" if share < 0.40 else "40_70" if share < 0.70 else "ge70"


def attribute_names(attr_ids: list[int], vocab: dict[int, tuple[str, str]]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for a in attr_ids:
        if a in vocab:
            sup, name = vocab[a]
            out.setdefault(sup, []).append(name)
    return out


def garment(ann: dict, kind: str, mask: np.ndarray, vocab) -> dict:
    return {"ann_id": ann["id"], "category": kind, "area": int(mask.sum()), "bbox": bbox(mask),
            "attributes": attribute_names(ann.get("attribute_ids", []), vocab)}


def index_image(im: dict, anns: list[dict], cat_name: dict[int, str], vocab) -> tuple[str, dict] | None:
    inners = [a for a in anns if cat_name[a["category_id"]] in INNER]
    outers = [a for a in anns if cat_name[a["category_id"]] in OUTER]
    # 여러 명이 찍혔거나 겹쳐 입은 이너가 여럿이면 어느 옷이 어느 옷 아래인지 애매해서 뺀다
    if len(inners) != 1 or len(outers) > 1:
        return None
    h, w = im["height"], im["width"]
    inner_ann = inners[0]
    inner = decode(inner_ann["segmentation"], h, w)
    if inner.sum() == 0:
        return None
    base = {"file": im["file_name"], "width": w, "height": h, "license": im.get("license"),
            "inner": garment(inner_ann, INNER[cat_name[inner_ann["category_id"]]], inner, vocab)}
    if outers:
        outer_ann = outers[0]
        outer = decode(outer_ann["segmentation"], h, w)
        if outer.sum() == 0:
            return None
        inside = float((inner & convex_hull(outer)).sum() / inner.sum())
        if inside < MIN_INSIDE:
            return None
        share = float(inner.sum() / (inner.sum() + outer.sum()))
        return "layered", {**base, "outer": garment(outer_ann, OUTER[cat_name[outer_ann["category_id"]]], outer, vocab),
                           "inside": round(inside, 3), "inner_share": round(share, 3), "share_bin": share_bin(share),
                           "split": split_of(im["file_name"], LAYERED_SPLITS)}
    area = float(inner.sum() / (h * w))
    solidity = float(inner.sum() / max(convex_hull(inner).sum(), 1))
    if area < MIN_SOLO_AREA or solidity < MIN_SOLO_SOLIDITY:
        return None
    return "solo", {**base, "area_frac": round(area, 4), "solidity": round(solidity, 3),
                    "split": split_of(im["file_name"], SOLO_SPLITS)}


def build(fashionpedia: Path, out: Path) -> dict:
    vocab_src = json.loads((fashionpedia / "instances_attributes_val2020.json").read_text())
    vocab = {a["id"]: (a["supercategory"], a["name"]) for a in vocab_src["attributes"]}
    rows: dict[str, list[dict]] = {"layered": [], "solo": []}
    for split in ("train", "val"):
        data = json.loads((fashionpedia / "commercial" / f"annotations_{split}.json").read_text())
        cat_name = {c["id"]: c["name"] for c in data["categories"]}
        by_image: dict[int, list[dict]] = {}
        for a in data["annotations"]:
            by_image.setdefault(a["image_id"], []).append(a)
        for im in data["images"]:
            hit = index_image(im, by_image.get(im["id"], []), cat_name, vocab)
            if hit:
                rows[hit[0]].append(hit[1])
    out.mkdir(parents=True, exist_ok=True)
    for kind, items in rows.items():
        items.sort(key=lambda r: r["file"])
        with (out / f"{'layered' if kind == 'layered' else 'solo_tops'}.jsonl").open("w") as fh:
            for r in items:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "layered": {"total": len(rows["layered"]),
                    "split": dict(Counter(r["split"] for r in rows["layered"])),
                    "share_bin": dict(Counter(r["share_bin"] for r in rows["layered"])),
                    "report_share_bin": dict(Counter(r["share_bin"] for r in rows["layered"] if r["split"] == "report")),
                    "pair": dict(Counter(f"{r['inner']['category']}<{r['outer']['category']}" for r in rows["layered"]))},
        "solo": {"total": len(rows["solo"]), "split": dict(Counter(r["split"] for r in rows["solo"])),
                 "category": dict(Counter(r["inner"]["category"] for r in rows["solo"]))},
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def load(path: Path, split: str | None = None) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    return [r for r in rows if split is None or r["split"] == split]


def sample_by_bin(rows: list[dict], per_bin: int, bins=("lt15", "15_40", "40_70")) -> list[dict]:
    """가시율 구간마다 해시 순서로 앞에서 per_bin 개씩. 다시 돌려도 같은 사진이 뽑힌다."""
    out = []
    for b in bins:
        hits = sorted((r for r in rows if r.get("share_bin") == b), key=lambda r: hashlib.sha1(r["file"].encode()).hexdigest())
        out.extend(hits[:per_bin])
    return out


class Annotations:
    """Fashionpedia 상업 부분 주석에서 마스크를 꺼낸다.

    train 과 val 의 주석 번호가 일부 겹치므로(80개) 항상 (사진 파일, 주석 번호) 로 찾는다.
    """

    def __init__(self, fashionpedia: Path):
        self.anns: dict[tuple[str, int], dict] = {}
        self.by_file: dict[str, list[dict]] = {}
        self.images: dict[str, dict] = {}
        for split in ("train", "val"):
            data = json.loads((fashionpedia / "commercial" / f"annotations_{split}.json").read_text())
            cat_name = {c["id"]: c["name"] for c in data["categories"]}
            images = {im["id"]: im for im in data["images"]}
            for a in data["annotations"]:
                im = images[a["image_id"]]
                a = {**a, "category": cat_name[a["category_id"]]}
                self.images[im["file_name"]] = im
                self.anns[(im["file_name"], a["id"])] = a
                self.by_file.setdefault(im["file_name"], []).append(a)
        self.image_dir = fashionpedia / "commercial" / "images"

    def mask(self, file: str, ann_id: int) -> np.ndarray:
        im = self.images[file]
        return decode(self.anns[(file, ann_id)]["segmentation"], im["height"], im["width"])

    def garments(self, file: str, names=tuple(INNER) + tuple(OUTER) + ("pants", "shorts", "skirt", "dress", "jumpsuit")):
        """사진 속 옷 주석(검색 순위 비교용). 액세서리·신발은 뺀다."""
        return [a for a in self.by_file.get(file, []) if a["category"] in names]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Fashionpedia 겹쳐 입은 사진·단독 상의 사진 색인")
    p.add_argument("--fashionpedia", default="data/fashionpedia")
    p.add_argument("--out", default="data/unpaired/index")
    args = p.parse_args(argv)
    print(json.dumps(build(Path(args.fashionpedia), Path(args.out)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

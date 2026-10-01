"""Fashionpedia 에서 상업 이용 가능한 사진만 골라 학습용 COCO 데이터로 만든다.

Fashionpedia 주석은 CC BY 4.0 이지만 사진 저작권은 원 출처에 있다. 주석 JSON 의
images[].license 가 사진별 라이선스라서, 그걸로 거른다.
  - 허용: CC BY 2.0(0), CC BY-SA 2.0(1), Public Domain Mark(6), freestocks(9, CC0)
  - 제외: NC 계열(3,4,5), ND(2), 저작권 보유/확인 불가(11)
  - 보류: unsplash(7), pexels(8), burst(10) — --include-held 로만 넣는다

출력 (out/)
  images/<file_name>        허용 사진만 zip 에서 꺼낸다(비상업 사진은 풀지 않는다)
  annotations_<split>.json  옷 단위 27종만 남긴 COCO. category 마다 우리 분류(coarse)를 붙인다
  sources.csv               사진별 원본 URL·라이선스

사용
  python -m phase0.prepare_fashionpedia --src data/fashionpedia --out data/fashionpedia/commercial
"""

from __future__ import annotations

import argparse
import csv
import json
import zipfile
from collections import Counter
from pathlib import Path

ALLOWED = {0, 1, 6, 9}
HELD = {7, 8, 10}

# Fashionpedia 0~26 은 옷·잡화 단위, 27~45 는 소매·칼라 같은 부위라 뺀다
COARSE = {
    "shirt, blouse": "상의", "top, t-shirt, sweatshirt": "상의", "sweater": "상의", "vest": "상의",
    "cardigan": "아우터", "jacket": "아우터", "coat": "아우터", "cape": "아우터",
    "pants": "하의", "shorts": "하의", "skirt": "하의",
    "dress": "원피스", "jumpsuit": "원피스",
    "shoe": "신발",
    "bag, wallet": "가방",
    "hat": "모자", "headband, head covering, hair accessory": "모자",
    "glasses": "액세서리", "tie": "액세서리", "glove": "액세서리", "watch": "액세서리",
    "belt": "액세서리", "leg warmer": "액세서리", "tights, stockings": "액세서리",
    "sock": "액세서리", "scarf": "액세서리", "umbrella": "액세서리",
}

SPLITS = {
    "train": ("instances_attributes_train2020.json", "train2020.zip"),
    "val": ("instances_attributes_val2020.json", "val_test2020.zip"),
}


def filter_coco(data: dict, allowed: set[int]) -> tuple[dict, Counter]:
    lic_name = {lic["id"]: lic["name"] for lic in data["licenses"]}
    keep_cats = [c for c in data["categories"] if c["name"] in COARSE]
    cat_ids = {c["id"] for c in keep_cats}
    images = [im for im in data["images"] if im.get("license") in allowed]
    img_ids = {im["id"] for im in images}
    anns = [a for a in data["annotations"] if a["image_id"] in img_ids and a["category_id"] in cat_ids]
    out = {
        "info": {**data.get("info", {}), "note": "FitCheck: 상업 이용 가능 라이선스 사진, 옷 단위 클래스만"},
        "licenses": data["licenses"],
        "categories": [{**c, "coarse": COARSE[c["name"]]} for c in keep_cats],
        "images": images,
        "annotations": anns,
    }
    stats = Counter(lic_name.get(im.get("license"), "?") for im in images)
    return out, stats


def extract(zip_path: Path, names: set[str], dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            base = Path(info.filename).name
            if base in names and not (dest / base).exists():
                (dest / base).write_bytes(zf.read(info))
                n += 1
    return n


def prepare(src: Path, out: Path, allowed: set[int]) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    rows = []
    for split, (ann_name, zip_name) in SPLITS.items():
        ann_path = src / ann_name
        if not ann_path.exists():
            continue
        data = json.loads(ann_path.read_text())
        filtered, lic_stats = filter_coco(data, allowed)
        (out / f"annotations_{split}.json").write_text(json.dumps(filtered, ensure_ascii=False))
        names = {im["file_name"] for im in filtered["images"]}
        extracted = extract(src / zip_name, names, out / "images") if (src / zip_name).exists() else 0
        lic_name = {lic["id"]: (lic["name"], lic["url"]) for lic in data["licenses"]}
        for im in filtered["images"]:
            name, url = lic_name[im["license"]]
            rows.append({"file": im["file_name"], "split": split, "fashionpedia_id": im["id"],
                         "original_url": im.get("original_url", ""), "license": name, "license_url": url})
        report[split] = {
            "images": len(filtered["images"]),
            "annotations": len(filtered["annotations"]),
            "extracted": extracted,
            "licenses": dict(lic_stats),
            "per_coarse": dict(Counter(
                next(c["coarse"] for c in filtered["categories"] if c["id"] == a["category_id"])
                for a in filtered["annotations"])),
        }
    with (out / "sources.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "split", "fashionpedia_id", "original_url",
                                                "license", "license_url"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "prepare_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Fashionpedia 상업 이용 가능 부분 추출")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--include-held", action="store_true", help="unsplash/pexels/burst 사진도 넣는다")
    args = p.parse_args(argv)
    allowed = ALLOWED | (HELD if args.include_held else set())
    print(json.dumps(prepare(Path(args.src), Path(args.out), allowed), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""LVIS v1(COCO 이미지)에서 상업 이용 가능한 사진의 옷 마스크만 골라 학습용 COCO 데이터로 만든다.

  - LVIS 주석은 CC BY 4.0. 사진은 COCO 의 사진별 license 로 거른다.
    허용: CC BY 2.0(4), CC BY-SA 2.0(5), No known copyright restrictions(7), US Government Work(8)
  - 옷·잡화 클래스만 남기고 우리 분류(coarse)를 붙인다.
  - LVIS 는 모든 클래스를 다 칠하지 않았다(federated). 사진마다 neg_category_ids(확실히 없음),
    not_exhaustive_category_ids(다 칠하지 않음)를 옷 클래스로 줄여 그대로 남긴다 → 학습 때 손실 마스킹에 쓴다.
  - 이미지는 coco_url 에서 골라 받는다.

사용
  python -m phase0.prepare_lvis --src data/lvis --out data/lvis/commercial [--no-download]
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from phase0.collect_commons import USER_AGENT

ALLOWED = {4, 5, 7, 8}

COARSE = {
    # 상의
    "shirt": "상의", "polo_shirt": "상의", "blouse": "상의", "sweater": "상의", "sweatshirt": "상의",
    "jersey": "상의", "vest": "상의", "nightshirt": "상의",
    # 아우터
    "jacket": "아우터", "coat": "아우터", "trench_coat": "아우터", "parka": "아우터", "ski_parka": "아우터",
    "raincoat": "아우터", "cardigan": "아우터", "cape": "아우터", "cloak": "아우터", "poncho": "아우터",
    "lab_coat": "아우터", "robe": "아우터", "bathrobe": "아우터", "suit_(clothing)": "아우터",
    "dress_suit": "아우터", "tux": "아우터",
    # 하의
    "jean": "하의", "trousers": "하의", "short_pants": "하의", "skirt": "하의", "ballet_skirt": "하의",
    "sweat_pants": "하의", "legging_(clothing)": "하의",
    # 원피스
    "dress": "원피스", "jumpsuit": "원피스", "overalls_(clothing)": "원피스", "kimono": "원피스",
    "swimsuit": "원피스", "wet_suit": "원피스",
    # 신발
    "shoe": "신발", "boot": "신발", "sandal_(type_of_shoe)": "신발", "flip-flop_(sandal)": "신발",
    "slipper_(footwear)": "신발", "arctic_(type_of_shoe)": "신발", "ski_boot": "신발",
    # 가방
    "backpack": "가방", "handbag": "가방", "clutch_bag": "가방", "duffel_bag": "가방", "satchel": "가방",
    "shoulder_bag": "가방", "tote_bag": "가방", "wallet": "가방",
    # 모자
    "hat": "모자", "baseball_cap": "모자", "cap_(headwear)": "모자", "beanie": "모자", "beret": "모자",
    "bonnet": "모자", "bowler_hat": "모자", "cowboy_hat": "모자", "dress_hat": "모자", "fedora": "모자",
    "sombrero": "모자", "sunhat": "모자", "skullcap": "모자", "turban": "모자", "headscarf": "모자",
    "headband": "모자",
    # 액세서리
    "belt": "액세서리", "bandanna": "액세서리", "bow-tie": "액세서리", "necktie": "액세서리",
    "neckerchief": "액세서리", "scarf": "액세서리", "shawl": "액세서리", "glove": "액세서리",
    "mitten": "액세서리", "sock": "액세서리", "tights_(clothing)": "액세서리", "sunglasses": "액세서리",
    "earring": "액세서리", "necklace": "액세서리", "bracelet": "액세서리", "watch": "액세서리",
    "ring": "액세서리", "wedding_ring": "액세서리", "tiara": "액세서리", "crown": "액세서리",
    "veil": "액세서리",
    # 기타 의류
    "apron": "기타", "underwear": "기타", "brassiere": "기타",
}

SPLITS = {"train": "lvis_v1_train.json", "val": "lvis_v1_val.json"}


def filter_lvis(data: dict, allowed: set[int]) -> dict:
    keep_cats = [c for c in data["categories"] if c["name"] in COARSE]
    cat_ids = {c["id"] for c in keep_cats}
    by_image: dict[int, list] = {}
    for a in data["annotations"]:
        if a["category_id"] in cat_ids:
            by_image.setdefault(a["image_id"], []).append(a)
    images = []
    for im in data["images"]:
        if im.get("license") not in allowed or im["id"] not in by_image:
            continue
        images.append({
            **im,
            "file_name": Path(im["coco_url"]).name,
            "neg_category_ids": [c for c in im.get("neg_category_ids", []) if c in cat_ids],
            "not_exhaustive_category_ids": [c for c in im.get("not_exhaustive_category_ids", []) if c in cat_ids],
        })
    img_ids = {im["id"] for im in images}
    anns = [a for ims in by_image.values() for a in ims if a["image_id"] in img_ids]
    return {
        "info": {**data.get("info", {}), "note": "FitCheck: 상업 이용 가능 라이선스 사진, 옷·잡화 클래스만"},
        "licenses": data["licenses"],
        "categories": [{**c, "coarse": COARSE[c["name"]]} for c in keep_cats],
        "images": images,
        "annotations": anns,
    }


def download_all(images: list[dict], dest: Path, workers: int = 4) -> Counter:
    import requests

    dest.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    stats: Counter = Counter()

    def one(im):
        path = dest / im["file_name"]
        if path.exists():
            return "exists"
        for attempt in range(5):
            try:
                r = session.get(im["coco_url"], timeout=60)
            except requests.RequestException:
                time.sleep(2 ** attempt * 2)
                continue
            if r.status_code == 200:
                path.write_bytes(r.content)
                return "ok"
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt * 2)
                continue
            return f"http_{r.status_code}"
        return "failed"

    with ThreadPoolExecutor(workers) as pool:
        for result in pool.map(one, images):
            stats[result] += 1
    return stats


def prepare(src: Path, out: Path, allowed: set[int], download: bool = True) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    report, rows = {}, []
    for split, ann_name in SPLITS.items():
        path = src / ann_name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        filtered = filter_lvis(data, allowed)
        (out / f"annotations_{split}.json").write_text(json.dumps(filtered, ensure_ascii=False))
        lic = {x["id"]: (x["name"], x["url"]) for x in data["licenses"]}
        coarse = {c["id"]: c["coarse"] for c in filtered["categories"]}
        for im in filtered["images"]:
            rows.append({"file": im["file_name"], "split": split, "coco_url": im["coco_url"],
                         "flickr_url": im.get("flickr_url", ""), "license": lic[im["license"]][0],
                         "license_url": lic[im["license"]][1]})
        report[split] = {
            "images": len(filtered["images"]),
            "annotations": len(filtered["annotations"]),
            "licenses": dict(Counter(lic[im["license"]][0] for im in filtered["images"])),
            "per_coarse": dict(Counter(coarse[a["category_id"]] for a in filtered["annotations"])),
        }
        if download:
            report[split]["download"] = dict(download_all(filtered["images"], out / "images"))
    with (out / "sources.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "split", "coco_url", "flickr_url", "license", "license_url"])
        writer.writeheader()
        writer.writerows(rows)
    (out / "prepare_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="LVIS 상업 이용 가능 옷 마스크 추출")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--no-download", action="store_true")
    args = p.parse_args(argv)
    print(json.dumps(prepare(Path(args.src), Path(args.out), ALLOWED, not args.no_download),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

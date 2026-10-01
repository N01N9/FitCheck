"""Open Images V7(분할 V5)에서 옷·잡화 인스턴스 마스크와 그 사진만 골라 받는다.

  - 주석·마스크: CC BY 4.0 (Google). 사진: "CC BY 2.0 으로 표기"(사진별 License·Author·원본 URL 을 남긴다).
  - 마스크는 모든 인스턴스를 칠한 것이 아니다(공식 문서). 학습 때 음성 손실을 조심해서 써야 한다.
  - 필요한 마스크 PNG 만 zip 에서 꺼내고, zip 은 다 쓴 뒤 지운다.
  - 사진은 공식 공개 버킷(open-images-dataset)에서 받는다.

출력 (out/)
  annotations_<split>.csv   원래 분할 CSV 에서 옷 클래스 행만 + coarse 열
  masks/<MaskPath>          해당 마스크 PNG
  images/<ImageID>.jpg      해당 사진
  sources.csv               사진별 원본 URL·저작자·라이선스

사용
  python -m phase0.prepare_openimages --work data/openimages --out data/openimages/clothing
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from phase0.collect_commons import USER_AGENT

BASE = "https://storage.googleapis.com/openimages"
IMAGES = "https://s3.amazonaws.com/open-images-dataset"

# 분할 클래스 중 옷·잡화 (MID → 우리 분류)
CLASSES = {
    "/m/01n4qj": ("Shirt", "상의"),
    "/m/01xyhv": ("Suit", "아우터"),
    "/m/0fly7": ("Jeans", "하의"), "/m/07mhn": ("Trousers", "하의"), "/m/01bfm9": ("Shorts", "하의"),
    "/m/02wv6h6": ("Skirt", "하의"), "/m/01cmb2": ("Miniskirt", "하의"),
    "/m/01d40f": ("Dress", "원피스"), "/m/01gkx_": ("Swimwear", "원피스"),
    "/m/01b638": ("Boot", "신발"), "/m/06k2mb": ("High heels", "신발"),
    "/m/02dl1y": ("Hat", "모자"), "/m/025rp__": ("Cowboy hat", "모자"), "/m/02fq_6": ("Fedora", "모자"),
    "/m/02jfl0": ("Sombrero", "모자"), "/m/02wbtzl": ("Sun hat", "모자"),
    "/m/080hkjn": ("Handbag", "가방"), "/m/01940j": ("Backpack", "가방"),
    "/m/0176mf": ("Belt", "액세서리"), "/m/01rkbr": ("Tie", "액세서리"), "/m/02h19r": ("Scarf", "액세서리"),
    "/m/0174n1": ("Glove", "액세서리"), "/m/01nq26": ("Sock", "액세서리"), "/m/0gjkl": ("Watch", "액세서리"),
    "/m/09j2d": ("Clothing", "기타"),
}

SPLITS = {
    "train": {
        "seg": "v5/train-annotations-object-segmentation.csv",
        "info": "2018_04/train/train-images-boxable-with-rotation.csv",
        "masks": [f"v5/train-masks/train-masks-{c}.zip" for c in "0123456789abcdef"],
    },
    "validation": {
        "seg": "v5/validation-annotations-object-segmentation.csv",
        "info": "2018_04/validation/validation-images-with-rotation.csv",
        "masks": [f"v5/validation-masks/validation-masks-{c}.zip" for c in "0123456789abcdef"],
    },
}


def session():
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def fetch(s, url: str, dest: Path) -> None:
    """큰 파일을 이어받기 없이 통째로 받는다(이미 있으면 건너뜀)."""
    if dest.exists():
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        try:
            with s.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            tmp.rename(dest)
            return
        except Exception:
            time.sleep(2 ** attempt * 5)
    raise RuntimeError(f"받기 실패: {url}")


def filter_rows(seg_csv: Path) -> list[dict]:
    with seg_csv.open(newline="") as fh:
        return [{**r, "coarse": CLASSES[r["LabelName"]][1], "label": CLASSES[r["LabelName"]][0]}
                for r in csv.DictReader(fh) if r["LabelName"] in CLASSES]


def image_info(info_csv: Path, ids: set[str]) -> dict[str, dict]:
    with info_csv.open(newline="") as fh:
        return {r["ImageID"]: r for r in csv.DictReader(fh) if r["ImageID"] in ids}


def extract_masks(zip_path: Path, names: set[str], dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            base = Path(info.filename).name
            if base in names and not (dest / base).exists():
                (dest / base).write_bytes(zf.read(info))
                n += 1
    return n


def download_images(s, split: str, ids: list[str], dest: Path, workers: int = 8) -> Counter:
    dest.mkdir(parents=True, exist_ok=True)
    folder = "train" if split == "train" else "validation"

    def one(image_id):
        path = dest / f"{image_id}.jpg"
        if path.exists():
            return "exists"
        for attempt in range(5):
            try:
                r = s.get(f"{IMAGES}/{folder}/{image_id}.jpg", timeout=60)
            except Exception:
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

    stats: Counter = Counter()
    with ThreadPoolExecutor(workers) as pool:
        for result in pool.map(one, ids):
            stats[result] += 1
    return stats


def prepare(work: Path, out: Path, splits: list[str], keep_zips: bool = False) -> dict:
    s = session()
    work.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    sources = []
    for split in splits:
        cfg = SPLITS[split]
        seg_csv = work / Path(cfg["seg"]).name
        info_csv = work / Path(cfg["info"]).name
        fetch(s, f"{BASE}/{cfg['seg']}", seg_csv)
        fetch(s, f"{BASE}/{cfg['info']}", info_csv)

        rows = filter_rows(seg_csv)
        with (out / f"annotations_{split}.csv").open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        ids = sorted({r["ImageID"] for r in rows})
        info = image_info(info_csv, set(ids))

        # 마스크: zip 은 MaskPath 첫 글자로 나뉘어 있다
        extracted = 0
        for zip_rel in cfg["masks"]:
            zip_path = work / Path(zip_rel).name
            shard = zip_path.stem[-1]
            names = {r["MaskPath"] for r in rows if r["MaskPath"].startswith(shard)}
            if not names:
                continue
            fetch(s, f"{BASE}/{zip_rel}", zip_path)
            extracted += extract_masks(zip_path, names, out / "masks")
            if not keep_zips:
                zip_path.unlink()

        dl = download_images(s, split, ids, out / "images")
        for image_id in ids:
            r = info.get(image_id, {})
            sources.append({"file": f"{image_id}.jpg", "split": split, "original_url": r.get("OriginalURL", ""),
                            "landing_url": r.get("OriginalLandingURL", ""), "author": r.get("Author", ""),
                            "license": r.get("License", "")})
        report[split] = {"images": len(ids), "masks": len(rows), "masks_extracted": extracted,
                         "download": dict(dl), "per_coarse": dict(Counter(r["coarse"] for r in rows)),
                         "licenses": dict(Counter(info.get(i, {}).get("License", "?") for i in ids))}
        (out / "prepare_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    with (out / "sources.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["file", "split", "original_url", "landing_url", "author", "license"])
        writer.writeheader()
        writer.writerows(sources)
    return report


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Open Images 옷 마스크 추출")
    p.add_argument("--work", required=True, help="원본 CSV·zip 을 둘 곳")
    p.add_argument("--out", required=True)
    p.add_argument("--split", action="append", choices=list(SPLITS), help="기본: validation, train")
    p.add_argument("--keep-zips", action="store_true")
    args = p.parse_args(argv)
    report = prepare(Path(args.work), Path(args.out), args.split or ["validation", "train"], args.keep_zips)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

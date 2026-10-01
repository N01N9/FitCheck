"""Openverse(CC 이미지 검색 API)에서 상업 이용·수정이 허용된 패션 사진을 모은다.

Openverse 는 Flickr 등에 올라온 CC 이미지의 색인이다. 이미지 자체는 원 출처(주로 Flickr)에서 받는다.
  - license_type=commercial,modification → CC BY / CC BY-SA / CC0 / PDM 만 나온다.
  - Commons 와 겹치지 않게 wikimedia 출처는 뺀다(collect_commons 가 따로 받는다).
  - 익명 이용은 검색어당 최대 240장(20장 × 12쪽)이라 검색어를 여러 개 돌린다.
  - 기록 형식은 collect_commons 의 sources.csv 와 같다.

사용
  python -m phase0.collect_openverse --out data/crawl/openverse
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Iterator

from phase0.collect_commons import CSV_FIELDS, USER_AGENT

API = "https://api.openverse.org/v1/images/"

QUERIES = [
    # 풀착장·스트리트·런웨이
    "street style", "street fashion", "street snap", "ootd", "outfit of the day", "outfit",
    "fashion week street style", "runway", "catwalk", "fashion show model", "lookbook",
    "full length portrait", "full body portrait", "standing woman full length",
    "standing man full length", "streetwear", "layered outfit", "coat outfit", "trench coat",
    "denim jacket outfit", "leather jacket outfit", "blazer outfit", "suit man standing",
    "dress woman standing", "winter outfit", "summer outfit", "casual outfit", "menswear",
    "womenswear", "korean fashion", "seoul street fashion", "seoul fashion week",
    "tokyo street fashion", "harajuku fashion", "japanese street fashion", "paris street style",
    "london street style", "new york street style", "milan street style", "fashion blogger",
    "hanbok", "school uniform", "office outfit", "sportswear outfit", "hoodie outfit",
    # 거울 셀카
    "mirror selfie", "outfit mirror selfie", "fitting room mirror", "mirror outfit",
    # 바닥·침대에 펼친 옷, 단품
    "flat lay clothes", "flatlay outfit", "clothes on bed", "clothes on floor", "folded clothes",
    "clothing flat lay", "outfit grid", "laid out clothes", "clothes pile", "t-shirt mockup",
    "clothes hanger", "wardrobe clothes", "thrift store clothes", "sneakers", "handbag",
    "hat fashion", "scarf fashion",
]


class Openverse:
    def __init__(self, session=None, delay: float = 1.0):
        if session is None:
            import requests

            session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
        self.s = session
        self.delay = delay

    def search(self, query: str, source: str | None = None, max_pages: int = 12) -> Iterator[dict]:
        import requests

        for page in range(1, max_pages + 1):
            params = {"q": query, "license_type": "commercial,modification", "page_size": 20,
                      "page": page, "mature": "false"}
            if source:  # 특정 출처(예: 박물관)만
                params["source"] = source
            else:  # Commons 는 collect_commons 가 따로 받는다
                params["excluded_source"] = "wikimedia"
            data = None
            for attempt in range(8):
                try:
                    r = self.s.get(API, params=params, timeout=60)
                except requests.ConnectionError:
                    time.sleep(2 ** attempt * 5)
                    continue
                if r.status_code == 429 or r.status_code >= 500:
                    # 익명 이용 한도. 기다렸다가 다시 묻는다
                    time.sleep(min(60 * 2 ** attempt, 1800))
                    continue
                if r.status_code in (400, 401):  # 마지막 쪽 너머
                    return
                r.raise_for_status()
                data = r.json()
                break
            time.sleep(self.delay)
            if not data or not data.get("results"):
                return
            yield from data["results"]
            if page >= data.get("page_count", 0):
                return

    def download(self, url: str) -> bytes | None:
        import requests

        for attempt in range(6):
            try:
                r = self.s.get(url, timeout=120)
            except requests.ConnectionError:
                time.sleep(2 ** attempt * 5)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt * 5)
                continue
            if r.status_code >= 400:
                return None  # 원 출처에서 지워진 사진
            time.sleep(self.delay / 2)
            return r.content
        return None


def license_name(item: dict) -> str:
    lic = (item.get("license") or "").lower()
    ver = item.get("license_version") or ""
    if lic in ("cc0", "pdm"):
        return {"cc0": "CC0", "pdm": "Public domain"}[lic]
    return f"CC {lic.upper()} {ver}".strip()


# 의상 소장품이 있고 상업 이용·수정이 허용되는 박물관 출처(Openverse 의 source 이름)
MUSEUM_SOURCES = [
    "clevelandmuseum", "smithsonian_cooper_hewitt_museum", "rijksmuseum", "brooklynmuseum",
    "digitaltmuseum", "smk", "europeana",
]
MUSEUM_QUERIES = [
    "dress", "gown", "costume", "coat", "jacket", "waistcoat", "shirt", "blouse", "skirt",
    "trousers", "suit", "uniform", "kimono", "hanbok", "shoes", "boots", "hat", "bonnet",
    "bag", "purse", "glove", "scarf", "shawl", "fashion", "garment", "clothing",
]


def collect(api: Openverse, out: Path, queries: list[str], max_files: int,
            min_short_side: int = 500, sources: list[str | None] | None = None) -> dict:
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    csv_path = out / "sources.csv"
    seen_ids, seen_sha1 = set(), set()
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                seen_ids.add(row["title"])
                seen_sha1.add(row["sha1"])
    new_file = not csv_path.exists()
    stats = {"seen": 0, "kept": len(seen_ids), "reject": {}}

    def reject(why):
        stats["reject"][why] = stats["reject"].get(why, 0) + 1

    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        for q, src in [(q, src) for src in (sources or [None]) for q in queries]:
            for item in api.search(q, src) if src else api.search(q):
                if stats["kept"] >= max_files:
                    break
                key = f"openverse:{item['id']}"
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                stats["seen"] += 1
                w, h = item.get("width") or 0, item.get("height") or 0
                if w and h and min(w, h) < min_short_side:
                    reject("small")
                    continue
                body = api.download(item["url"])
                if not body:
                    reject("download_failed")
                    continue
                sha1 = hashlib.sha1(body).hexdigest()
                if sha1 in seen_sha1:
                    reject("duplicate")
                    continue
                seen_sha1.add(sha1)
                name = f"{sha1[:16]}.jpg"
                (raw / name).write_bytes(body)
                writer.writerow({
                    "file": name,
                    "title": key,
                    "page_url": item.get("foreign_landing_url", ""),
                    "license": license_name(item),
                    "license_url": item.get("license_url", ""),
                    "artist": item.get("creator") or "",
                    "credit": f"{item.get('source', '')} via Openverse: {item.get('title') or ''}",
                    "restrictions": "",
                    "date": "",
                    "width": w,
                    "height": h,
                    "sha1": sha1,
                    "seed": f"openverse:{src + ':' if src else ''}{q}",
                })
                fh.flush()
                stats["kept"] += 1
            if stats["kept"] >= max_files:
                break

    (out / "collect_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Openverse 패션 사진 수집")
    p.add_argument("--out", required=True)
    p.add_argument("--max-files", type=int, default=20000)
    p.add_argument("--query", action="append", help="검색어 (없으면 QUERIES 전부)")
    p.add_argument("--museums", action="store_true", help="박물관 출처만, 의상 검색어로")
    args = p.parse_args(argv)
    if args.museums:
        stats = collect(Openverse(), Path(args.out), args.query or MUSEUM_QUERIES, args.max_files,
                        sources=MUSEUM_SOURCES)
    else:
        stats = collect(Openverse(), Path(args.out), args.query or QUERIES, args.max_files)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

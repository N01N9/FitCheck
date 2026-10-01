"""스미소니언 Open Access 에서 CC0 의상·잡화 사진을 모은다.

  - 검색어(의류 이름) + online_media_type:Images 로 찾고, 미디어마다 usage.access == "CC0" 인 것만 받는다.
  - 한 소장품의 여러 사진(다른 각도)을 최대 --views 장까지 받는다.
  - 긴 변 1600 으로 받는다(ids.si.edu 의 max 파라미터).
  - API 키(api.data.gov)는 저장소에 두지 않는다: 환경변수 DATAGOV_KEY 또는 ~/.config/fitcheck/datagov_key
  - 기록 형식은 collect_commons 의 sources.csv 와 같다.

사용
  python -m phase0.collect_smithsonian --out data/crawl/smithsonian
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterator

from phase0.collect_commons import CSV_FIELDS, USER_AGENT

API = "https://api.si.edu/openaccess/api/v1.0/search"
QUERIES = [
    "dress", "gown", "coat", "jacket", "shirt", "blouse", "skirt", "trousers", "pants", "jeans",
    "suit", "uniform", "vest", "waistcoat", "sweater", "kimono", "costume", "garment",
    "shoes", "boots", "sneakers", "sandals", "hat", "bonnet", "cap", "handbag", "purse",
    "glove", "scarf", "shawl", "belt", "necktie",
]


def load_key() -> str:
    key = os.environ.get("DATAGOV_KEY")
    if not key:
        path = Path.home() / ".config/fitcheck/datagov_key"
        key = path.read_text().strip() if path.exists() else ""
    if not key:
        raise SystemExit("api.data.gov 키가 없습니다(DATAGOV_KEY 또는 ~/.config/fitcheck/datagov_key)")
    return key


class Smithsonian:
    def __init__(self, key: str, session=None, delay: float = 0.3):
        if session is None:
            import requests

            session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
        self.s = session
        self.key = key
        self.delay = delay

    def _get(self, url: str, params=None, timeout: int = 60):
        import requests

        for attempt in range(7):
            try:
                r = self.s.get(url, params=params, timeout=timeout)
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(2 ** attempt * 3)
                continue
            if r.status_code == 429:  # api.data.gov 시간당 한도
                time.sleep(min(60 * 2 ** attempt, 1800))
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt * 3)
                continue
            time.sleep(self.delay)
            return r
        return None

    def search(self, query: str, rows: int = 100) -> Iterator[dict]:
        start = 0
        while True:
            r = self._get(API, {"api_key": self.key, "q": f"{query} AND online_media_type:Images",
                                "start": start, "rows": rows})
            if r is None or r.status_code != 200:
                return
            data = r.json().get("response", {})
            batch = data.get("rows") or []
            yield from batch
            start += rows
            if not batch or start >= data.get("rowCount", 0):
                return

    def download(self, url: str) -> bytes | None:
        r = self._get(url, timeout=120)
        return r.content if r is not None and r.status_code == 200 else None


def cc0_media(row: dict, views: int) -> list[str]:
    media = row.get("content", {}).get("descriptiveNonRepeating", {}).get("online_media", {}).get("media", [])
    urls = [m["content"] for m in media
            if m.get("type") == "Images" and (m.get("usage") or {}).get("access") == "CC0" and m.get("content")]
    return urls[:views]


def collect(api: Smithsonian, out: Path, queries: list[str], max_objects: int, views: int = 4,
            max_side: int = 1600) -> dict:
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    csv_path = out / "sources.csv"
    done, seen_sha1 = set(), set()
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                done.add(row["title"].rsplit(":", 1)[0])
                seen_sha1.add(row["sha1"])
    new_file = not csv_path.exists()
    stats = {"objects_seen": 0, "objects_kept": len(done), "images": 0, "reject": {}}

    def reject(why):
        stats["reject"][why] = stats["reject"].get(why, 0) + 1

    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        for q in queries:
            for row in api.search(q):
                if stats["objects_kept"] >= max_objects:
                    break
                oid = f"si:{row.get('id')}"
                if oid in done:
                    continue
                done.add(oid)
                stats["objects_seen"] += 1
                urls = cc0_media(row, views)
                if not urls:
                    reject("not_cc0")
                    continue
                dn = row.get("content", {}).get("descriptiveNonRepeating", {})
                kept = False
                for k, url in enumerate(urls):
                    body = api.download(f"{url}&max={max_side}")
                    if not body or body[:2] != b"\xff\xd8":  # JPEG 가 아니면 버린다
                        reject("download_failed")
                        continue
                    sha1 = hashlib.sha1(body).hexdigest()
                    if sha1 in seen_sha1:  # 여러 소장품이 같은 사진을 공유하기도 한다
                        reject("duplicate")
                        continue
                    seen_sha1.add(sha1)
                    name = f"{sha1[:16]}.jpg"
                    (raw / name).write_bytes(body)
                    writer.writerow({
                        "file": name,
                        "title": f"{oid}:{k}",
                        "page_url": dn.get("record_link", ""),
                        "license": "CC0",
                        "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                        "artist": "",
                        "credit": f"Smithsonian {dn.get('unit_code', '')}: {dn.get('data_source', '')}",
                        "restrictions": "",
                        "date": "",
                        "width": "",
                        "height": "",
                        "sha1": sha1,
                        "seed": f"smithsonian:{q}|{row.get('title', '')[:80]}",
                    })
                    stats["images"] += 1
                    kept = True
                fh.flush()
                if kept:
                    stats["objects_kept"] += 1
            if stats["objects_kept"] >= max_objects:
                break

    (out / "collect_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="스미소니언 Open Access CC0 의상 사진 수집")
    p.add_argument("--out", required=True)
    p.add_argument("--max-objects", type=int, default=100000)
    p.add_argument("--views", type=int, default=4)
    p.add_argument("--query", action="append")
    args = p.parse_args(argv)
    stats = collect(Smithsonian(load_key()), Path(args.out), args.query or QUERIES, args.max_objects, args.views)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

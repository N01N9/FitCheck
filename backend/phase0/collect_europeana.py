"""Europeana(유럽 문화유산 통합 API) 패션 테마에서 개방 라이선스 사진을 모은다.

  - theme=fashion, reusability=open → Public Domain / CC0 / CC BY / CC BY-SA 만 나온다.
    rights 값을 한 번 더 확인해 NC·ND 가 섞여 들어오지 않게 한다.
  - 이미지는 각 기관 서버(edmIsShownBy)에서 받는다. 받은 뒤 짧은 변이 작으면 버리고, 긴 변 1600 으로 줄여 저장한다.
  - API 키는 저장소에 두지 않는다: 환경변수 EUROPEANA_KEY 또는 ~/.config/fitcheck/europeana_key
  - 기록 형식은 collect_commons 의 sources.csv 와 같다.

사용
  python -m phase0.collect_europeana --out data/crawl/europeana
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Iterator

from phase0.collect_commons import CSV_FIELDS, USER_AGENT
from phase0.collect_met import shrink

API = "https://api.europeana.eu/record/v2/search.json"
OPEN_RIGHTS = re.compile(r"publicdomain/(zero|mark)|licenses/by(-sa)?/", re.I)
BLOCKED_RIGHTS = re.compile(r"-nc|-nd|rightsstatements", re.I)


def load_key() -> str:
    key = os.environ.get("EUROPEANA_KEY")
    if not key:
        path = Path.home() / ".config/fitcheck/europeana_key"
        key = path.read_text().strip() if path.exists() else ""
    if not key:
        raise SystemExit("Europeana API 키가 없습니다(EUROPEANA_KEY 또는 ~/.config/fitcheck/europeana_key)")
    return key


def rights_ok(url: str) -> bool:
    return bool(OPEN_RIGHTS.search(url or "")) and not BLOCKED_RIGHTS.search(url or "")


def rights_name(url: str) -> str:
    u = url.lower()
    if "publicdomain/zero" in u:
        return "CC0"
    if "publicdomain/mark" in u:
        return "Public domain"
    m = re.search(r"licenses/(by(?:-sa)?)/(\d\.\d)", u)
    return f"CC {m.group(1).upper()} {m.group(2)}" if m else url


class Europeana:
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

        for attempt in range(6):
            try:
                r = self.s.get(url, params=params, timeout=timeout)
            except (requests.ConnectionError, requests.Timeout):
                time.sleep(2 ** attempt * 3)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt * 3)
                continue
            time.sleep(self.delay)
            return r
        return None

    def search(self, query: str = "*", theme: str = "fashion") -> Iterator[dict]:
        cursor = "*"
        while cursor:
            params = {"wskey": self.key, "query": query, "theme": theme, "reusability": "open",
                      "media": "true", "qf": "TYPE:IMAGE", "rows": 100, "cursor": cursor,
                      "profile": "rich"}
            r = self._get(API, params)
            if r is None or r.status_code != 200:
                return
            data = r.json()
            yield from data.get("items", [])
            cursor = data.get("nextCursor")

    def download(self, url: str, deadline: float = 90.0, max_bytes: int = 60 << 20) -> bytes | None:
        """기관 서버가 아주 느리게 흘려보내면 요청 timeout 이 매번 새로 시작돼 한 파일에 계속 매달린다.
        그래서 파일 하나에 전체 시간 상한(deadline)과 크기 상한을 두고, 넘으면 건너뛴다."""
        import requests

        start = time.monotonic()
        try:
            with self.s.get(url, timeout=30, stream=True) as r:
                if r.status_code != 200:
                    return None
                buf = bytearray()
                for chunk in r.iter_content(1 << 16):
                    buf += chunk
                    if time.monotonic() - start > deadline or len(buf) > max_bytes:
                        return None
                time.sleep(self.delay)
                return bytes(buf)
        except requests.RequestException:
            return None


def image_size(body: bytes) -> tuple[int, int] | None:
    from PIL import Image

    try:
        return Image.open(io.BytesIO(body)).size
    except Exception:
        return None


def collect(api: Europeana, out: Path, max_files: int, min_short_side: int = 400,
            query: str = "*", theme: str = "fashion", max_side: int = 1600) -> dict:
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
        for item in api.search(query, theme):
            if stats["kept"] >= max_files:
                break
            key = f"europeana:{item.get('id')}"
            if key in seen_ids:
                continue
            seen_ids.add(key)
            stats["seen"] += 1
            if stats["seen"] % 200 == 0:  # 건너뛰는 구간이 길 수 있어 진행 상황을 로그로 남긴다
                provider = "; ".join(item.get("dataProvider") or [])
                print(json.dumps({"provider": provider, **stats}, ensure_ascii=False), flush=True)
            rights = (item.get("rights") or [""])[0]
            if not rights_ok(rights):
                reject("license")
                continue
            url = (item.get("edmIsShownBy") or [""])[0]
            if not url:
                reject("no_image")
                continue
            body = api.download(url)
            if not body:
                reject("download_failed")
                continue
            size = image_size(body)
            if size is None:
                reject("not_image")
                continue
            if min(size) < min_short_side:
                reject("small")
                continue
            sha1 = hashlib.sha1(body).hexdigest()
            if sha1 in seen_sha1:
                reject("duplicate")
                continue
            seen_sha1.add(sha1)
            name = f"{sha1[:16]}.jpg"
            # 기관 원본은 수 MB 라 디스크를 많이 먹는다. 긴 변 max_side 로 줄여 저장한다(sha1 은 원본 기준)
            (raw / name).write_bytes(shrink(body, max_side))
            writer.writerow({
                "file": name,
                "title": key,
                "page_url": item.get("guid", ""),
                "license": rights_name(rights),
                "license_url": rights,
                "artist": "; ".join(item.get("dcCreator") or []),
                "credit": "; ".join(item.get("dataProvider") or []),
                "restrictions": "",
                "date": "; ".join(item.get("year") or []),
                "width": size[0],
                "height": size[1],
                "sha1": sha1,
                "seed": f"europeana:{theme}:{(item.get('title') or [''])[0][:80]}",
            })
            fh.flush()
            stats["kept"] += 1

    (out / "collect_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Europeana 패션 사진 수집")
    p.add_argument("--out", required=True)
    p.add_argument("--max-files", type=int, default=100000)
    p.add_argument("--query", default="*")
    p.add_argument("--theme", default="fashion")
    args = p.parse_args(argv)
    stats = collect(Europeana(load_key()), Path(args.out), args.max_files, query=args.query, theme=args.theme)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

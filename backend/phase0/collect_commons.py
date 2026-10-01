"""Wikimedia Commons 에서 상업 이용 가능한 라이선스의 패션 사진을 모은다.

왜 Commons 인가
  Pexels 는 약관에서 ML 목적 자동 수집을 금지하고, Unsplash 는 ML 용도를 별도 유료
  라이선스로 안내한다. Commons 는 파일마다 라이선스 메타데이터가 있고 API 이용이 허용된다.

규칙
  - 라이선스: CC0 / Public domain / CC BY / CC BY-SA 만. NC·ND 가 붙으면 버린다.
  - 파일마다 저작자·라이선스·원본 페이지·초상권 경고(Restrictions)를 sources.csv 에 남긴다.
  - User-Agent 로 봇 정보를 밝히고, maxlag 와 요청 간격을 지킨다.

사용
  python -m phase0.collect_commons --out data/crawl/commons --max-files 3000
  (카테고리·검색어는 SEED_CATEGORIES / SEED_QUERIES 또는 --category / --query 로 준다)
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "FitCheckDataBot/0.1 (https://github.com/N01N9/FitCheck; research data collection)"

# 런웨이·스트리트 스냅: 정면 전신 + 겹쳐 입기가 많은 곳
SEED_CATEGORIES = [
    ("Category:Fashion shows in South Korea", 3),
    ("Category:Seoul Fashion Week", 3),
    ("Category:IDOL RUNWAY COLLECTION 2025", 1),
    ("Category:IDOL RUNWAY COLLECTION 2026", 1),
    ("Category:Shibuya Fashion Street Snap (by Dick Thomas Johnson)", 2),
    ("Category:Street fashion", 2),
    ("Category:Street Style (fashion)", 2),
    ("Category:Fashion photography of unidentified female models", 2),
    ("Category:Fashion shows", 2),
    ("Category:Harajuku fashion", 3),
    ("Category:Berlin Fashion Week", 3),
    ("Category:New York Fashion Week", 3),
    ("Category:Paris Fashion Week", 3),
    ("Category:London Fashion Week", 3),
    ("Category:Milan Fashion Week", 2),
]
SEED_QUERIES = [
    "street snap fashion",
    "street style fashion week",
    "fashion week runway look",
    "catwalk model runway",
    "street fashion outfit",
    "Seoul fashion week street style",
    "Tokyo street fashion",
    "lookbook fashion",
    "outfit of the day",
    "street style portrait full length",
    "mirror selfie",
    "flat lay clothes",
    "clothes on bed",
    "clothes on floor",
    "folded clothes",
    "clothes hanger shirt",
    "fashion model standing full length",
    "Korean street fashion",
]

ALLOWED_LICENSE = re.compile(r"^(cc0|cc[- ]zero|public domain|pd\b|pd-|cc[- ]by(-sa)?[- ]?\d)", re.I)
BLOCKED_LICENSE = re.compile(r"\b(nc|nd)\b|noncommercial|no ?derivatives", re.I)
TAG = re.compile(r"<[^>]+>")

CSV_FIELDS = [
    "file", "title", "page_url", "license", "license_url", "artist", "credit",
    "restrictions", "date", "width", "height", "sha1", "seed",
]


def clean(text: str | None) -> str:
    return html.unescape(TAG.sub("", text or "")).strip()


def license_ok(short_name: str) -> bool:
    name = short_name.strip()
    return bool(name) and bool(ALLOWED_LICENSE.search(name)) and not BLOCKED_LICENSE.search(name)


def year_of(date_text: str) -> int | None:
    m = re.search(r"(19|20)\d{2}", date_text or "")
    return int(m.group(0)) if m else None


@dataclass
class Filters:
    min_short_side: int = 500
    min_aspect: float = 0.0  # 세로/가로. 가로 전신 스냅도 많아 기본은 끈다(전신 여부는 다음 단계에서 거른다)
    min_year: int = 0  # 0 이면 연도로 거르지 않는다(선별은 다음 단계에서)


def passes(info: dict, meta: dict, f: Filters) -> tuple[bool, str]:
    if info.get("mime") not in ("image/jpeg", "image/png"):
        return False, "mime"
    w, h = info.get("width", 0), info.get("height", 0)
    if min(w, h) < f.min_short_side:
        return False, "small"
    if h / max(w, 1) < f.min_aspect:
        return False, "aspect"
    if not license_ok(clean(meta.get("LicenseShortName", {}).get("value"))):
        return False, "license"
    year = year_of(clean(meta.get("DateTimeOriginal", {}).get("value")))
    if year is not None and year < f.min_year:
        return False, "old"
    return True, "ok"


class Commons:
    def __init__(self, session=None, delay: float = 0.5):
        if session is None:
            import requests

            session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
        self.s = session
        self.delay = delay

    def get(self, **params) -> dict:
        params = {"format": "json", "formatversion": "2", "maxlag": "5", **params}
        for attempt in range(6):
            import requests

            # 제목 50개를 한 번에 물으면 GET 주소가 너무 길어져(414) POST 로 보낸다
            try:
                r = self.s.post(API, data=params, timeout=60)
            except requests.ConnectionError:
                time.sleep(2 ** attempt * 5)
                continue
            if r.status_code == 429 or r.status_code >= 500 or "maxlag" in r.text[:200]:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            time.sleep(self.delay)
            return r.json()
        raise RuntimeError("Commons API 가 계속 거절합니다(429/maxlag)")

    def _paged(self, **params) -> Iterator[dict]:
        cont: dict = {}
        while True:
            data = self.get(**params, **cont)
            yield data
            if "continue" not in data:
                return
            cont = data["continue"]

    def category_files(self, category: str, depth: int, seen: set[str]) -> Iterator[str]:
        if category in seen:
            return
        seen.add(category)
        subcats = []
        for data in self._paged(action="query", list="categorymembers", cmtitle=category,
                                cmtype="file|subcat", cmlimit="500"):
            for m in data["query"]["categorymembers"]:
                if m["ns"] == 6:
                    yield m["title"]
                elif m["ns"] == 14:
                    subcats.append(m["title"])
        if depth > 0:
            for sub in subcats:
                yield from self.category_files(sub, depth - 1, seen)

    def search_files(self, query: str, limit: int = 1000) -> Iterator[str]:
        n = 0
        for data in self._paged(action="query", list="search", srnamespace="6", srlimit="500",
                                srsearch=f"{query} filetype:bitmap"):
            for m in data["query"]["search"]:
                yield m["title"]
                n += 1
                if n >= limit:
                    return

    def file_infos(self, titles: list[str], thumb_width: int) -> list[dict]:
        data = self.get(action="query", prop="imageinfo", titles="|".join(titles),
                        iiprop="url|size|mime|sha1|extmetadata", iiurlwidth=str(thumb_width))
        return [p for p in data["query"]["pages"] if "imageinfo" in p]

    def download(self, url: str, dest: Path) -> bool:
        """받으면 True. 연결이 끊기거나 계속 거절되면 False(그 파일만 건너뛴다)."""
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
            r.raise_for_status()
            dest.write_bytes(r.content)
            time.sleep(self.delay)
            return True
        return False


def collect(api: Commons, out: Path, seeds: list[tuple[str, str]], max_files: int,
            filters: Filters, thumb_width: int = 1600, per_seed_limit: int = 1500) -> dict:
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    csv_path = out / "sources.csv"
    done = set()
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as fh:
            done = {row["title"] for row in csv.DictReader(fh)}
    new_file = not csv_path.exists()
    stats = {"seen": 0, "kept": len(done), "reject": {}}
    seen_titles: set[str] = set(done)
    seen_sha1: set[str] = set()
    seen_cats: set[str] = set()

    def titles() -> Iterator[tuple[str, str]]:
        # 한 곳(예: 특정 패션쇼 한 회)에 쏠리지 않게 씨앗마다 살펴볼 파일 수를 제한한다
        for kind, value in seeds:
            if kind == "category":
                name, _, depth = value.rpartition("@")
                found, seed = api.category_files(name, int(depth), seen_cats), name
            else:
                found, seed = api.search_files(value), f"search:{value}"
            for i, t in enumerate(found):
                if i >= per_seed_limit:
                    break
                yield t, seed

    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        batch: list[tuple[str, str]] = []

        def flush():
            infos = api.file_infos([t for t, _ in batch], thumb_width)
            seed_of = dict(batch)
            for page in infos:
                if stats["kept"] >= max_files:
                    break
                info = page["imageinfo"][0]
                meta = info.get("extmetadata", {})
                ok, why = passes(info, meta, filters)
                if ok and info["sha1"] in seen_sha1:
                    ok, why = False, "duplicate"
                if not ok:
                    stats["reject"][why] = stats["reject"].get(why, 0) + 1
                    continue
                name = f"{info['sha1'][:16]}.jpg"
                if not api.download(info.get("thumburl") or info["url"], raw / name):
                    stats["reject"]["download_failed"] = stats["reject"].get("download_failed", 0) + 1
                    continue
                seen_sha1.add(info["sha1"])
                writer.writerow({
                    "file": name,
                    "title": page["title"],
                    "page_url": info.get("descriptionurl", ""),
                    "license": clean(meta.get("LicenseShortName", {}).get("value")),
                    "license_url": clean(meta.get("LicenseUrl", {}).get("value")),
                    "artist": clean(meta.get("Artist", {}).get("value")),
                    "credit": clean(meta.get("Credit", {}).get("value")),
                    "restrictions": clean(meta.get("Restrictions", {}).get("value")),
                    "date": clean(meta.get("DateTimeOriginal", {}).get("value")),
                    "width": info.get("width"),
                    "height": info.get("height"),
                    "sha1": info["sha1"],
                    "seed": seed_of.get(page["title"], ""),
                })
                fh.flush()
                stats["kept"] += 1
            batch.clear()

        for title, seed in titles():
            if stats["kept"] >= max_files:
                break
            if title in seen_titles:
                continue
            seen_titles.add(title)
            stats["seen"] += 1
            batch.append((title, seed))
            if len(batch) == 50:
                flush()
        if batch and stats["kept"] < max_files:
            flush()

    (out / "collect_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Wikimedia Commons 패션 사진 수집")
    p.add_argument("--out", required=True)
    p.add_argument("--max-files", type=int, default=3000)
    p.add_argument("--category", action="append", help="'Category:이름@깊이' (여러 번 가능)")
    p.add_argument("--query", action="append", help="검색어 (여러 번 가능)")
    p.add_argument("--min-short-side", type=int, default=500)
    p.add_argument("--min-aspect", type=float, default=0.0)
    p.add_argument("--min-year", type=int, default=0)
    p.add_argument("--per-seed-limit", type=int, default=1500)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.category or args.query:
        seeds = [("category", c) for c in args.category or []] + [("query", q) for q in args.query or []]
    else:
        seeds = [("category", f"{c}@{d}") for c, d in SEED_CATEGORIES] + [("query", q) for q in SEED_QUERIES]
    filters = Filters(args.min_short_side, args.min_aspect, args.min_year)
    stats = collect(Commons(), Path(args.out), seeds, args.max_files, filters,
                    per_seed_limit=args.per_seed_limit)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

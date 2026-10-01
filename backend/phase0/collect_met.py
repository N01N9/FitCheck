"""메트로폴리탄 미술관(The Met) Open Access 의상 사진을 모은다.

The Met 은 퍼블릭 도메인 소장품 이미지를 CC0 로 공개하고 API 이용을 허용한다(초당 80회 이하).
의상 부서(Costume Institute, departmentId=8)에서 isPublicDomain 인 것만 받는다.
  - 흰·회색 배경 단품 사진이 많아 "흰 배경 단품" 유형을 채운다.
  - 같은 옷의 다른 각도 사진(additionalImages)도 받는다 → 앞·뒤·옆 다각도 정답으로도 쓸 수 있다.
  - 기록 형식은 collect_commons 의 sources.csv 와 같다. seed 에 objectName(예: Dress)을 남긴다.

사용
  python -m phase0.collect_met --out data/crawl/met
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import time
from pathlib import Path

from phase0.collect_commons import CSV_FIELDS, USER_AGENT

API = "https://collectionapi.metmuseum.org/public/collection/v1"
COSTUME_DEPT = 8


class Met:
    def __init__(self, session=None, delay: float = 0.15):
        if session is None:
            import requests

            session = requests.Session()
        session.headers["User-Agent"] = USER_AGENT
        self.s = session
        self.delay = delay

    def _get(self, url: str, **params):
        import requests

        for attempt in range(6):
            try:
                r = self.s.get(url, params=params or None, timeout=120)
            except requests.ConnectionError:
                time.sleep(2 ** attempt * 5)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt * 5)
                continue
            time.sleep(self.delay)
            return r
        return None

    def object_ids(self, department: int = COSTUME_DEPT) -> list[int]:
        r = self._get(f"{API}/search", departmentId=department, hasImages="true", q="*")
        return sorted(r.json().get("objectIDs") or []) if r is not None else []

    def obj(self, object_id: int) -> dict | None:
        r = self._get(f"{API}/objects/{object_id}")
        return r.json() if r is not None and r.status_code == 200 else None

    def download(self, url: str) -> bytes | None:
        r = self._get(url)
        return r.content if r is not None and r.status_code == 200 else None


def shrink(body: bytes, max_side: int) -> bytes:
    """긴 변을 max_side 로 줄인 JPEG. 이미 작으면 그대로."""
    from PIL import Image

    img = Image.open(io.BytesIO(body))
    if max(img.size) <= max_side and img.format == "JPEG":
        return body
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=92)
    return buf.getvalue()


def collect(api: Met, out: Path, max_objects: int, extra_views: int = 3, max_side: int = 1600) -> dict:
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    csv_path = out / "sources.csv"
    done_objects: set[str] = set()
    if csv_path.exists():
        with csv_path.open(encoding="utf-8") as fh:
            done_objects = {row["title"].split(":")[1] for row in csv.DictReader(fh)}
    new_file = not csv_path.exists()
    stats = {"objects_seen": 0, "objects_kept": len(done_objects), "images": 0, "reject": {}}

    def reject(why):
        stats["reject"][why] = stats["reject"].get(why, 0) + 1

    with csv_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        for oid in api.object_ids():
            if stats["objects_kept"] >= max_objects:
                break
            if str(oid) in done_objects:
                continue
            stats["objects_seen"] += 1
            o = api.obj(oid)
            if not o:
                reject("fetch_failed")
                continue
            if not o.get("isPublicDomain"):
                reject("not_public_domain")
                continue
            urls = [o.get("primaryImage")] + list(o.get("additionalImages") or [])[:extra_views]
            urls = [u for u in urls if u]
            if not urls:
                reject("no_image")
                continue
            kept_any = False
            for k, url in enumerate(urls):
                body = api.download(url)
                if not body:
                    reject("download_failed")
                    continue
                body = shrink(body, max_side)
                sha1 = hashlib.sha1(body).hexdigest()
                name = f"met{oid}_{k}.jpg"
                (raw / name).write_bytes(body)
                writer.writerow({
                    "file": name,
                    "title": f"met:{oid}:{k}",
                    "page_url": o.get("objectURL", ""),
                    "license": "CC0",
                    "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                    "artist": o.get("artistDisplayName", ""),
                    "credit": f"The Metropolitan Museum of Art, {o.get('creditLine', '')}",
                    "restrictions": "",
                    "date": o.get("objectDate", ""),
                    "width": "",
                    "height": "",
                    "sha1": sha1,
                    "seed": f"met:{o.get('objectName', '')}|{o.get('title', '')}",
                })
                stats["images"] += 1
                kept_any = True
            fh.flush()
            if kept_any:
                stats["objects_kept"] += 1

    (out / "collect_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="The Met Open Access 의상 사진 수집")
    p.add_argument("--out", required=True)
    p.add_argument("--max-objects", type=int, default=100000)
    p.add_argument("--extra-views", type=int, default=3)
    args = p.parse_args(argv)
    print(json.dumps(collect(Met(), Path(args.out), args.max_objects, args.extra_views),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

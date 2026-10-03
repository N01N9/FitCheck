"""데이터 엔진 수율 집계: 시도·승인 수, 떨어진 이유, 승인 1건당 GPU 시간, 구간별 승인율.

첫 2주 실험의 통과 기준(보고서): 승인율 20~30% 이상, 승인 1건당 GB10 약 60초 이내.

사용
  python -m unpaired.yields data/unpaired/engine/swap_pilot data/unpaired/engine/layer_pilot
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def summarize(attempts: list[dict]) -> dict:
    n = len(attempts)
    ok = [a for a in attempts if a["approved"]]
    seconds = [a["seconds"] for a in attempts]
    out = {"attempts": n, "approved": len(ok), "approval_rate": round(len(ok) / n, 3) if n else None,
           "median_seconds": round(statistics.median(seconds), 1) if seconds else None,
           "gpu_seconds_per_approved": round(sum(seconds) / len(ok), 1) if ok else None,
           "reasons": dict(Counter(r for a in attempts for r in a["reasons"]).most_common())}
    bins = Counter(a.get("share_bin") for a in attempts if a.get("share_bin"))
    if bins:
        out["approval_by_share_bin"] = {
            b: round(sum(a["approved"] for a in attempts if a.get("share_bin") == b) / c, 3) for b, c in bins.items()}
    cats = Counter(a["product_category"] for a in attempts)
    out["approval_by_product"] = {
        c: f"{sum(a['approved'] for a in attempts if a['product_category'] == c)}/{k}" for c, k in cats.most_common()}
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="데이터 엔진 수율 집계")
    p.add_argument("dirs", nargs="+")
    args = p.parse_args(argv)
    report = {}
    for d in args.dirs:
        path = Path(d) / "attempts.jsonl"
        if path.exists():
            report[Path(d).name] = summarize([json.loads(line) for line in path.read_text().splitlines() if line])
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

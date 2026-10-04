"""두 실행을 같은 사례끼리 짝지어 비교한다(정확 McNemar 검정). 보고서: "짝지은 통계 검정(McNemar)으로 한다".

각 실행은 score.py 의 scores.jsonl. 사례마다 첫 장(k=0)의 right_item / faithful 을 쓴다.

사용
  python -m unpaired.compare results/unpaired/eval_report/lora_v1_base:lora results/unpaired/eval_report/klein4b:outline_dimcrop
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path

from unpaired.score import faithful, right_item


def first_samples(run: Path, variant: str) -> dict[str, dict]:
    out = {}
    for line in (run / "scores.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["variant"] == variant and r["k"] == 0:
            out[r["file"]] = r
    return out


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """양측 정확 검정 p 값(불일치 쌍만 쓴다)."""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def compare(a: dict[str, dict], b: dict[str, dict], metric) -> dict:
    files = sorted(set(a) & set(b))
    a_only = sum(1 for f in files if metric(a[f]) and not metric(b[f]))
    b_only = sum(1 for f in files if metric(b[f]) and not metric(a[f]))
    return {"n": len(files), "a_rate": round(sum(metric(a[f]) for f in files) / max(len(files), 1), 3),
            "b_rate": round(sum(metric(b[f]) for f in files) / max(len(files), 1), 3),
            "a_only": a_only, "b_only": b_only, "p": mcnemar_exact(a_only, b_only)}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="두 실행의 짝지은 비교(McNemar)")
    p.add_argument("a", help="실행폴더:변형")
    p.add_argument("b", help="실행폴더:변형")
    args = p.parse_args(argv)
    (ra, va), (rb, vb) = (x.rsplit(":", 1) for x in (args.a, args.b))
    a, b = first_samples(Path(ra), va), first_samples(Path(rb), vb)
    print(json.dumps({"right_item": compare(a, b, right_item), "faithful": compare(a, b, faithful)}, indent=2))


if __name__ == "__main__":
    main()

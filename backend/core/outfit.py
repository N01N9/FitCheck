"""옷장 코디 추천: 규칙으로 후보를 만들고 점수를 매겨 상위 코디를 고른다 (기획서 3장 F4, 9.3).

LLM이 최종 선별·설명을 붙이는 단계는 이 결과(점수와 이유)를 입력으로 쓴다.
체형 적합도는 체형 분석(F1) 규칙이 들어올 자리로, 지금은 함수를 주입받는 인터페이스만 둔다.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Callable

from core.color import REFERENCE_COLORS, srgb_to_lab

NEUTRALS = {"블랙", "화이트", "아이보리", "그레이", "차콜", "네이비", "베이지", "브라운", "카키"}
VOLUME = {"슬림": 0, "레귤러": 1, "세미오버": 2, "오버": 3}
DEFAULT_WEIGHTS = {"color": 0.25, "silhouette": 0.15, "style": 0.15, "formality": 0.15, "freshness": 0.15, "body": 0.15}


@dataclass
class Item:
    id: str
    category: str  # 상의/하의/아우터/원피스/신발/...
    subcategory: str = ""
    primary_color: str = "기타"
    secondary_colors: list[str] = field(default_factory=list)
    pattern: str = "무지"
    fit: str = "레귤러"
    season: list[str] = field(default_factory=lambda: ["봄", "여름", "가을", "겨울"])
    formality: int = 2
    style_tags: list[str] = field(default_factory=list)
    available: bool = True  # 세탁 필요·세탁 중이면 False (laundry.GarmentCare.available)
    days_since_worn: int | None = None


@dataclass
class Context:
    temperature: float  # 체감 기온 ℃
    formality: int = 2  # 원하는 격식(1~5): 운동 1, 캐주얼 2, 출근 3~4, 하객·면접 4~5
    rain: bool = False
    style: str | None = None  # 원하는 분위기(스타일 태그)


@dataclass
class Outfit:
    items: list[Item]
    score: float
    parts: dict[str, float]
    reasons: list[str]

    @property
    def ids(self) -> list[str]:
        return [i.id for i in self.items]


def seasons_for(temp: float) -> set[str]:
    if temp >= 23:
        return {"여름"}
    if temp >= 17:
        return {"봄", "가을", "여름"}
    if temp >= 9:
        return {"봄", "가을"}
    return {"겨울", "가을"}


def needs_outer(temp: float) -> bool:
    return temp < 17


def _hue(name: str) -> float | None:
    rgb = REFERENCE_COLORS.get(name)
    if rgb is None:
        return None
    lab = srgb_to_lab(rgb)
    return math.degrees(math.atan2(lab[2], lab[1])) % 360


def color_score(items: list[Item]) -> tuple[float, str]:
    accents = sorted({i.primary_color for i in items if i.primary_color not in NEUTRALS})
    patterned = sum(1 for i in items if i.pattern != "무지")
    if len(accents) == 0:
        score, why = 0.85, "무채색·기본색 위주의 안정적인 조합"
    elif len(accents) == 1:
        score, why = 1.0, f"기본색에 {accents[0]} 포인트 한 가지"
    elif len(accents) == 2:
        h1, h2 = _hue(accents[0]), _hue(accents[1])
        close = h1 is not None and h2 is not None and min(abs(h1 - h2), 360 - abs(h1 - h2)) < 40
        score, why = (0.8, "비슷한 계열 색끼리 맞춘 톤온톤") if close else (0.55, "포인트 색이 두 가지라 다소 강함")
    else:
        score, why = 0.3, "색이 많아 산만할 수 있음"
    if patterned > 1:
        score -= 0.2 * (patterned - 1)
        why += ", 무늬 있는 옷이 여러 개"
    return max(score, 0.0), why


def silhouette_score(items: list[Item]) -> tuple[float, str]:
    top = next((i for i in items if i.category in ("상의", "원피스")), None)
    bottom = next((i for i in items if i.category == "하의"), None)
    if not top or not bottom or top.fit not in VOLUME or bottom.fit not in VOLUME:
        return 0.8, "실루엣 정보 부족"
    t, b = VOLUME[top.fit], VOLUME[bottom.fit]
    if t >= 2 and b >= 2:
        return 0.6, "상하의 모두 넉넉해 부해 보일 수 있음"
    if t == 0 and b == 0:
        return 0.85, "상하의 모두 슬림한 날씬한 실루엣"
    if abs(t - b) >= 2:
        return 1.0, "넉넉함과 슬림함이 대비되는 균형 잡힌 실루엣"
    return 0.9, "무난한 실루엣"


def style_score(items: list[Item], wanted: str | None) -> float:
    tagged = [set(i.style_tags) for i in items if i.style_tags]
    if len(tagged) < 2:
        base = 0.7
    else:
        pairs = list(itertools.combinations(tagged, 2))
        base = sum(len(a & b) / len(a | b) for a, b in pairs) / len(pairs)
        base = 0.4 + 0.6 * base
    if wanted:
        base = 0.7 * base + 0.3 * (sum(wanted in i.style_tags for i in items) / len(items))
    return base


def formality_score(items: list[Item], target: int) -> float:
    mean = sum(i.formality for i in items) / len(items)
    return max(0.0, 1 - abs(mean - target) / 4)


def freshness_score(items: list[Item]) -> float:
    def one(i: Item) -> float:
        d = i.days_since_worn
        if d is None or d >= 4:
            return 1.0
        return 0.2 if d < 2 else 0.6

    return sum(one(i) for i in items) / len(items)


def candidates(wardrobe: list[Item], ctx: Context) -> list[list[Item]]:
    """날씨·격식·세탁 상태 조건을 통과한 슬롯 조합들."""
    ok_seasons = seasons_for(ctx.temperature)

    def usable(i: Item) -> bool:
        return i.available and bool(ok_seasons & set(i.season)) and abs(i.formality - ctx.formality) <= 2

    pool = {c: [i for i in wardrobe if i.category == c and usable(i)] for c in ("상의", "하의", "원피스", "아우터", "신발")}
    bases = [[t, b] for t in pool["상의"] for b in pool["하의"]] + [[d] for d in pool["원피스"]]
    outers: list[Item | None] = pool["아우터"] if needs_outer(ctx.temperature) else [None]
    if needs_outer(ctx.temperature) and not outers:
        outers = [None]  # 아우터가 없어도 추천은 한다(이유에 표시)
    shoes: list[Item | None] = pool["신발"] or [None]
    return [[x for x in base + [o, s] if x is not None] for base in bases for o in outers for s in shoes]


def score_outfit(items: list[Item], ctx: Context, weights: dict[str, float] = DEFAULT_WEIGHTS,
                 body_fit: Callable[[list[Item]], float] | None = None) -> Outfit:
    c, c_why = color_score(items)
    s, s_why = silhouette_score(items)
    parts = {
        "color": c,
        "silhouette": s,
        "style": style_score(items, ctx.style),
        "formality": formality_score(items, ctx.formality),
        "freshness": freshness_score(items),
        "body": body_fit(items) if body_fit else 1.0,
    }
    total = sum(weights[k] * v for k, v in parts.items()) / sum(weights.values())
    reasons = [c_why, s_why]
    if needs_outer(ctx.temperature) and not any(i.category == "아우터" for i in items):
        reasons.append(f"{ctx.temperature:.0f}℃라 아우터가 필요하지만 입을 수 있는 아우터가 없음")
    if parts["freshness"] < 1:
        reasons.append("최근에 입은 옷이 포함됨")
    return Outfit(items, round(total, 4), {k: round(v, 3) for k, v in parts.items()}, reasons)


def recommend(wardrobe: list[Item], ctx: Context, top_n: int = 3, max_reuse: int = 1,
              body_fit: Callable[[list[Item]], float] | None = None) -> list[Outfit]:
    """점수 순으로 고르되, 같은 옷이 결과에 max_reuse번 넘게 나오지 않게 해 다양성을 지킨다."""
    scored = sorted((score_outfit(c, ctx, body_fit=body_fit) for c in candidates(wardrobe, ctx)),
                    key=lambda o: (-o.score, o.ids))
    picked, used = [], {}
    for outfit in scored:
        core_ids = [i.id for i in outfit.items if i.category != "신발"]
        if any(used.get(i, 0) >= max_reuse for i in core_ids):
            continue
        picked.append(outfit)
        for i in core_ids:
            used[i] = used.get(i, 0) + 1
        if len(picked) == top_n:
            break
    return picked

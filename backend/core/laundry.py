"""착용·세탁 상태 관리 (기획서 3장 F5, 9.4).

옷마다 "마지막 세탁 후 착용 부담"을 쌓고, 세탁 주기에 닿으면 "세탁 필요"로 바꾼다.
덥고 습한 날·운동한 날은 한 번 입어도 더 많이 쌓인다. "아직 괜찮아요"를 누르면 그 옷의 주기를 늘린다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

CLEAN, WORN, NEEDS_WASH, WASHING = "깨끗함", "착용중", "세탁 필요", "세탁 중"

# 세부 종류별 기본 세탁 주기(착용 횟수). 없으면 카테고리 기본값.
WASH_INTERVAL = {
    "티셔츠": 1, "민소매": 1, "폴로": 1, "트레이닝": 1, "레깅스": 1, "양말": 1,
    "셔츠": 2, "블라우스": 2, "반바지": 2, "원피스": 2,
    "맨투맨": 3, "후드": 3,
    "니트": 4, "슬랙스": 4, "면바지": 4, "치마": 4,
    "가디건": 5, "조끼": 5,
    "청바지": 8,
    "재킷": 10, "블레이저": 10, "점퍼": 10, "바람막이": 10,
    "코트": 30, "패딩": 30,  # 시즌당 1~2회
}
CATEGORY_INTERVAL = {"상의": 2, "하의": 4, "아우터": 10, "원피스": 2, "신발": 20, "가방": 60, "액세서리": 10}
MAX_LEARNED_FACTOR = 2.0  # 사용자가 미뤄도 기본 주기의 2배까지만 늘린다

DRY_ONLY = "드라이클리닝"
LIGHT, DARK, WHITE = "밝은색", "어두운색", "흰색"
_WHITE_COLORS = {"화이트", "아이보리"}
_DARK_COLORS = {"블랙", "차콜", "네이비", "버건디", "브라운", "카키", "그린", "레드", "퍼플"}


def color_group(primary_color: str | None) -> str:
    if primary_color in _WHITE_COLORS:
        return WHITE
    if primary_color in _DARK_COLORS:
        return DARK
    return LIGHT


def default_care(material: str | None, subcategory: str | None) -> tuple[str, int]:
    """케어 라벨이 없을 때 소재·종류로 추정한 (세탁 방식, 최고 물 온도). 앱에서는 '추정'으로 표시한다."""
    m, s = (material or ""), (subcategory or "")
    if any(w in m for w in ("가죽", "스웨이드", "모피")) or s in ("코트", "블레이저"):
        return DRY_ONLY, 0
    if any(w in m for w in ("실크", "레이온")):
        return "손세탁", 30
    if any(w in m for w in ("울", "캐시미어", "니트", "앙고라")) or s in ("니트", "가디건"):
        return "울코스", 30
    return "일반세탁", 40


@dataclass
class GarmentCare:
    id: str
    category: str
    subcategory: str | None = None
    primary_color: str | None = None
    wash_method: str = "일반세탁"
    max_temp: int = 40
    care_estimated: bool = True  # 케어 라벨로 확인했으면 False
    state: str = CLEAN
    load: float = 0.0  # 마지막 세탁 후 쌓인 착용 부담
    interval_override: float | None = None  # 사용자 행동으로 학습한 주기
    wear_count: int = 0  # 전체 착용 횟수(1회 착용당 비용 계산용)
    last_worn: date | None = None
    last_washed: date | None = None

    @property
    def base_interval(self) -> float:
        return WASH_INTERVAL.get(self.subcategory or "", CATEGORY_INTERVAL.get(self.category, 3))

    @property
    def interval(self) -> float:
        return self.interval_override or self.base_interval

    @property
    def available(self) -> bool:
        """오늘 코디 추천에 넣어도 되는지."""
        return self.state in (CLEAN, WORN)


def record_wear(item: GarmentCare, day: date, hot_humid: bool = False, sweaty: bool = False):
    """한 번 입은 것을 기록한다. 운동처럼 땀을 많이 흘렸으면 바로 세탁 필요."""
    item.wear_count += 1
    item.last_worn = day
    item.load += item.interval if sweaty else (1.5 if hot_humid else 1.0)
    item.state = NEEDS_WASH if item.load >= item.interval - 1e-9 else WORN


def snooze(item: GarmentCare):
    """'아직 괜찮아요': 이번에는 세탁을 미루고, 이 옷의 주기를 1회 늘려 학습한다."""
    if item.state != NEEDS_WASH:
        return
    item.interval_override = min(item.interval + 1, item.base_interval * MAX_LEARNED_FACTOR)
    # 주기를 최대치까지 늘렸는데도 부담이 넘치면 세탁 필요 상태를 유지한다
    item.state = WORN if item.load < item.interval else NEEDS_WASH


def start_wash(items: list[GarmentCare]):
    for item in items:
        item.state = WASHING


def finish_wash(items: list[GarmentCare], day: date):
    for item in items:
        item.state, item.load, item.last_washed = CLEAN, 0.0, day


@dataclass
class Bundle:
    method: str
    color: str | None
    max_temp: int
    items: list[GarmentCare] = field(default_factory=list)

    @property
    def title(self) -> str:
        if self.method == DRY_ONLY:
            return f"드라이클리닝 맡길 옷 {len(self.items)}벌"
        return f"{self.color} 옷 · {self.method} {self.max_temp}℃ 이하 · {len(self.items)}벌"


def laundry_bundles(items: list[GarmentCare]) -> list[Bundle]:
    """세탁 필요한 옷을 한 번에 돌릴 수 있는 묶음으로 나눈다.
    같은 세탁 방식·같은 색 계열끼리 묶고, 물 온도는 묶음에서 가장 낮은 것에 맞춘다."""
    groups: dict[tuple, Bundle] = {}
    for item in items:
        if item.state != NEEDS_WASH:
            continue
        if item.wash_method == DRY_ONLY:
            key = (DRY_ONLY, None)
        else:
            key = (item.wash_method, color_group(item.primary_color))
        bundle = groups.setdefault(key, Bundle(key[0], key[1], item.max_temp))
        bundle.items.append(item)
        bundle.max_temp = min(bundle.max_temp, item.max_temp)
    return sorted(groups.values(), key=lambda b: (b.method == DRY_ONLY, -len(b.items)))


def cost_per_wear(price: float, item: GarmentCare) -> float | None:
    return round(price / item.wear_count) if item.wear_count else None


def unworn_for(items: list[GarmentCare], today: date, days: int = 180) -> list[GarmentCare]:
    """오래 안 입은 옷(중고 판매 제안 후보). 한 번도 안 입은 옷도 포함한다."""
    out = []
    for item in items:
        if item.last_worn is None or (today - item.last_worn).days >= days:
            out.append(item)
    return out


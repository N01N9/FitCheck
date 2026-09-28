"""숫자 핏 리포트: 옷 실측과 신체 치수를 비교해 "내가 입으면 어떤 핏인지"를 계산한다.

기획서 3장 F3, 9.2. 구간값(THRESHOLDS 등)은 초기값이며 AI Hub 의류 통합 데이터와
실제 착용 피드백으로 보정한다. 길이 단위는 모두 cm.

옷 실측 표기는 한국 쇼핑몰 방식을 따른다.
  상의: length(총장, 목 옆점부터), shoulder(어깨너비), chest(가슴단면), sleeve(소매길이, 어깨 끝부터)
  하의: waist(허리단면), hip(엉덩이단면), thigh(허벅지단면), rise(밑위), inseam(안쪽 기장), length(총장)
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 신체 랜드마크 높이(바닥부터)를 키에 대한 비율로 둔 추정값. 3D 체형 측정 결과가 있으면 덮어쓴다.
LANDMARK_RATIOS = {
    "hps": 0.815,  # 목 옆점(상의 총장의 시작점)
    "waist": 0.61,
    "hip": 0.50,  # 엉덩이 가장 넓은 곳
    "crotch": 0.46,
    "knee": 0.28,
    "ankle": 0.05,
}
# 키로 추정하는 치수(측정값이 없을 때만 쓰고 estimated로 표시)
LENGTH_RATIOS = {"arm_length": 0.33, "inseam": 0.45}

FIT_LABELS = ["타이트", "슬림", "레귤러", "세미오버", "오버"]
# 둘레 여유량(옷 둘레 − 몸 둘레) 구간 경계. 경계값 4개 → 라벨 5개
THRESHOLDS = {
    ("상의", "chest"): [5, 10, 16, 24],
    ("원피스", "chest"): [5, 10, 16, 24],
    ("아우터", "chest"): [10, 16, 22, 30],  # 안에 옷을 입으므로 여유가 더 필요
    ("하의", "hip"): [2, 6, 12, 20],
    ("하의", "thigh"): [3, 8, 14, 22],
}
BOTTOM_THIGH_LABELS = ["스키니", "슬림", "레귤러", "와이드", "오버와이드"]
# 신축성이 있으면 같은 여유량이라도 덜 끼므로 구간을 낮춘다(cm)
STRETCH_SHIFT = {"없음": 0, "약간": 2, "많음": 5}


@dataclass
class Body:
    height: float
    shoulder_width: float | None = None
    chest: float | None = None
    waist: float | None = None
    hip: float | None = None
    thigh: float | None = None
    arm_length: float | None = None  # 어깨 끝~손목
    inseam: float | None = None  # 가랑이~발목
    landmarks: dict[str, float] = field(default_factory=dict)  # 높이 직접 지정(3D 체형 결과)

    def landmark(self, name: str) -> float:
        return self.landmarks.get(name, LANDMARK_RATIOS[name] * self.height)


@dataclass
class Zone:
    zone: str
    label: str
    detail: str
    ease_cm: float | None = None


@dataclass
class FitReport:
    category: str
    zones: list[Zone]
    warnings: list[str]
    estimated: list[str]  # 키로 추정한 신체 치수

    def summary(self) -> str:
        return ", ".join(f"{z.zone} {z.label}" for z in self.zones)

    def label_of(self, zone: str) -> str | None:
        return next((z.label for z in self.zones if z.zone == zone), None)


def _band(value: float, bounds: list[float], labels: list[str]) -> str:
    for bound, label in zip(bounds, labels):
        if value < bound:
            return label
    return labels[-1]


def _get(body: Body, name: str, estimated: list[str]) -> float | None:
    value = getattr(body, name)
    if value is None and name in LENGTH_RATIOS:
        estimated.append(name)
        return round(LENGTH_RATIOS[name] * body.height, 1)
    return value


def shoulder_zone(garment_shoulder: float, body_shoulder: float) -> Zone:
    diff = garment_shoulder - body_shoulder
    if diff < -2:
        return Zone("어깨", "작음", f"옷 어깨가 {-diff:.0f}cm 좁아 끼거나 당길 수 있음", diff)
    if diff <= 2:
        return Zone("어깨", "정핏", "어깨선이 어깨 끝에 맞음", diff)
    drop = diff / 2
    return Zone("어깨", "드롭숄더", f"어깨선이 양쪽으로 약 {drop:.0f}cm 내려옴", diff)


def hem_zone(length: float, body: Body) -> Zone:
    """상의 총장(목 옆점부터)으로 밑단이 몸의 어디에 오는지."""
    hem = body.landmark("hps") - length
    waist, hip, crotch, knee = (body.landmark(n) for n in ("waist", "hip", "crotch", "knee"))
    if hem >= waist + 4:
        label = "크롭"
    elif hem >= waist - 4:
        label = "허리선"
    elif hem >= hip + 3:
        label = "골반"
    elif hem >= hip - 3:
        label = "엉덩이 중간"
    elif hem >= crotch - 12:
        label = "엉덩이를 덮음"
    elif hem >= knee + 5:
        label = "허벅지"
    elif hem >= knee - 5:
        label = "무릎"
    else:
        label = "무릎 아래"
    return Zone("기장", label, f"밑단이 바닥에서 약 {hem:.0f}cm 높이에 옴")


def sleeve_zone(sleeve: float, drop: float, arm_length: float) -> Zone:
    """소매 끝이 팔의 어디에 오는지. 드롭숄더면 소매가 그만큼 아래에서 시작한다."""
    ratio = (sleeve + max(drop, 0)) / arm_length
    label = _band(
        ratio,
        [0.25, 0.5, 0.65, 0.9, 1.02, 1.12],
        ["짧은 소매", "팔꿈치 위", "팔꿈치", "7부", "손목", "손등을 덮음", "손가락까지 덮음"],
    )
    return Zone("소매", label, f"소매 끝이 팔 길이의 약 {ratio * 100:.0f}% 지점")


def top_report(garment: dict, body: Body, category: str = "상의", stretch: str = "없음") -> FitReport:
    zones, warnings, estimated = [], [], []
    drop = 0.0
    if garment.get("shoulder") and body.shoulder_width:
        z = shoulder_zone(garment["shoulder"], body.shoulder_width)
        drop = max(z.ease_cm / 2, 0)
        zones.append(z)
    if garment.get("chest") and body.chest:
        ease = garment["chest"] * 2 - body.chest
        bounds = [b - STRETCH_SHIFT[stretch] for b in THRESHOLDS[(category, "chest")]]
        label = _band(ease, bounds, FIT_LABELS)
        zones.append(Zone("가슴", label, f"가슴 둘레 여유 {ease:+.0f}cm", round(ease, 1)))
        if ease < 0 and stretch == "없음":
            warnings.append("가슴 둘레가 몸보다 작아 입기 어려울 수 있음")
    if garment.get("length"):
        zones.append(hem_zone(garment["length"], body))
    if garment.get("sleeve"):
        arm = _get(body, "arm_length", estimated)
        zones.append(sleeve_zone(garment["sleeve"], drop, arm))
    return FitReport(category, zones, warnings, estimated)


def bottom_report(garment: dict, body: Body, stretch: str = "없음") -> FitReport:
    zones, warnings, estimated = [], [], []
    shift = STRETCH_SHIFT[stretch]
    if garment.get("waist") and body.waist:
        ease = garment["waist"] * 2 - body.waist + shift
        label = _band(ease, [0, 3, 6], ["작음", "딱 맞음", "여유", "큼(벨트 필요)"])
        zones.append(Zone("허리", label, f"허리 둘레 여유 {ease - shift:+.0f}cm", round(ease - shift, 1)))
        if label == "작음":
            warnings.append("허리가 작아 잠기지 않을 수 있음")
    if garment.get("hip") and body.hip:
        ease = garment["hip"] * 2 - body.hip
        bounds = [b - shift for b in THRESHOLDS[("하의", "hip")]]
        zones.append(Zone("엉덩이", _band(ease, bounds, FIT_LABELS), f"엉덩이 둘레 여유 {ease:+.0f}cm", round(ease, 1)))
    if garment.get("thigh") and body.thigh:
        ease = garment["thigh"] * 2 - body.thigh
        bounds = [b - shift for b in THRESHOLDS[("하의", "thigh")]]
        zones.append(Zone("허벅지", _band(ease, bounds, BOTTOM_THIGH_LABELS), f"허벅지 둘레 여유 {ease:+.0f}cm", round(ease, 1)))
    inseam = garment.get("inseam")
    if inseam is None and garment.get("length") and garment.get("rise"):
        inseam = garment["length"] - garment["rise"]
    if inseam:
        body_inseam = _get(body, "inseam", estimated)
        diff = inseam - body_inseam
        label = _band(diff, [-8, -3, 2, 6], ["크롭(발목 위)", "발목", "기본(신발에 살짝 닿음)", "김(롤업 필요)", "바닥에 끌림"])
        zones.append(Zone("기장", label, f"안쪽 기장이 다리보다 {diff:+.0f}cm", round(diff, 1)))
    return FitReport("하의", zones, warnings, estimated)


def fit_report(category: str, garment: dict, body: Body, stretch: str = "없음") -> FitReport:
    if category == "하의":
        return bottom_report(garment, body, stretch)
    if category in ("상의", "아우터", "원피스"):
        return top_report(garment, body, category, stretch)
    raise ValueError(f"핏 리포트를 지원하지 않는 카테고리: {category}")


def recommend_size(category: str, size_chart: dict[str, dict], body: Body, preferred: str = "레귤러",
                   stretch: str = "없음") -> tuple[str | None, dict[str, FitReport]]:
    """사이즈표({"S": {...}, "M": {...}})에서 선호 핏에 가장 가까운 사이즈를 고른다.
    경고(입기 어려움)가 있는 사이즈는 고르지 않는다."""
    main_zone = "엉덩이" if category == "하의" else "가슴"
    target = FIT_LABELS.index(preferred)
    reports = {size: fit_report(category, m, body, stretch) for size, m in size_chart.items()}
    best, best_gap = None, None
    for size, report in reports.items():
        label = report.label_of(main_zone)
        if report.warnings or label not in FIT_LABELS:
            continue
        gap = abs(FIT_LABELS.index(label) - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = size, gap
    return best, reports

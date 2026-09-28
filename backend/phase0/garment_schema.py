"""옷 태깅 결과의 형식(JSON 스키마)과 검증.

폰(엣지)·Spark·클라우드 어느 모델이 태깅하든 이 형식으로 맞춘다(기획서 7.3).
값은 사용자가 정답 라벨을 한국어로 바로 적을 수 있게 한국어로 둔다.
"""

from __future__ import annotations

CATEGORIES = ["상의", "하의", "아우터", "원피스", "신발", "가방", "액세서리"]
SUBCATEGORY_HINTS = {
    "상의": ["티셔츠", "셔츠", "블라우스", "니트", "맨투맨", "후드", "민소매", "폴로"],
    "하의": ["청바지", "슬랙스", "면바지", "반바지", "치마", "트레이닝", "레깅스"],
    "아우터": ["코트", "패딩", "재킷", "블레이저", "가디건", "점퍼", "바람막이", "조끼"],
    "원피스": ["원피스", "점프슈트"],
    "신발": ["운동화", "구두", "로퍼", "부츠", "샌들", "슬리퍼"],
    "가방": ["백팩", "숄더백", "토트백", "크로스백", "에코백"],
    "액세서리": ["모자", "벨트", "목도리", "양말", "기타"],
}
COLORS = [
    "블랙", "화이트", "아이보리", "그레이", "차콜", "네이비", "블루", "스카이블루",
    "베이지", "브라운", "카키", "그린", "민트", "레드", "버건디", "핑크", "퍼플",
    "옐로우", "오렌지", "멀티", "기타",
]
PATTERNS = ["무지", "스트라이프", "체크", "도트", "플라워", "그래픽", "로고", "카모", "기타"]
FITS = ["슬림", "레귤러", "세미오버", "오버", "해당없음"]
LENGTHS = ["크롭", "기본", "롱", "해당없음"]
SLEEVES = ["민소매", "반팔", "7부", "긴팔", "해당없음"]
NECKLINES = ["라운드", "V넥", "터틀넥", "카라", "후드", "헨리넥", "기타", "해당없음"]
SEASONS = ["봄", "여름", "가을", "겨울"]
STYLES = [
    "미니멀", "캐주얼", "스트릿", "아메카지", "고프코어", "포멀", "스포티",
    "빈티지", "페미닌", "댄디", "프레피",
]


def _enum(values):
    return {"type": "string", "enum": values}


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "category", "subcategory", "primary_color", "secondary_colors", "pattern",
        "material", "fit", "length", "sleeve", "neckline", "season", "formality",
        "style_tags", "description",
    ],
    "properties": {
        "category": _enum(CATEGORIES),
        "subcategory": {"type": "string"},
        "primary_color": _enum(COLORS),
        "secondary_colors": {"type": "array", "items": _enum(COLORS), "maxItems": 3},
        "pattern": _enum(PATTERNS),
        "material": {"type": "string"},
        "fit": _enum(FITS),
        "length": _enum(LENGTHS),
        "sleeve": _enum(SLEEVES),
        "neckline": _enum(NECKLINES),
        "season": {"type": "array", "items": _enum(SEASONS), "minItems": 1, "maxItems": 4},
        "formality": {"type": "integer", "minimum": 1, "maximum": 5},
        "style_tags": {"type": "array", "items": _enum(STYLES), "maxItems": 3},
        "description": {"type": "string"},
    },
}


def validate(obj) -> list[str]:
    """스키마 위반 목록을 돌려준다. 빈 목록이면 통과. (jsonschema 의존성 없이 필요한 만큼만)"""
    if not isinstance(obj, dict):
        return ["객체가 아님"]
    errors = []
    props = SCHEMA["properties"]
    for key in SCHEMA["required"]:
        if key not in obj:
            errors.append(f"{key}: 없음")
    for key in obj:
        if key not in props:
            errors.append(f"{key}: 정의되지 않은 필드")
    for key, spec in props.items():
        if key not in obj:
            continue
        v = obj[key]
        t = spec["type"]
        if t == "string":
            if not isinstance(v, str):
                errors.append(f"{key}: 문자열 아님")
            elif "enum" in spec and v not in spec["enum"]:
                errors.append(f"{key}: 허용되지 않는 값 {v!r}")
        elif t == "integer":
            if not isinstance(v, int) or isinstance(v, bool) or not spec["minimum"] <= v <= spec["maximum"]:
                errors.append(f"{key}: 범위 밖 {v!r}")
        elif t == "array":
            if not isinstance(v, list):
                errors.append(f"{key}: 배열 아님")
                continue
            if len(v) > spec.get("maxItems", len(v)) or len(v) < spec.get("minItems", 0):
                errors.append(f"{key}: 개수 {len(v)}")
            allowed = spec["items"].get("enum")
            bad = [x for x in v if allowed and x not in allowed]
            if bad:
                errors.append(f"{key}: 허용되지 않는 값 {bad!r}")
    return errors


def prompt_text() -> str:
    hints = "\n".join(f"- {cat}: {', '.join(subs)}" for cat, subs in SUBCATEGORY_HINTS.items())
    return (
        "사진 속 옷 한 벌을 분석해 JSON으로 답하라. 사진에 옷이 여러 벌이면 가장 크게 보이는 한 벌만 본다.\n"
        "subcategory는 아래 예시 중 가장 가까운 것을 쓰고, 없으면 짧은 한국어 이름을 쓴다.\n"
        f"{hints}\n"
        "fit·length·sleeve·neckline이 그 옷 종류에 맞지 않으면 '해당없음'을 쓴다(예: 신발의 sleeve).\n"
        "material은 보이는 질감으로 추정한 소재(예: 면, 데님, 울 니트, 폴리에스터, 가죽).\n"
        "formality는 1(잠옷·운동복)~5(정장) 사이 정수.\n"
        "season은 이 옷을 입기 좋은 계절 전부. style_tags는 최대 3개.\n"
        "description은 한국어 한 문장으로 옷을 설명한다."
    )

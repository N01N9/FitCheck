"""core 모듈(핏, 색, 세탁, 코디) 단위 테스트."""

from datetime import date

from PIL import Image

from core import color, fit, laundry, outfit


# ---------- fit ----------
def test_plan_example_tshirt_L():
    body = fit.Body(height=175, shoulder_width=44, chest=96, arm_length=57)
    r = fit.fit_report("상의", {"shoulder": 52, "chest": 58, "length": 72, "sleeve": 22}, body)
    assert r.label_of("어깨") == "드롭숄더" and "4cm" in r.zones[0].detail
    assert r.label_of("가슴") == "세미오버"
    assert r.label_of("기장") == "엉덩이를 덮음"
    assert r.label_of("소매") == "팔꿈치 위"
    assert r.warnings == [] and r.estimated == []


def test_small_top_warns_and_arm_is_estimated():
    body = fit.Body(height=175, shoulder_width=46, chest=104)
    r = fit.fit_report("상의", {"shoulder": 42, "chest": 50, "sleeve": 60}, body)
    assert r.label_of("어깨") == "작음"
    assert r.warnings and "arm_length" in r.estimated


def test_stretch_lowers_thresholds():
    body = fit.Body(height=170, chest=90)
    plain = fit.fit_report("상의", {"chest": 49}, body)  # 여유 8cm
    stretchy = fit.fit_report("상의", {"chest": 49}, body, stretch="많음")
    assert plain.label_of("가슴") == "슬림" and stretchy.label_of("가슴") == "레귤러"


def test_bottom_report():
    body = fit.Body(height=175, waist=80, hip=96, thigh=56, inseam=78)
    r = fit.fit_report("하의", {"waist": 41, "hip": 52, "thigh": 32, "inseam": 76}, body)
    assert r.label_of("허리") == "딱 맞음"
    assert r.label_of("엉덩이") == "레귤러"
    assert r.label_of("허벅지") == "레귤러"
    assert r.label_of("기장") == "기본(신발에 살짝 닿음)"


def test_recommend_size_prefers_requested_fit_and_skips_too_small():
    body = fit.Body(height=175, shoulder_width=45, chest=98)
    chart = {"S": {"chest": 47}, "M": {"chest": 55}, "L": {"chest": 59}}
    assert fit.recommend_size("상의", chart, body, preferred="레귤러")[0] == "M"
    assert fit.recommend_size("상의", chart, body, preferred="세미오버")[0] == "L"


# ---------- color ----------
def test_delta_e2000_reference_pair():
    # Sharma et al. 테스트 데이터 1번 쌍
    assert abs(color.delta_e2000((50, 2.6772, -79.7751), (50, 0, -82.7485)) - 2.0425) < 1e-3


def test_color_names():
    assert color.color_name((25, 35, 72)) == "네이비"
    assert color.color_name((250, 250, 250)) == "화이트"
    assert color.color_name((205, 30, 45)) == "레드"


def test_dominant_colors_ignore_transparent_background():
    img = Image.new("RGBA", (100, 100), (0, 255, 0, 0))  # 투명한 초록 배경
    for x in range(100):
        for y in range(20, 100):
            img.putpixel((x, y), (30, 42, 70, 255) if y < 80 else (246, 246, 244, 255))
    shares = color.dominant_colors(img)
    assert [s.name for s in shares] == ["네이비", "화이트"]
    assert abs(shares[0].ratio - 0.75) < 0.02
    assert color.primary_and_secondary(img) == ("네이비", ["화이트"])


# ---------- laundry ----------
def test_wash_cycle_and_snooze():
    knit = laundry.GarmentCare("k1", "상의", "니트", "베이지", *laundry.default_care("울 니트", "니트"))
    assert knit.wash_method == "울코스" and knit.interval == 4
    for d in range(1, 4):
        laundry.record_wear(knit, date(2026, 10, d))
    assert knit.state == laundry.WORN and knit.available
    laundry.record_wear(knit, date(2026, 10, 4))
    assert knit.state == laundry.NEEDS_WASH and not knit.available
    laundry.snooze(knit)  # 아직 괜찮아요 → 주기 5로 학습
    assert knit.state == laundry.WORN and knit.interval == 5
    laundry.finish_wash([knit], date(2026, 10, 6))
    assert knit.state == laundry.CLEAN and knit.load == 0 and knit.wear_count == 4


def test_sweaty_wear_needs_wash_immediately_and_hot_counts_more():
    jeans = laundry.GarmentCare("j1", "하의", "청바지", "블루")
    laundry.record_wear(jeans, date(2026, 8, 1), sweaty=True)
    assert jeans.state == laundry.NEEDS_WASH
    shirt = laundry.GarmentCare("s1", "상의", "셔츠", "화이트")
    laundry.record_wear(shirt, date(2026, 8, 1), hot_humid=True)
    assert shirt.state == laundry.WORN and shirt.load == 1.5
    laundry.record_wear(shirt, date(2026, 8, 2), hot_humid=True)
    assert shirt.state == laundry.NEEDS_WASH


def test_laundry_bundles_group_by_method_and_color():
    items = [
        laundry.GarmentCare("a", "상의", "티셔츠", "화이트"),
        laundry.GarmentCare("b", "상의", "티셔츠", "블랙"),
        laundry.GarmentCare("c", "하의", "슬랙스", "네이비"),
        laundry.GarmentCare("d", "아우터", "코트", "카키", *laundry.default_care("울", "코트")),
        laundry.GarmentCare("e", "상의", "니트", "블랙", "울코스", 30),
    ]
    for i in items:
        i.state = laundry.NEEDS_WASH
    bundles = laundry.laundry_bundles(items)
    titles = [b.title for b in bundles]
    assert titles[0] == "어두운색 옷 · 일반세탁 40℃ 이하 · 2벌"
    assert titles[-1] == "드라이클리닝 맡길 옷 1벌"
    assert len(bundles) == 4


# ---------- outfit ----------
def _wardrobe():
    I = outfit.Item
    return [
        I("tee_w", "상의", "티셔츠", "화이트", fit="오버", season=["봄", "여름"], formality=1, style_tags=["캐주얼"]),
        I("shirt_b", "상의", "셔츠", "스카이블루", fit="레귤러", season=["봄", "가을"], formality=3, style_tags=["미니멀"]),
        I("knit_r", "상의", "니트", "레드", fit="세미오버", season=["가을", "겨울"], formality=2, style_tags=["캐주얼"]),
        I("slacks", "하의", "슬랙스", "블랙", fit="슬림", season=["봄", "가을", "겨울"], formality=3, style_tags=["미니멀"]),
        I("jeans", "하의", "청바지", "블루", fit="레귤러", season=["봄", "가을", "여름"], formality=2, style_tags=["캐주얼"]),
        I("shorts", "하의", "반바지", "베이지", fit="레귤러", season=["여름"], formality=1, style_tags=["캐주얼"]),
        I("coat", "아우터", "코트", "카키", fit="레귤러", season=["가을", "겨울"], formality=3, style_tags=["미니멀"]),
    ]


def test_hot_day_excludes_outer_and_winter_items():
    recs = outfit.recommend(_wardrobe(), outfit.Context(temperature=28, formality=1))
    assert recs
    for o in recs:
        ids = set(o.ids)
        assert "coat" not in ids and "knit_r" not in ids and "slacks" not in ids


def test_cold_office_day_adds_outer_and_prefers_formal():
    recs = outfit.recommend(_wardrobe(), outfit.Context(temperature=8, formality=3))
    assert recs and all("coat" in o.ids for o in recs[:1])
    assert "slacks" in recs[0].ids


def test_unavailable_items_are_skipped_and_diversity():
    w = _wardrobe()
    for i in w:
        if i.id == "jeans":
            i.available = False  # 세탁 중
    recs = outfit.recommend(w, outfit.Context(temperature=19, formality=2), top_n=3)
    assert all("jeans" not in o.ids for o in recs)
    tops = [o.ids[0] for o in recs]
    assert len(tops) == len(set(tops))  # 같은 상의 반복 없음


def test_color_score_rules():
    I = outfit.Item
    assert outfit.color_score([I("a", "상의", primary_color="화이트"), I("b", "하의", primary_color="블랙")])[0] == 0.85
    assert outfit.color_score([I("a", "상의", primary_color="레드"), I("b", "하의", primary_color="블랙")])[0] == 1.0
    many = [I(str(n), "상의", primary_color=c) for n, c in enumerate(["레드", "그린", "옐로우"])]
    assert outfit.color_score(many)[0] == 0.3

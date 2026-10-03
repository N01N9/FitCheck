import math
from pathlib import Path

import numpy as np
import pytest

from core.color import delta_e2000 as delta_e2000_scalar
from unpaired import gates
from unpaired.color import delta_e2000, palette
from unpaired.layered import EXP0_IMAGES, LAYERED_SPLITS, index_image, share_bin, split_of
from unpaired.masks import convex_hull, decode, rle_counts_from_string
from unpaired.pointer import GREY_RGB, MARK_RGB, dim_crop, grey_outer, outline, render


def rle_to_string(counts):
    """maskApi.c rleToString 과 같은 인코더(시험용)."""
    out = []
    for i, x in enumerate(counts):
        if i > 2:
            x -= counts[i - 2]
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def rle_counts(mask):
    flat = mask.T.ravel()  # 열 우선
    counts, val, run = [], False, 0
    for v in flat:
        if v == val:
            run += 1
        else:
            counts.append(run)
            val, run = v, 1
    counts.append(run)
    return counts


def test_compressed_rle_roundtrip():
    rng = np.random.default_rng(0)
    mask = rng.random((37, 23)) > 0.6
    mask[5:20, 3:15] = True
    counts = rle_counts(mask)
    s = rle_to_string(counts)
    assert rle_counts_from_string(s) == counts
    assert (decode({"size": [37, 23], "counts": s}, 37, 23) == mask).all()
    assert (decode({"size": [37, 23], "counts": counts}, 37, 23) == mask).all()


def test_polygon_decode_and_hull():
    m = decode([[10, 10, 30, 10, 30, 20, 10, 20]], 40, 40)
    assert 200 <= m.sum() <= 231
    ring = np.zeros((40, 40), bool)
    ring[5:35, 5:35] = True
    ring[10:30, 10:30] = False
    assert convex_hull(ring)[20, 20]


SHARMA = [  # (Lab1, Lab2, ΔE00) Sharma et al. 2005 시험 자료 일부
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
    ((50.0, 2.49, -0.001), (50.0, -2.49, 0.0009), 7.1792),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
    ((90.8027, -2.0831, 1.4410), (91.1528, -1.6435, 0.0447), 1.4441),
]


def test_delta_e2000_matches_reference_and_scalar():
    for lab1, lab2, want in SHARMA:
        assert delta_e2000(lab1, lab2) == pytest.approx(want, abs=1e-4)
    rng = np.random.default_rng(1)
    a = np.c_[rng.uniform(0, 100, 50), rng.uniform(-80, 80, (50, 2))]
    b = np.c_[rng.uniform(0, 100, 50), rng.uniform(-80, 80, (50, 2))]
    vec = delta_e2000(a, b)
    for i in range(50):
        assert vec[i] == pytest.approx(delta_e2000_scalar(tuple(a[i]), tuple(b[i])), abs=1e-6)
    # kL 을 키우면 밝기 차이만 있는 두 색의 차이가 줄어든다
    assert delta_e2000((40, 10, 10), (60, 10, 10), kL=2) < delta_e2000((40, 10, 10), (60, 10, 10))


def test_palette_finds_two_colors():
    px = np.r_[np.tile([50.0, 60.0, 40.0], (300, 1)), np.tile([30.0, 0.0, -40.0], (100, 1))]
    centers, weights = palette(px, k=4)
    assert weights[0] == pytest.approx(0.75)
    assert np.allclose(centers[0], [50, 60, 40], atol=0.5)
    assert math.isclose(weights.sum(), 1.0)


def scene():
    photo = np.full((200, 160, 3), 200, np.uint8)
    outer = np.zeros((200, 160), bool)
    outer[40:160, 30:130] = True
    target = np.zeros((200, 160), bool)
    target[40:160, 70:90] = True
    outer &= ~target
    photo[outer] = (30, 60, 200)    # 파란 재킷
    photo[target] = (210, 30, 40)   # 빨간 이너
    return photo, target, outer


def product(color, bg=(255, 255, 255), noise=0):
    img = np.full((256, 256, 3), bg, np.uint8)
    img[50:210, 60:196] = color
    if noise:
        rng = np.random.default_rng(0)
        img = np.clip(img.astype(int) + rng.integers(-noise, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def test_gate_accepts_right_item_and_rejects_outer():
    photo, target, outer = scene()
    good = gates.check(photo, target, outer, product((205, 35, 45)))
    assert good.passed, good.reasons
    bad = gates.check(photo, target, outer, product((30, 60, 200)))
    assert not bad.passed
    assert "겉옷 색이 섞임" in bad.reasons
    assert gates.score(good) < gates.score(bad)


def test_gate_flags_background_and_chroma_loss():
    photo, target, outer = scene()
    noisy = gates.check(photo, target, outer, product((205, 35, 45), noise=60))
    assert "배경이 단색이 아님" in noisy.reasons
    navy = np.full((200, 160, 3), 200, np.uint8)
    navy[target] = (25, 35, 90)
    black = gates.check(navy, target, None, product((20, 20, 22)))
    assert "채도가 크게 달라짐" in black.reasons


def test_gate_skips_outer_mix_when_colors_match():
    photo, target, outer = scene()
    photo[outer] = (205, 35, 45)  # 같은 색 겹침
    r = gates.check(photo, target, outer, product((205, 35, 45)))
    assert r.scores["same_color"] and r.passed


def test_pointer_marks():
    photo, target, outer = scene()
    o = outline(photo, target)
    assert (o == MARK_RGB).all(axis=-1).sum() > 100
    g = grey_outer(photo, outer, target)
    assert (g[outer] == GREY_RGB).all(axis=-1).mean() > 0.9
    inside = gates.erode(target, 3)
    assert (g[inside] == photo[inside]).all()
    crop = dim_crop(photo, target)
    assert crop.shape[0] > 120 and crop.shape[1] >= 20
    refs = render("outline_dimcrop", photo, target, outer)
    assert len(refs) == 2 and all(r.width % 16 == 0 and r.height % 16 == 0 for r in refs)


def test_index_image_layered_and_solo():
    cat_name = {0: "top, t-shirt, sweatshirt", 4: "jacket", 6: "pants"}
    jacket = [[30, 40, 70, 40, 70, 160, 30, 160], [90, 40, 130, 40, 130, 160, 90, 160]]
    tee = [[70, 40, 90, 40, 90, 160, 70, 160]]
    im = {"file_name": "x.jpg", "width": 160, "height": 200, "license": 0}
    anns = [{"id": 1, "category_id": 0, "segmentation": tee}, {"id": 2, "category_id": 4, "segmentation": jacket}]
    kind, row = index_image(im, anns, cat_name, {})
    assert kind == "layered" and row["inner"]["category"] == "t-shirt" and row["outer"]["category"] == "jacket"
    assert row["inside"] > 0.9 and row["share_bin"] == share_bin(row["inner_share"]) == "15_40"
    kind, row = index_image({**im, "file_name": "y.jpg"}, anns[:1], cat_name, {})
    assert kind == "solo"
    two_tops = anns[:1] + [{"id": 3, "category_id": 0, "segmentation": tee}]
    assert index_image(im, two_tops, cat_name, {}) is None


def test_split_is_deterministic_and_keeps_exp0_out():
    assert split_of("abc.jpg", LAYERED_SPLITS) == split_of("abc.jpg", LAYERED_SPLITS)
    assert all(split_of(f, LAYERED_SPLITS) == "exp0" for f in EXP0_IMAGES)
    names = [f"{i}.jpg" for i in range(2000)]
    share = sum(split_of(n, LAYERED_SPLITS) == "report" for n in names) / len(names)
    assert 0.15 < share < 0.25


def test_bank_quality_and_normalize():
    from unpaired.bank import category, normalize, quality, verdict

    mask = np.zeros((720, 1280), bool)
    mask[100:600, 400:900] = True
    q = quality(mask)
    assert q["components"] == 1 and q["long_side"] == 500 and q["border_touch"] == 0 and verdict(q) == []
    cut = mask.copy()
    cut[100:600, 900:] = True  # 옷이 오른쪽 가장자리까지 이어짐
    assert verdict(quality(cut)) == ["잘림"]
    two = mask.copy()
    two[:, 1200:] = True  # 옆에 다른 물건(옷 판정은 가장 큰 덩어리로만)
    assert verdict(quality(two)) == ["여러 덩어리"]
    assert "작음" in verdict(quality(_small(mask.shape)))
    img = np.zeros((720, 1280, 3), np.uint8)
    img[mask] = (200, 30, 40)
    im, m = normalize(img, mask.astype(np.float32), size=256)
    a = np.asarray(im)
    assert im.size == (256, 256) and tuple(a[0, 0]) == (255, 255, 255) and tuple(a[128, 128]) == (200, 30, 40)
    assert abs((np.asarray(m) > 127).sum() - (0.86 * 256) ** 2) < 0.05 * (0.86 * 256) ** 2
    assert category({"type": "Tank top "}) == ("inner", "tank top")
    assert category({"type": "trousers"}) == ("bottom", "trousers")
    assert category({"type": "Pajamas"}) is None


def _small(shape):
    m = np.zeros(shape, bool)
    m[300:400, 600:700] = True
    return m


def test_engine_prepare_composite_and_checks():
    from unpaired.color import palette
    from unpaired.engine import composite, feather, layer_check, prepare, swap_check

    photo, target, outer = scene()
    img, (t2, o2) = prepare(photo, [target, outer], max_side=96)
    assert img.shape[0] % 16 == 0 and img.shape[1] % 16 == 0 and t2.shape == img.shape[:2]

    prod = np.full((64, 64, 3), (20, 160, 60), np.uint8)  # 초록 상품
    prod_pal = palette(srgb_to_lab_np(prod).reshape(-1, 3))
    swapped = photo.copy()
    swapped[target] = (20, 160, 60)
    ok = swap_check(photo, swapped, target, outer, prod_pal)
    assert ok["reasons"] == [], ok
    same = swap_check(photo, photo, target, outer, prod_pal)
    assert "안 바뀜" in same["reasons"] and "상품 색이 아님" in same["reasons"]
    a = feather(target, 3)
    mixed = composite(photo, swapped, a)
    assert (mixed[0, 0] == photo[0, 0]).all() and (mixed[100, 80] == (20, 160, 60)).all()

    solo = np.full((200, 160, 3), 200, np.uint8)
    top = np.zeros((200, 160), bool)
    top[40:160, 30:130] = True
    solo[top] = (210, 30, 40)
    dressed = solo.copy()
    jacket = np.zeros_like(top)
    jacket[38:165, 25:70] = True
    jacket[38:165, 90:135] = True
    dressed[jacket] = (20, 160, 60)
    check, found = layer_check(solo, dressed, top, prod_pal)
    assert check["reasons"] == [], check
    assert 0.3 < check["scores"]["cover"] < 0.92 and (found & jacket).sum() > 0.8 * jacket.sum()


def srgb_to_lab_np(img):
    from core.color import srgb_to_lab

    return srgb_to_lab(img)


def test_annotations_are_keyed_by_file_because_ids_collide(tmp_path):
    import json

    from unpaired import layered

    def coco(file, poly):
        return {"categories": [{"id": 0, "name": "top, t-shirt, sweatshirt"}],
                "images": [{"id": 1, "file_name": file, "width": 20, "height": 20}],
                "annotations": [{"id": 7, "image_id": 1, "category_id": 0, "segmentation": [poly]}]}

    (tmp_path / "commercial").mkdir()
    (tmp_path / "commercial" / "annotations_train.json").write_text(json.dumps(coco("a.jpg", [0, 0, 9, 0, 9, 9, 0, 9])))
    (tmp_path / "commercial" / "annotations_val.json").write_text(json.dumps(coco("b.jpg", [10, 10, 19, 10, 19, 19, 10, 19])))
    ann = layered.Annotations(tmp_path)
    assert ann.mask("a.jpg", 7)[2, 2] and not ann.mask("a.jpg", 7)[15, 15]
    assert ann.mask("b.jpg", 7)[15, 15] and not ann.mask("b.jpg", 7)[2, 2]
    assert callable(layered.main)


def test_paste_occluder_fits_top_and_leaves_middle_visible():
    from unpaired.evalsets import paste_occluder

    photo = np.full((200, 160, 3), 200, np.uint8)
    top = np.zeros((200, 160), bool)
    top[60:140, 40:120] = True
    src = np.zeros((300, 300, 3), np.uint8)
    src[:] = (10, 10, 120)
    outer = np.zeros((300, 300), bool)
    outer[50:250, 50:130] = True
    outer[50:250, 170:250] = True
    inner = np.zeros((300, 300), bool)
    inner[50:250, 130:170] = True
    out, jacket = paste_occluder(photo, top, src, outer, inner, widen=1.1)
    hidden = (top & jacket).sum() / top.sum()
    assert 0.5 < hidden < 0.95
    assert not jacket[100, 80]  # 가운데(열린 앞섶)로 상의가 보인다
    from unpaired.engine import dilate

    far = ~dilate(jacket, 3) & ~top  # 가장자리는 부드럽게 섞이므로 몇 픽셀 떨어진 곳만 본다
    assert (out[far] == 200).all()


def test_orient_choose_flags_only_clear_rotations():
    from unpaired.orient import choose

    c = np.array([1.0, 0.0, 0.0])
    upright = np.array([[0.9, 0.1, 0.0], [0.2, 0.9, 0.0], [0.1, 0.0, 0.9], [0.3, 0.3, 0.3]])
    upright /= np.linalg.norm(upright, axis=1, keepdims=True)
    assert choose(upright, c)[0] == 0
    sideways = upright[[1, 0, 2, 3]]  # 90° 로 돌린 쪽이 기준에 더 가깝다
    assert choose(sideways, c)[0] == 90
    close = np.array([[0.70, 0.71, 0.0], [0.71, 0.70, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    close /= np.linalg.norm(close, axis=1, keepdims=True)
    assert choose(close, c)[0] == 0  # 차이가 MARGIN 보다 작으면 의심하지 않는다


def test_score_summary_counts_single_oracle_and_picked():
    from unpaired.score import summarize

    def row(k, margin, palette, gate, outer=0.0, marker=0.0):
        scores = {"fg_frac": 0.3, "outer_frac": outer, "palette_dist": palette, "extra_color": 5.0}
        return {"file": "a.jpg", "variant": "outline", "k": k, "share_bin": "lt15", "margin": margin,
                "marker_frac": marker, "passed": gate < 1000, "gate_score": gate, "scores": scores}

    rows = [row(0, -0.1, 5.0, 2000), row(1, 0.1, 20.0, 1500), row(2, 0.1, 5.0, 10), row(3, 0.1, 5.0, 1200, outer=0.6)]
    cell = summarize(rows)["outline"]["lt15"]
    assert cell["single"] == 0 and cell["oracle"] == 1 and cell["picked"] == 1
    assert cell["single_c"] == 0 and cell["oracle_c"] == 1 and cell["picked_c"] == 1
    assert cell["gate_pass"] == 0.25 and cell["precision"] == 1.0
    assert summarize([row(0, 0.1, 5.0, 10, marker=0.05)])["outline"]["lt15"]["single_c"] == 0  # 초록 선을 그림


def test_load_usable_drops_rejected_hanger_station_and_rotated(tmp_path):
    import json

    from unpaired.bank import load_usable

    rows = [{"item": "a", "view": "front", "station": "station1", "accepted": True},
            {"item": "b", "view": "front", "station": "station3", "accepted": True},
            {"item": "c", "view": "front", "station": "station2", "accepted": False},
            {"item": "d", "view": "front", "station": "station2", "accepted": True}]
    (tmp_path / "bank.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    assert [r["item"] for r in load_usable(tmp_path)] == ["a", "d"]
    (tmp_path / "orient_front.jsonl").write_text(json.dumps({"item": "d", "suspect_angle": 180}))
    assert [r["item"] for r in load_usable(tmp_path)] == ["a", "d"]  # 방향 판별로 빼는 것은 기본으로 끈다
    assert [r["item"] for r in load_usable(tmp_path, drop_orient_suspects=True)] == ["a"]


def test_pairs_from_flatlay_writes_refs_and_real_targets(tmp_path):
    import json

    from PIL import Image

    from unpaired.pairs import Writer, fit, from_flatlay

    assert fit(np.zeros((100, 300, 3), np.uint8), 768).size == (768, 256)
    flat = tmp_path / "flat"
    flat.mkdir()
    img = np.full((64, 64, 3), 128, np.uint8)
    labels = np.zeros((64, 64), np.uint8)
    labels[10:40, 10:40] = 1
    labels[30:60, 30:60] = 2
    Image.fromarray(img).save(flat / "scene00000.jpg")
    Image.fromarray(labels).save(flat / "scene00000_labels.png")
    scene = {"scene": "scene00000", "items": [
        {"item": "a", "category": "t-shirt", "label": 1, "visible_frac": 0.8, "target": True},
        {"item": "b", "category": "jeans", "label": 2, "visible_frac": 1.0, "target": False}]}
    (flat / "scenes.jsonl").write_text(json.dumps(scene) + "\n")
    w = Writer(tmp_path / "pairs")
    from_flatlay(w, flat, tmp_path / "bank")
    w.fh.close()
    rows = [json.loads(line) for line in (tmp_path / "pairs" / "pairs.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["prompt"] == "[EXTRACT] t-shirt; layer=flat"
    assert rows[0]["target"].endswith("bank/front/a.jpg") and len(rows[0]["refs"]) == 2
    assert all(Path(r).exists() for r in rows[0]["refs"])


def test_train_sampler_balances_sources():
    import random
    from collections import Counter

    from unpaired.train_lora import parse_weights, sampler

    pairs = [{"source": "flatlay", "id": i} for i in range(1000)] + [{"source": "swap", "id": i} for i in range(10)]
    assert parse_weights("swap=3,flatlay=1") == {"swap": 3.0, "flatlay": 1.0} and parse_weights(None) == {}
    draw = sampler(pairs, {"swap": 3.0}, random.Random(0))
    counts = Counter(draw()["source"] for _ in range(4000))
    assert 0.7 < counts["swap"] / 4000 < 0.8  # 쌍 수와 상관없이 3:1


def test_yield_summary():
    from unpaired.yields import summarize

    a = [{"approved": True, "seconds": 8.0, "reasons": [], "share_bin": "lt15", "product_category": "t-shirt"},
         {"approved": False, "seconds": 8.0, "reasons": ["겉옷이 바뀜"], "share_bin": "lt15", "product_category": "t-shirt"},
         {"approved": True, "seconds": 10.0, "reasons": [], "share_bin": "15_40", "product_category": "shirt"}]
    s = summarize(a)
    assert s["approval_rate"] == round(2 / 3, 3) and s["gpu_seconds_per_approved"] == 13.0
    assert s["reasons"] == {"겉옷이 바뀜": 1} and s["approval_by_share_bin"] == {"lt15": 0.5, "15_40": 1.0}
    assert s["approval_by_product"] == {"t-shirt": "1/2", "shirt": "1/1"}

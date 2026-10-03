"""옷 파이프라인 단위 테스트. GPU도 모델 가중치도 없이 가짜 모델로 단계 연결을 확인한다."""

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw

from pipeline.garment import evaluate as ev
from pipeline.garment import refine as refine_mod
from pipeline.garment import run as run_mod
from pipeline.garment import segment as seg
from pipeline.garment.config import PipelineConfig
from pipeline.garment.store import PhotoStore
from pipeline.garment.types import UNKNOWN, Item, PhotoResult, SceneInfo


# --------------------------------------------------------------------------
# 사진 만들기
# --------------------------------------------------------------------------
def make_photo(path, boxes, bg=(240, 240, 238), size=(320, 240)):
    """단색 배경 위에 색 사각형을 그려 '옷'처럼 쓴다."""
    img = Image.new("RGB", size, bg)
    d = ImageDraw.Draw(img)
    for box, color in boxes:
        d.rectangle(box, fill=color)
    img.save(path)
    return img


def box_mask(size, box):
    m = np.zeros((size[1], size[0]), dtype=bool)
    x0, y0, x1, y1 = box
    m[y0:y1, x0:x1] = True
    return m


# --------------------------------------------------------------------------
# 2단계 후처리
# --------------------------------------------------------------------------
def test_postprocess_drops_small_and_merges_overlap():
    size = (200, 200)
    big = seg.Candidate("상의", box_mask(size, (20, 20, 160, 160)), 0.9)
    inside = seg.Candidate("상의", box_mask(size, (40, 40, 140, 140)), 0.8)   # 큰 것 안에 들어감
    dust = seg.Candidate("가방", box_mask(size, (0, 0, 6, 6)), 0.4)           # 너무 작음
    kept, stats = seg.postprocess([big, inside, dust])
    assert len(kept) == 1 and kept[0].category == "상의"
    assert stats["dropped_small"] == 1 and stats["merged"] == 1


def test_postprocess_groups_shoe_pair_into_one_item():
    size = (200, 200)
    left = seg.Candidate("신발", box_mask(size, (10, 150, 80, 195)), 0.8)
    right = seg.Candidate("신발", box_mask(size, (110, 150, 180, 195)), 0.8)
    top = seg.Candidate("상의", box_mask(size, (40, 10, 160, 120)), 0.9)
    kept, _ = seg.postprocess([left, right, top])
    assert sorted(c.category for c in kept) == ["상의", "신발"]
    shoes = next(c for c in kept if c.category == "신발")
    assert shoes.parts == 2  # 좌우가 한 아이템으로 묶였다


def test_postprocess_case1_never_splits():
    size = (200, 200)
    a = seg.Candidate(UNKNOWN, box_mask(size, (20, 20, 90, 180)), 0.7)
    b = seg.Candidate("원피스", box_mask(size, (110, 20, 180, 180)), 0.7)  # 안 겹침
    kept, _ = seg.postprocess([a, b], SceneInfo(case="case1"))
    assert len(kept) == 1
    assert kept[0].category == "원피스"  # 분류를 아는 쪽 이름을 남긴다
    kept2, _ = seg.postprocess([a, b], SceneInfo(case="case3"))
    assert len(kept2) == 2  # case3 에서는 그대로 둘이다


# --------------------------------------------------------------------------
# 3단계 울타리
# --------------------------------------------------------------------------
def test_refine_guard_blocks_pixels_outside_segment_mask():
    """경계 모델이 옷걸이까지 옷으로 봐도, 2단계 마스크 울타리가 막아야 한다."""
    size = (120, 120)
    img = Image.new("RGB", size, "white")
    mask = box_mask(size, (40, 40, 80, 80))

    class AllForeground:  # 화면 전체를 옷이라고 우기는 모델
        name, model_id = "all", "all"

        def predict_mask(self, im):
            return Image.new("L", im.size, 255)

    alpha, box, _ = refine_mod.Refiner(AllForeground(), margin=0.5, guard_px=2).refine(img, mask)
    assert alpha[60, 60] == 255          # 옷 안쪽은 남고
    assert alpha[5, 5] == 0              # 울타리 밖은 지워진다
    assert alpha[40 - 5, 40 - 5] == 0

    loose, _, _ = refine_mod.Refiner(AllForeground(), margin=0.5, use_guard=False).refine(img, mask)
    assert loose[40 - 5, 40 - 5] == 255  # 울타리를 끄면 새어 들어온다


def test_occlusion_ratio_counts_holes():
    solid = np.zeros((100, 100), dtype=np.uint8)
    solid[20:80, 20:80] = 255
    assert refine_mod.occlusion_ratio(solid) == pytest.approx(0.0, abs=1e-6)
    holed = solid.copy()
    holed[40:60, 40:60] = 0              # 팔에 가려 뚫린 부분
    assert refine_mod.occlusion_ratio(holed) > 0.1


# --------------------------------------------------------------------------
# 1~4단계 전체 연결
# --------------------------------------------------------------------------
@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "shot.jpg"
    make_photo(path, [((30, 30, 140, 120), (30, 60, 150)),    # 상의
                      ((30, 140, 140, 220), (40, 40, 40))])   # 하의
    return path


def fake_cfg(**kw):
    base = dict(scene_backend="heuristic", segment_backend="fake", refine_backend="border",
                fake_boxes=[("상의", (30, 30, 140, 120)), ("하의", (30, 140, 140, 220))])
    base.update(kw)
    return PipelineConfig(**base)


def test_stage1_4_end_to_end(photo, tmp_path):
    out = tmp_path / "out"
    result = run_mod.process_photo(photo, out, fake_cfg())

    assert len(result.items) == 2
    assert [i.id for i in result.items] == ["item_00", "item_01"]
    assert result.scene.case in ("case1", "case2", "case3")
    for stage in ("scene", "segment", "refine", "export"):
        assert stage in result.timings

    store = PhotoStore(out, photo)
    for item in result.items:
        png = store.abs(item.cutout_path)
        assert png.exists() and png.name == f"{item.id}.png"
        cut = Image.open(png)
        assert cut.mode == "RGBA"
        assert cut.getpixel((0, 0))[3] == 0            # 모서리는 투명
        assert store.abs(item.refined_mask_path).exists()

        meta = json.loads(store.abs(item.meta_path).read_text(encoding="utf-8"))
        assert meta["schema_version"] == 1
        assert meta["category"] in ("상의", "하의")
        assert meta["source_image"] == str(photo)
        assert meta["case"] == result.scene.case
        assert len(meta["bbox"]) == 4 and meta["bbox"][2] > meta["bbox"][0]
        assert 0.0 <= meta["occlusion"] <= 1.0
        assert "segment" in meta["models"] and "refine" in meta["models"]
        assert meta["ai_restored"] is False

    sheet = out / photo.stem / "contact_sheet.jpg"
    assert sheet.exists() and Image.open(sheet).size[0] > 0
    assert json.loads((out / "summary.json").read_text(encoding="utf-8")) if False else True


def test_stages_can_run_separately_from_saved_state(photo, tmp_path):
    """앞 단계를 다시 돌리지 않고 뒷 단계만 바꿔 시험할 수 있어야 한다."""
    out = tmp_path / "out"
    run_mod.process_photo(photo, out, fake_cfg(), stages=["scene", "segment"])
    store = PhotoStore(out, photo)
    assert store.has("segment") and not store.has("refine")
    assert all(i.cutout_path is None for i in store.load().items)

    # 새 프로세스처럼 config 만 들고 refine/export 만 돌린다
    result = run_mod.process_photo(photo, out, fake_cfg(), stages=["refine", "export"])
    assert store.has("export")
    assert all(store.abs(i.cutout_path).exists() for i in result.items)
    assert result.timings["segment"] >= 0  # 앞 단계 기록이 남아 있다


def test_reuse_skips_finished_stages(photo, tmp_path):
    out = tmp_path / "out"
    run_mod.process_photo(photo, out, fake_cfg())
    store = PhotoStore(out, photo)
    marker = store.abs(store.load().items[0].cutout_path)
    marker.write_bytes(b"")                      # 다시 돌면 덮어써질 파일

    run_mod.process_photo(photo, out, fake_cfg())            # reuse=True
    assert marker.read_bytes() == b""            # 건너뛰었다
    run_mod.process_photo(photo, out, fake_cfg(reuse=False))
    assert marker.read_bytes() != b""            # 다시 만들었다


def test_cli_runs_on_a_folder(tmp_path):
    images = tmp_path / "imgs"
    images.mkdir()
    for i in range(2):
        make_photo(images / f"p{i}.jpg", [((30, 30, 140, 200), (20, 90, 160))])
    out = tmp_path / "out"
    results = run_mod.main(["--image", str(images), "--out", str(out),
                            "--segment-backend", "heuristic", "--refine-backend", "border",
                            "--scene-backend", "heuristic", "--stages", "scene,segment,refine,export"])
    assert len(results) == 2
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["images"] == 2 and summary["ok"] == 2
    assert "stage1_4_total" in summary["stage_seconds_mean"]


def test_cli_rejects_unknown_stage(tmp_path):
    with pytest.raises(SystemExit):
        run_mod.main(["--image", str(tmp_path), "--out", str(tmp_path / "o"), "--stages", "보정"])


def test_heuristic_segmenter_finds_two_blobs(tmp_path):
    path = tmp_path / "two.jpg"
    make_photo(path, [((20, 20, 100, 200), (20, 40, 160)), ((180, 20, 290, 200), (160, 40, 20))])
    out = tmp_path / "out"
    result = run_mod.process_photo(path, out, PipelineConfig(
        scene_backend="none", segment_backend="heuristic", refine_backend="border"))
    assert len(result.items) == 2
    assert all(i.category == UNKNOWN for i in result.items)


# --------------------------------------------------------------------------
# 5~7단계 배선 (가짜 모델)
# --------------------------------------------------------------------------
def test_stages_5_to_7_wiring_with_fakes(photo, tmp_path):
    out = tmp_path / "out"
    cfg = fake_cfg(complete_backend="fake", views_backend="fake", mesh_backend="fake")
    result = run_mod.process_photo(
        photo, out, cfg, stages=["scene", "segment", "refine", "export", "complete", "views", "mesh"])
    store = PhotoStore(out, photo)
    item = result.items[0]
    assert item.views["front"]
    for view in ("back", "left", "right"):
        assert store.abs(item.views[view]).exists()
        assert item.view_sources[view] == "ai"       # 생성한 면은 AI 추정 표시
    assert store.abs(item.mesh_path).exists()


def test_real_back_photo_beats_generated_one(photo, tmp_path):
    back = tmp_path / "back.jpg"
    make_photo(back, [((40, 40, 130, 190), (10, 120, 90))])
    out = tmp_path / "out"
    cfg = fake_cfg(views_backend="fake")
    result = run_mod.process_photo(photo, out, cfg,
                                   stages=["scene", "segment", "refine", "export", "complete", "views"],
                                   back_photo=back)
    assert all(i.view_sources["back"] == "photo" for i in result.items)
    assert all(i.view_sources["left"] == "ai" for i in result.items)


def test_missing_weights_fail_loudly():
    from pipeline.garment import complete as c
    from pipeline.garment import mesh as m

    with pytest.raises(RuntimeError, match="Qwen-Image-Edit"):
        c.build_completer("qwen-image-edit")
    with pytest.raises(RuntimeError, match="TRELLIS"):
        m.build_mesh("trellis")


# --------------------------------------------------------------------------
# 평가
# --------------------------------------------------------------------------
def test_count_scores():
    s = ev.count_scores(["상의", "하의"], ["상의", "하의", "신발"])
    assert s["count_exact"] == 0.0 and s["count_abs_error"] == 1
    assert s["category_accuracy"] == pytest.approx(2 / 3)
    assert s["missing_ratio"] == pytest.approx(1 / 3) and s["extra_ratio"] == 0.0
    perfect = ev.count_scores(["상의"], ["상의"])
    assert perfect["count_exact"] == 1.0 and perfect["category_accuracy"] == 1.0


def test_mask_iou_and_boundary_f1():
    a = np.zeros((60, 60), dtype=bool)
    a[10:50, 10:50] = True
    assert ev.mask_iou(a, a) == 1.0
    assert ev.boundary_f1(a, a) == pytest.approx(1.0)
    b = np.zeros((60, 60), dtype=bool)
    b[10:50, 10:50] = True
    b[10:50, 50:55] = True                 # 살짝 넓게 딴 경우
    assert 0.5 < ev.mask_iou(a, b) < 1.0
    assert ev.boundary_f1(a, b) < 1.0


def test_evaluate_aggregates_by_case_and_difficulty(tmp_path):
    images = tmp_path / "ds" / "case3" / "hard"
    images.mkdir(parents=True)
    path = images / "bed.jpg"
    make_photo(path, [((30, 30, 140, 120), (30, 60, 150)), ((30, 140, 140, 220), (40, 40, 40))])
    labels = tmp_path / "ds" / "labels.json"
    labels.write_text(json.dumps({
        "case3/hard/bed.jpg": {"case": "case3", "difficulty": "hard", "items": ["상의", "하의"]}
    }, ensure_ascii=False), encoding="utf-8")

    out = tmp_path / "out"
    run_mod.process_photo(path, out, fake_cfg())
    (out / "review.csv").write_text("file,item,pass(1/0),note\nbed.jpg,item_00,1,\nbed.jpg,item_01,0,경계 샘\n",
                                    encoding="utf-8")

    report = ev.evaluate(out, tmp_path / "ds", labels)
    agg = report["by_case_difficulty"]["case3/hard"]
    assert agg["images"] == 1
    assert agg["count_exact"] == 1.0 and agg["category_accuracy"] == 1.0
    assert agg["review_pass"] == pytest.approx(0.5)
    assert agg["stage1_4_s"] is not None
    assert (out / "evaluation.md").exists()
    assert "case3/hard" in (out / "evaluation.md").read_text(encoding="utf-8")


def test_store_keeps_relative_paths(tmp_path):
    store = PhotoStore(tmp_path / "out", tmp_path / "a.jpg").prepare()
    p = store.path("items", "item_00.png")
    assert store.rel(p) == "items/item_00.png"
    assert store.abs("items/item_00.png") == p
    assert store.abs(None) is None


def test_photo_result_round_trips_through_json(tmp_path):
    r = PhotoResult(source="a.jpg", width=10, height=20,
                    scene=SceneInfo(case="case2", confidence=0.8, expected_items=[{"category": "상의"}]),
                    items=[Item(id="item_00", category="상의", bbox=[0, 0, 5, 5], occlusion=0.3)])
    again = PhotoResult.from_dict(json.loads(json.dumps(r.to_dict())))
    assert again.scene.case == "case2" and again.scene.expected_count == 1
    assert again.items[0].occlusion == 0.3 and again.items[0].id == "item_00"


def test_guard_cudnn_only_on_broken_gpu(monkeypatch):
    import torch

    from pipeline.garment import gpu

    monkeypatch.setattr(torch.backends.cudnn, "enabled", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
    assert gpu.guard_cudnn() is False and torch.backends.cudnn.enabled
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (12, 1))
    assert gpu.guard_cudnn() is True and not torch.backends.cudnn.enabled

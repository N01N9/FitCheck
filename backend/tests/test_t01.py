"""T01 파이프라인 점검: GPU·모델 없이 border 방식으로 끝까지 돌려 본다."""

import json

from PIL import Image, ImageDraw

from phase0 import t01_bg_removal as t01


def make_garment(path, box):
    img = Image.new("RGB", (200, 160), (245, 245, 240))  # 밝은 바닥
    ImageDraw.Draw(img).rectangle(box, fill=(30, 60, 140))  # 네이비 옷
    img.save(path)
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return mask


def test_border_pipeline_end_to_end(tmp_path):
    images, masks = tmp_path / "imgs", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    for i, box in enumerate([(40, 30, 150, 130), (20, 20, 100, 140), (60, 50, 180, 120)]):
        make_garment(images / f"g{i}.jpg", box).save(masks / f"g{i}.png")

    out = tmp_path / "out"
    data = t01.run(t01.parse_args(["--images", str(images), "--out", str(out), "--model", "border", "--masks", str(masks), "--warmup", "1"]))

    assert data["images"] == 3
    assert data["mean_iou"] > 0.95  # 단순한 배경이면 고전 방식도 잘 된다
    assert data["speed"]["count"] == 2  # warmup 1장 제외
    for name in ("report.json", "report.md", "contact_sheet.jpg", "review.csv"):
        assert (out / name).exists()
    assert len(list((out / "cutouts").glob("*.png"))) == 3
    cutout = Image.open(out / "cutouts" / "g0.png")
    assert cutout.mode == "RGBA" and cutout.getpixel((5, 5))[3] == 0  # 배경은 투명
    assert json.loads((out / "report.json").read_text(encoding="utf-8"))["model"] == "border"


def test_review_csv_is_not_overwritten(tmp_path):
    images = tmp_path / "imgs"
    images.mkdir()
    make_garment(images / "a.jpg", (40, 30, 150, 130))
    out = tmp_path / "out"
    args = t01.parse_args(["--images", str(images), "--out", str(out), "--model", "border"])
    t01.run(args)
    (out / "review.csv").write_text("file,pass(1/0),note\na.jpg,1,good\n", encoding="utf-8")
    t01.run(args)
    assert "good" in (out / "review.csv").read_text(encoding="utf-8")


def test_iou_perfect_and_disjoint():
    a = Image.new("L", (10, 10), 0)
    ImageDraw.Draw(a).rectangle((0, 0, 4, 9), fill=255)
    b = Image.new("L", (10, 10), 0)
    ImageDraw.Draw(b).rectangle((5, 0, 9, 9), fill=255)
    assert t01.iou_and_mae(a, a)[0] == 1.0
    assert t01.iou_and_mae(a, b)[0] == 0.0

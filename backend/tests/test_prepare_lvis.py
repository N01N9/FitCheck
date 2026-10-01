import json

from phase0.prepare_lvis import ALLOWED, prepare


def test_prepare_filters_license_and_clothing(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    data = {
        "info": {},
        "licenses": [{"id": 1, "name": "NC", "url": "nc"}, {"id": 4, "name": "Attribution License", "url": "by"}],
        "categories": [{"id": 1, "name": "jacket"}, {"id": 2, "name": "bagel"}, {"id": 3, "name": "shoe"}],
        "images": [
            {"id": 10, "license": 4, "coco_url": "http://x/train2017/a.jpg",
             "neg_category_ids": [2, 3], "not_exhaustive_category_ids": [1]},
            {"id": 11, "license": 1, "coco_url": "http://x/train2017/b.jpg"},
            {"id": 12, "license": 4, "coco_url": "http://x/train2017/c.jpg"},  # 옷 없음
        ],
        "annotations": [
            {"id": 1, "image_id": 10, "category_id": 1},
            {"id": 2, "image_id": 10, "category_id": 2},
            {"id": 3, "image_id": 11, "category_id": 1},
            {"id": 4, "image_id": 12, "category_id": 2},
        ],
    }
    (src / "lvis_v1_train.json").write_text(json.dumps(data))

    report = prepare(src, out, ALLOWED, download=False)

    assert report["train"]["images"] == 1 and report["train"]["annotations"] == 1
    assert report["train"]["per_coarse"] == {"아우터": 1}
    coco = json.loads((out / "annotations_train.json").read_text())
    im = coco["images"][0]
    assert im["file_name"] == "a.jpg"
    assert im["neg_category_ids"] == [3] and im["not_exhaustive_category_ids"] == [1]
    assert {c["name"] for c in coco["categories"]} == {"jacket", "shoe"}

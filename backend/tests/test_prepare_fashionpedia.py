import csv
import json
import zipfile

from phase0.prepare_fashionpedia import prepare


def test_prepare_keeps_only_commercial_images_and_garment_classes(tmp_path):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    data = {
        "info": {},
        "licenses": [{"id": 0, "name": "Attribution License", "url": "by"},
                     {"id": 3, "name": "Attribution-NonCommercial-ShareAlike License", "url": "by-nc-sa"}],
        "categories": [{"id": 4, "name": "jacket", "supercategory": "upperbody"},
                       {"id": 31, "name": "sleeve", "supercategory": "garment parts"}],
        "images": [{"id": 1, "file_name": "a.jpg", "license": 0, "original_url": "http://x/a"},
                   {"id": 2, "file_name": "b.jpg", "license": 3, "original_url": "http://x/b"}],
        "annotations": [{"id": 10, "image_id": 1, "category_id": 4},
                        {"id": 11, "image_id": 1, "category_id": 31},
                        {"id": 12, "image_id": 2, "category_id": 4}],
    }
    (src / "instances_attributes_train2020.json").write_text(json.dumps(data))
    with zipfile.ZipFile(src / "train2020.zip", "w") as zf:
        zf.writestr("train/a.jpg", b"A")
        zf.writestr("train/b.jpg", b"B")

    report = prepare(src, out, allowed={0})

    assert report["train"]["images"] == 1 and report["train"]["annotations"] == 1
    assert report["train"]["per_coarse"] == {"아우터": 1}
    assert (out / "images" / "a.jpg").exists() and not (out / "images" / "b.jpg").exists()
    coco = json.loads((out / "annotations_train.json").read_text())
    assert [c["coarse"] for c in coco["categories"]] == ["아우터"]
    rows = list(csv.DictReader((out / "sources.csv").open(encoding="utf-8")))
    assert rows == [{"file": "a.jpg", "split": "train", "fashionpedia_id": "1",
                     "original_url": "http://x/a", "license": "Attribution License", "license_url": "by"}]

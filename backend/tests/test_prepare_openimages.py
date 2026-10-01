import csv
import zipfile

from phase0.prepare_openimages import extract_masks, filter_rows


def test_filter_rows_keeps_clothing_classes(tmp_path):
    seg = tmp_path / "seg.csv"
    with seg.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["MaskPath", "ImageID", "LabelName", "BoxID"])
        w.writerow(["a_m01n4qj_1.png", "a", "/m/01n4qj", "1"])  # Shirt
        w.writerow(["a_m0fbw6_2.png", "a", "/m/0fbw6", "2"])    # Cabbage
    rows = filter_rows(seg)
    assert len(rows) == 1 and rows[0]["coarse"] == "상의" and rows[0]["label"] == "Shirt"


def test_extract_masks_only_requested(tmp_path):
    z = tmp_path / "m.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("a1.png", b"1")
        zf.writestr("a2.png", b"2")
    assert extract_masks(z, {"a1.png"}, tmp_path / "out") == 1
    assert (tmp_path / "out" / "a1.png").exists() and not (tmp_path / "out" / "a2.png").exists()

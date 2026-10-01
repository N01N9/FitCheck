import csv
import io

from PIL import Image

from phase0.collect_met import collect


def jpeg(size):
    buf = io.BytesIO()
    Image.new("RGB", size, "white").save(buf, "JPEG")
    return buf.getvalue()


class FakeMet:
    def object_ids(self):
        return [1, 2, 3]

    def obj(self, oid):
        return {
            1: {"isPublicDomain": True, "primaryImage": "p1", "additionalImages": ["a1", "a2"],
                "objectName": "Dress", "title": "Evening dress", "objectURL": "u1"},
            2: {"isPublicDomain": False, "primaryImage": "p2"},
            3: {"isPublicDomain": True, "primaryImage": ""},
        }[oid]

    def download(self, url):
        return jpeg((3000, 2000))


def test_collect_keeps_public_domain_views_and_shrinks(tmp_path):
    stats = collect(FakeMet(), tmp_path, max_objects=10, extra_views=1)
    assert stats["objects_kept"] == 1 and stats["images"] == 2
    assert stats["reject"] == {"not_public_domain": 1, "no_image": 1}
    rows = list(csv.DictReader((tmp_path / "sources.csv").open(encoding="utf-8")))
    assert [r["file"] for r in rows] == ["met1_0.jpg", "met1_1.jpg"]
    assert rows[0]["license"] == "CC0" and rows[0]["seed"] == "met:Dress|Evening dress"
    assert max(Image.open(tmp_path / "raw" / "met1_0.jpg").size) == 1600

    # 다시 돌리면 받은 소장품은 건너뛴다
    again = collect(FakeMet(), tmp_path, max_objects=10, extra_views=1)
    assert again["images"] == 0


def test_collect_uses_given_ids(tmp_path):
    class NoSearch(FakeMet):
        def object_ids(self):
            raise AssertionError("ids 를 주면 검색하지 않는다")

    stats = collect(NoSearch(), tmp_path, max_objects=10, extra_views=0, ids=[1])
    assert stats["objects_kept"] == 1 and stats["images"] == 1

import csv

from phase0.collect_smithsonian import cc0_media, collect


def row(rid, access="CC0", n=2):
    media = [{"type": "Images", "usage": {"access": access}, "content": f"https://ids/{rid}_{i}"} for i in range(n)]
    return {"id": rid, "title": f"Dress {rid}", "content": {"descriptiveNonRepeating": {
        "record_link": f"https://si/{rid}", "unit_code": "CHNDM", "data_source": "Cooper Hewitt",
        "online_media": {"media": media}}}}


def test_cc0_media_limits_views_and_license():
    assert cc0_media(row("a", n=5), views=3) == ["https://ids/a_0", "https://ids/a_1", "https://ids/a_2"]
    assert cc0_media(row("b", access="Usage conditions apply"), views=3) == []


class FakeApi:
    def search(self, query):
        yield row("a")
        yield row("b", access="Usage conditions apply")
        yield row("a")  # 다른 검색어에서 같은 소장품

    def download(self, url):
        return b"\xff\xd8" + url.encode()


def test_collect_keeps_cc0_views_once(tmp_path):
    stats = collect(FakeApi(), tmp_path, ["dress"], max_objects=10, views=2)
    assert stats["objects_kept"] == 1 and stats["images"] == 2
    assert stats["reject"] == {"not_cc0": 1}
    rows = list(csv.DictReader((tmp_path / "sources.csv").open(encoding="utf-8")))
    assert [r["title"] for r in rows] == ["si:a:0", "si:a:1"]
    assert rows[0]["license"] == "CC0" and rows[0]["seed"] == "smithsonian:dress|Dress a"

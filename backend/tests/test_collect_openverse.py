import csv

from phase0.collect_openverse import collect, license_name


def test_license_name():
    assert license_name({"license": "by-sa", "license_version": "2.0"}) == "CC BY-SA 2.0"
    assert license_name({"license": "cc0"}) == "CC0"
    assert license_name({"license": "pdm"}) == "Public domain"


class FakeApi:
    def search(self, query):
        yield {"id": "a", "url": "u/a", "width": 800, "height": 1200, "license": "by", "license_version": "2.0",
               "foreign_landing_url": "https://flickr/a", "creator": "Lee", "source": "flickr", "title": "A"}
        yield {"id": "b", "url": "u/b", "width": 300, "height": 400, "license": "by"}  # 작음
        yield {"id": "c", "url": "u/c", "width": 800, "height": 1200, "license": "cc0"}  # a 와 같은 파일

    def download(self, url):
        return b"same-bytes"


def test_collect_dedups_and_records(tmp_path):
    stats = collect(FakeApi(), tmp_path, ["street style", "ootd"], max_files=10)
    assert stats["kept"] == 1
    assert stats["reject"] == {"small": 1, "duplicate": 1}
    rows = list(csv.DictReader((tmp_path / "sources.csv").open(encoding="utf-8")))
    assert rows[0]["license"] == "CC BY 2.0" and rows[0]["artist"] == "Lee"
    assert rows[0]["seed"] == "openverse:street style"
    assert (tmp_path / "raw" / rows[0]["file"]).exists()

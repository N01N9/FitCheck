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


class SourceApi(FakeApi):
    def __init__(self):
        self.calls = []

    def search(self, query, source=None):
        self.calls.append((query, source))
        yield {"id": f"{source}-{query}", "url": "u", "width": 800, "height": 800, "license": "cc0"}

    def download(self, url):
        return url.encode() + str(len(self.calls)).encode()


def test_collect_with_sources_records_source_in_seed(tmp_path):
    api = SourceApi()
    stats = collect(api, tmp_path, ["dress"], max_files=10, sources=["smk", "rijksmuseum"])
    assert api.calls == [("dress", "smk"), ("dress", "rijksmuseum")]
    rows = list(csv.DictReader((tmp_path / "sources.csv").open(encoding="utf-8")))
    assert [r["seed"] for r in rows] == ["openverse:smk:dress", "openverse:rijksmuseum:dress"]
    assert stats["kept"] == 2

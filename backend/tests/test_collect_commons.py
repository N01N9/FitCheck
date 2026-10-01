import csv

from phase0.collect_commons import Filters, collect, license_ok, passes


def test_license_ok():
    for ok in ["CC BY-SA 4.0", "CC BY 2.0", "CC0", "Public domain", "CC-BY-SA-3.0"]:
        assert license_ok(ok), ok
    for bad in ["CC BY-NC 2.0", "CC BY-NC-SA 4.0", "CC BY-ND 4.0", "GFDL", "", "Copyrighted"]:
        assert not license_ok(bad), bad


def _meta(lic="CC BY 2.0", date="2019-05-01"):
    return {"LicenseShortName": {"value": lic}, "DateTimeOriginal": {"value": date}}


def test_passes_filters():
    f = Filters(min_short_side=700, min_aspect=1.1, min_year=2000)
    good = {"mime": "image/jpeg", "width": 1000, "height": 1500}
    assert passes(good, _meta(), f) == (True, "ok")
    assert passes({**good, "width": 1500, "height": 1000}, _meta(), f)[1] == "aspect"
    assert passes({**good, "width": 500, "height": 900}, _meta(), f)[1] == "small"
    assert passes(good, _meta(lic="CC BY-NC 2.0"), f)[1] == "license"
    assert passes(good, _meta(date="1958"), f)[1] == "old"


class FakeApi:
    def __init__(self):
        self.downloads = []

    def category_files(self, name, depth, seen):
        yield from ["File:a.jpg", "File:b.jpg", "File:c.jpg"]

    def search_files(self, query):
        yield "File:a.jpg"  # 카테고리와 중복

    def file_infos(self, titles, thumb_width):
        lic = {"File:a.jpg": "CC BY 2.0", "File:b.jpg": "CC BY-NC 2.0", "File:c.jpg": "CC0"}
        return [
            {"title": t, "imageinfo": [{
                "mime": "image/jpeg", "width": 1000, "height": 1600, "sha1": f"{i:040x}",
                "url": f"https://x/{t}", "thumburl": f"https://x/thumb/{t}",
                "descriptionurl": f"https://commons/{t}",
                "extmetadata": {**_meta(lic[t]), "Artist": {"value": "<a>Kim</a>"}},
            }]}
            for i, t in enumerate(titles)
        ]

    def download(self, url, dest):
        self.downloads.append(url)
        dest.write_bytes(b"jpg")
        return True


def test_collect_keeps_only_allowed_and_records_sources(tmp_path):
    api = FakeApi()
    seeds = [("category", "Category:X@1"), ("query", "street")]
    stats = collect(api, tmp_path, seeds, max_files=10, filters=Filters())
    assert stats["kept"] == 2 and stats["reject"] == {"license": 1}
    rows = list(csv.DictReader((tmp_path / "sources.csv").open(encoding="utf-8")))
    assert {r["title"] for r in rows} == {"File:a.jpg", "File:c.jpg"}
    assert rows[0]["artist"] == "Kim" and rows[0]["seed"] == "Category:X"
    assert all((tmp_path / "raw" / r["file"]).exists() for r in rows)

    # 다시 돌리면 이미 받은 파일은 건너뛴다
    again = collect(FakeApi(), tmp_path, seeds, max_files=10, filters=Filters())
    assert again["seen"] == 1  # b.jpg 만 다시 검사

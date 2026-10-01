import csv
import io

from PIL import Image

from phase0.collect_europeana import collect, rights_name, rights_ok


def test_rights():
    assert rights_ok("http://creativecommons.org/publicdomain/zero/1.0/")
    assert rights_ok("http://creativecommons.org/licenses/by-sa/4.0/")
    assert not rights_ok("http://creativecommons.org/licenses/by-nc-sa/4.0/")
    assert not rights_ok("http://rightsstatements.org/vocab/InC/1.0/")
    assert rights_name("http://creativecommons.org/licenses/by/4.0/") == "CC BY 4.0"
    assert rights_name("http://creativecommons.org/publicdomain/mark/1.0/") == "Public domain"


def jpeg(size, color="white"):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


class FakeApi:
    def search(self, query, theme):
        cc0 = ["http://creativecommons.org/publicdomain/zero/1.0/"]
        yield {"id": "/1/a", "rights": cc0, "edmIsShownBy": ["big"], "title": ["Dress"],
               "dataProvider": ["Museum"], "guid": "g1"}
        yield {"id": "/1/b", "rights": ["http://creativecommons.org/licenses/by-nc/4.0/"],
               "edmIsShownBy": ["big2"]}
        yield {"id": "/1/c", "rights": cc0, "edmIsShownBy": ["small"]}
        yield {"id": "/1/d", "rights": cc0, "edmIsShownBy": []}

    def download(self, url):
        return jpeg((100, 100)) if url == "small" else jpeg((800, 1000))


def test_collect_filters_rights_size_and_records(tmp_path):
    stats = collect(FakeApi(), tmp_path, max_files=10)
    assert stats["kept"] == 1
    assert stats["reject"] == {"license": 1, "small": 1, "no_image": 1}
    rows = list(csv.DictReader((tmp_path / "sources.csv").open(encoding="utf-8")))
    assert rows[0]["license"] == "CC0" and rows[0]["credit"] == "Museum"
    assert rows[0]["seed"] == "europeana:fashion:Dress"


def test_collect_shrinks_large_images(tmp_path):
    class Big(FakeApi):
        def download(self, url):
            return jpeg((4000, 3000))

    collect(Big(), tmp_path, max_files=1)
    name = next((tmp_path / "raw").iterdir())
    assert max(Image.open(name).size) == 1600


def test_collect_skips_provider_whose_server_is_down(tmp_path):
    cc0 = ["http://creativecommons.org/publicdomain/zero/1.0/"]

    class Down:
        downloads = 0

        def search(self, query, theme):
            for i in range(100):
                yield {"id": f"/x/{i}", "rights": cc0, "edmIsShownBy": [f"u{i}"], "dataProvider": ["Dead Museum"]}

        def download(self, url):
            Down.downloads += 1
            return None

    stats = collect(Down(), tmp_path, max_files=10)
    assert Down.downloads == 30
    assert stats["reject"] == {"download_failed": 30, "provider_unavailable": 70}

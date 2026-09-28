"""T02 점검: 가짜 OpenAI 호환 서버를 띄워 실제 HTTP 경로로 태깅·채점을 돌려 본다."""

import csv
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from PIL import Image

from phase0 import t02_tagging as t02
from phase0.garment_schema import SCHEMA, validate

GOOD = {
    "category": "상의", "subcategory": "티셔츠", "primary_color": "화이트", "secondary_colors": [],
    "pattern": "무지", "material": "면", "fit": "레귤러", "length": "기본", "sleeve": "반팔",
    "neckline": "라운드", "season": ["봄", "여름"], "formality": 2, "style_tags": ["캐주얼"],
    "description": "흰색 면 반팔 티셔츠",
}


@pytest.fixture
def server():
    """두 번째 요청에만 JSON이 아닌 답을 주는 가짜 vLLM 서버."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            content = "죄송합니다" if len(seen) == 2 else json.dumps(GOOD, ensure_ascii=False)
            payload = json.dumps({"choices": [{"message": {"content": content}}],
                                  "usage": {"completion_tokens": 50}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/v1", seen
    httpd.shutdown()


def make_images(folder, n):
    folder.mkdir()
    for i in range(n):
        Image.new("RGB", (2000, 1500), (250, 250, 250)).save(folder / f"g{i}.jpg")


def test_tagging_scoring_and_report(tmp_path, server):
    url, seen = server
    make_images(tmp_path / "imgs", 3)
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(t02.LABEL_COLUMNS)
        w.writerow(["g0.jpg", "상의", "티셔츠", "화이트", "무지", "레귤러", "기본", "반팔", "라운드", "봄;여름", "2"])
        w.writerow(["g1.jpg", "상의", "셔츠", "", "", "", "", "", "", "", ""])  # 이 요청은 실패 응답
        w.writerow(["g2.jpg", "상의", "셔츠", "블랙", "", "", "", "", "", "여름", "4"])

    out = tmp_path / "out"
    data = t02.run(t02.parse_args(["--images", str(tmp_path / "imgs"), "--out", str(out),
                                   "--labels", str(labels), "--base-url", url, "--model", "fake"]))

    assert data["schema_valid_rate"] == round(2 / 3, 3)
    assert data["accuracy"]["category"] == round(2 / 3, 3)  # g1 실패 → 0점
    assert data["accuracy"]["subcategory"] == round(1 / 3, 3)
    assert data["accuracy"]["season"] == round((1 + 0.5) / 2, 3)
    assert data["accuracy"]["formality"] == 0.5  # g2: 2 vs 4 → 차이 2
    assert data["failures"][0]["file"] == "g1.jpg"
    assert (out / "report.md").exists() and (out / "predictions.jsonl").exists()

    body = seen[0]
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_label_template_when_no_labels(tmp_path, server):
    url, _ = server
    make_images(tmp_path / "imgs", 1)
    out = tmp_path / "out"
    data = t02.run(t02.parse_args(["--images", str(tmp_path / "imgs"), "--out", str(out), "--base-url", url]))
    assert data["accuracy"] == {}
    header = (out / "label_template.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    assert header.split(",") == t02.LABEL_COLUMNS


def test_unreachable_server_is_reported_not_crashed(tmp_path):
    make_images(tmp_path / "imgs", 1)
    data = t02.run(t02.parse_args(["--images", str(tmp_path / "imgs"), "--out", str(tmp_path / "o"),
                                   "--base-url", "http://127.0.0.1:9/v1"]))
    assert data["schema_valid_rate"] == 0.0
    assert "요청 실패" in data["failures"][0]["errors"][0]


def test_validate_catches_bad_values():
    assert validate(GOOD) == []
    bad = dict(GOOD, category="모자", formality=9, season=[], extra=1)
    errors = validate(bad)
    assert any("category" in e for e in errors)
    assert any("formality" in e for e in errors)
    assert any("season" in e for e in errors)
    assert any("extra" in e for e in errors)


def test_parse_json_handles_code_fence():
    assert t02.parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert t02.parse_json("not json") is None

"""T02. 옷 태깅 테스트.

OpenAI 호환 API(vLLM, Ollama 등)로 서빙 중인 비전 모델(VLM)에 옷 사진을 보내
garment_schema 형식의 JSON 태그를 받고, 정답 라벨과 비교해 정확도·속도를 리포트로 남긴다.

실행 예 (backend 폴더에서, vLLM 서버가 떠 있는 상태)
  python -m phase0.t02_tagging --images data/garments --labels data/garment_labels.csv \
      --base-url http://localhost:8000/v1 --model Qwen/Qwen3.6-35B-A3B --out results/t02

정답 라벨 CSV (빈 칸은 채점에서 뺀다, season은 ;로 구분)
  file,category,subcategory,primary_color,pattern,fit,length,sleeve,neckline,season,formality
  tee01.jpg,상의,티셔츠,화이트,무지,레귤러,기본,반팔,라운드,봄;여름,2
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

from phase0.common import Stopwatch, Timings, environment, list_images, load_rgb, write_report
from phase0.garment_schema import SCHEMA, prompt_text, validate

# 합격 기준 (기획서 Phase 1 목표와 같게)
TARGET_CATEGORY = 0.95
TARGET_KEY_ATTRS = 0.85
KEY_ATTRS = ["subcategory", "primary_color", "pattern", "fit", "sleeve"]
EXACT_FIELDS = ["category", "subcategory", "primary_color", "pattern", "fit", "length", "sleeve", "neckline"]
LABEL_COLUMNS = ["file", *EXACT_FIELDS, "season", "formality"]


def encode_image(path: Path, max_side: int) -> str:
    img = load_rgb(path)
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class OpenAICompatTagger:
    """vLLM·Ollama 등이 제공하는 /v1/chat/completions 로 태깅한다."""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", max_side: int = 1024,
                 timeout: float = 180, thinking: bool = False):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.max_side = model, api_key, max_side
        self.timeout, self.thinking = timeout, thinking

    def request_body(self, image_url: str) -> dict:
        return {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": "너는 패션 상품 태깅 전문가다. 지정된 JSON 형식으로만 답한다."},
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_text()},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ]},
            ],
            # 서버가 스키마에 맞는 JSON만 생성하도록 강제한다
            "response_format": {"type": "json_schema", "json_schema": {"name": "garment", "schema": SCHEMA, "strict": True}},
            # Qwen3 계열의 생각(thinking) 모드는 태깅에 불필요하고 느리다 (vLLM이 읽는 옵션)
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }

    def tag(self, path: Path) -> tuple[str, dict]:
        body = json.dumps(self.request_body(encode_image(path, self.max_side))).encode()
        req = urllib.request.Request(self.url, data=body, headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.load(resp)
        return data["choices"][0]["message"].get("content") or "", data.get("usage", {})


def parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):  # 코드블록으로 감싸 답하는 모델 대비
        text = text.strip("`").removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def load_labels(path: Path | None) -> dict[str, dict]:
    if not path:
        return {}
    with path.open(encoding="utf-8-sig") as f:
        return {row["file"]: row for row in csv.DictReader(f)}


def score(pred: dict, label: dict) -> dict:
    """필드별 점수(1/0, season은 자카드 유사도). 라벨이 빈 칸인 필드는 뺀다."""
    out = {}
    for field in EXACT_FIELDS:
        truth = (label.get(field) or "").strip()
        if truth:
            out[field] = float(pred.get(field) == truth)
    truth_season = {s.strip() for s in (label.get("season") or "").split(";") if s.strip()}
    if truth_season:
        guess = set(pred.get("season") or [])
        out["season"] = len(guess & truth_season) / len(guess | truth_season)
    if (label.get("formality") or "").strip():
        out["formality"] = float(abs(int(pred.get("formality", 0)) - int(label["formality"])) <= 1)
    return out


def write_label_template(path: Path, files: list[str]):
    with path.open("w", newline="", encoding="utf-8-sig") as f:  # 엑셀에서 한글이 깨지지 않게 BOM
        w = csv.writer(f)
        w.writerow(LABEL_COLUMNS)
        for name in files:
            w.writerow([name] + [""] * (len(LABEL_COLUMNS) - 1))


def run(args, tagger=None) -> dict:
    images = list_images(Path(args.images))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tagger = tagger or OpenAICompatTagger(args.base_url, args.model, args.api_key, args.max_side,
                                          thinking=args.thinking)
    labels = load_labels(Path(args.labels) if args.labels else None)

    timings = Timings(warmup=min(args.warmup, max(len(images) - 1, 0)))
    rows, field_scores, out_tokens, gen_seconds = [], {}, 0, 0.0
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as pred_file:
        for path in images:
            row = {"file": path.name}
            try:
                with Stopwatch() as sw:
                    text, usage = tagger.tag(path)
                timings.add(sw.seconds)
                row["seconds"] = round(sw.seconds, 2)
                pred = parse_json(text)
                row["valid_json"] = pred is not None
                row["schema_errors"] = validate(pred) if pred is not None else ["JSON 아님"]
                row["prediction"] = pred
                if not pred:
                    row["raw"] = text[:500]
                tokens = usage.get("completion_tokens") or 0
                out_tokens += tokens
                gen_seconds += sw.seconds if tokens else 0
            except (urllib.error.URLError, TimeoutError, KeyError) as e:
                row.update(valid_json=False, schema_errors=[f"요청 실패: {e}"], prediction=None)
            label = labels.get(path.name)
            if label and row.get("prediction"):
                row["scores"] = score(row["prediction"], label)
                for k, v in row["scores"].items():
                    field_scores.setdefault(k, []).append(v)
            elif label:  # 실패한 예측은 라벨 있는 필드 전부 0점
                for k in score({}, label):
                    field_scores.setdefault(k, []).append(0.0)
            rows.append(row)
            pred_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not labels:
        template = out / "label_template.csv"
        write_label_template(template, [p.name for p in images])

    accuracy = {k: round(sum(v) / len(v), 3) for k, v in field_scores.items()}
    key = [accuracy[k] for k in KEY_ATTRS if k in accuracy]
    data = {
        "test": "T02 옷 태깅",
        "model": getattr(tagger, "model", None),
        "environment": environment(),
        "images": len(images),
        "labeled": sum(1 for r in rows if r["file"] in labels),
        "schema_valid_rate": round(sum(1 for r in rows if not r["schema_errors"]) / len(rows), 3),
        "speed": timings.summary(),
        "output_tokens_per_s": round(out_tokens / gen_seconds, 1) if gen_seconds else None,
        "accuracy": accuracy,
        "key_attr_mean": round(sum(key) / len(key), 3) if key else None,
        "targets": {"category": TARGET_CATEGORY, "key_attr_mean": TARGET_KEY_ATTRS},
        "failures": [{"file": r["file"], "errors": r["schema_errors"]} for r in rows if r["schema_errors"]],
    }
    write_report(out, "T02 옷 태깅 결과", data, markdown(data))
    return data


def markdown(d: dict) -> str:
    s = d["speed"]
    lines = [
        f"- 모델: `{d['model']}`",
        f"- 환경: {d['environment']}",
        f"- 이미지: {d['images']}장 (정답 라벨 {d['labeled']}장)",
        f"- 스키마 통과율: {d['schema_valid_rate']}",
        f"- 속도: 중앙값 {s.get('p50_s')}초/장, 95% {s.get('p95_s')}초/장, 출력 {d['output_tokens_per_s']} 토큰/초",
    ]
    if d["accuracy"]:
        lines += ["", "## 정확도", "| 항목 | 정확도 |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in d["accuracy"].items()]
        lines += ["", f"- 카테고리 목표 ≥ {TARGET_CATEGORY}, 주요 속성({', '.join(KEY_ATTRS)}) 평균 {d['key_attr_mean']} (목표 ≥ {TARGET_KEY_ATTRS})"]
    else:
        lines += ["", "정답 라벨이 없어 정확도는 계산하지 않았다. `label_template.csv`를 채워 `--labels`로 다시 실행하면 된다."]
    if d["failures"]:
        lines += ["", f"## 실패 {len(d['failures'])}건 (자세한 내용은 predictions.jsonl)"]
        lines += [f"- {f['file']}: {'; '.join(f['errors'][:3])}" for f in d["failures"][:10]]
    return "\n".join(lines)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="T02 옷 태깅 테스트")
    p.add_argument("--images", required=True, help="옷 사진 폴더 (T01 cutouts 폴더도 가능)")
    p.add_argument("--out", required=True)
    p.add_argument("--labels", help="정답 라벨 CSV (없으면 템플릿을 만들어 준다)")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--max-side", type=int, default=1024, help="보내기 전 이미지 긴 변 크기")
    p.add_argument("--thinking", action="store_true", help="생각 모드 켜기 (기본 끔)")
    p.add_argument("--warmup", type=int, default=1)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args)
    print((Path(args.out) / "report.md").read_text(encoding="utf-8"))

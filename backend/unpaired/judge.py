"""엄격한 눈 채점기: 사진 속 대상 옷과 만든 상품 사진이 정말 같은 옷인지 VLM 에게 항목별로 묻는다.

score.py 의 색·팔레트 수치는 사진 조명 때문에 "같은 빨강인데 더 쨍해짐", "없던 허리띠" 같은 틀린 결과를 걸러 내지
못했다(사람 눈 판정 84건 대비 정밀도 0.66). 그래서 Qwen2.5-VL-7B(QIE-2511 의 글 인코더, Apache-2.0, 이미 받아 둔
가중치)에게 질문 하나에 한 가지씩 Yes/No 로 묻고, 첫 토큰의 Yes·No 로짓 차이를 점수로 쓴다. 문턱값은 사람 판정
(results/unpaired/review/labels_*.json)에 맞춘다.

  type     같은 종류·목선·소매 길이인가 (티 ↔ 캐미솔·민소매·터틀넥 혼동)
  color    같은 색인가 (색상·밝기·채도)
  pattern  같은 무늬·그림·글자인가 (민무늬는 민무늬로)
  added    대상 옷에 없는 것이 생겼나 (넥타이·벨트·목걸이·겉옷 무늬·장식) → Yes 가 나쁨

사용 (컨테이너 안)
  python -m unpaired.judge --run results/unpaired/eval_report/lora_v2_distilled --variant lora
  python -m unpaired.judge --run ... --labels results/unpaired/review/labels_v2d.json   # 사람 판정과 맞춰 보기
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

QIE_ID = "Qwen/Qwen-Image-Edit-2511"
INTRO = ("Image 1 is a photo of a person. One garment in it is highlighted: it is shown bright and everything else is "
         "darkened. Image 2 is a product photo that is supposed to show exactly that highlighted garment on its own. ")
QUESTIONS = {
    "type": "Is the garment in Image 2 the same kind of garment as the highlighted one, with the same neckline and the "
            "same sleeve length wherever these are visible in Image 1 (for example a camisole, tank top or turtleneck "
            "must not become an ordinary t-shirt)? Answer Yes or No.",
    "color": "Is the main color of the garment in Image 2 the same as the highlighted garment in Image 1: the same hue, "
             "about as light or dark, and about as vivid or muted? Answer Yes or No.",
    "pattern": "Does the garment in Image 2 have the same pattern, print, graphic and text as the highlighted garment "
               "(a plain garment must stay plain, and any text must read the same)? Answer Yes or No.",
    "added": "Does Image 2 show anything that is not part of the highlighted garment itself, such as a tie, belt, "
             "necklace, scarf, brooch, bag strap, the pattern of a jacket worn over it, or decorations or colored "
             "lines that the highlighted garment does not have? Answer Yes or No.",
}
BAD_IF_YES = {"added"}


class ApiJudge:
    """OpenAI 호환 서버(sglang 의 qwen3.8-27b 등)에 묻는다. 첫 토큰 logprob 에서 Yes - No 를 잰다."""

    def __init__(self, url: str = "http://127.0.0.1:30000/v1", model: str = "qwen3.8-27b"):
        self.url, self.model = url.rstrip("/"), model

    @staticmethod
    def _b64(img: Image.Image) -> str:
        import base64
        import io
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def ask(self, crop: Image.Image, result: Image.Image, question: str) -> float:
        import math
        import urllib.request
        body = {"model": self.model, "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
                "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": self._b64(crop)}},
                    {"type": "image_url", "image_url": {"url": self._b64(result)}},
                    {"type": "text", "text": INTRO + question}]}]}
        import os
        headers = {"Content-Type": "application/json"}
        if os.environ.get("JUDGE_API_KEY"):  # 키는 코드·로그에 남기지 않고 환경 변수로만 받는다
            headers["Authorization"] = "Bearer " + os.environ["JUDGE_API_KEY"]
        req = urllib.request.Request(self.url + "/chat/completions", json.dumps(body).encode(), headers)
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read())
        top = out["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        def mass(words):
            ps = [math.exp(t["logprob"]) for t in top if t["token"].strip().lower() in words]
            return math.log(sum(ps)) if ps else -30.0
        return mass({"yes"}) - mass({"no"})

    def __call__(self, crop: Image.Image, result: Image.Image) -> dict[str, float]:
        out = {}
        for k, q in QUESTIONS.items():
            s = self.ask(crop, result, q)
            out[k] = -s if k in BAD_IF_YES else s
        return out


REASON_PROMPT = INTRO + (
    "Judge strictly whether Image 2 is a faithful product photo of the highlighted garment. First describe the "
    "highlighted garment in Image 1: garment type, neckline, sleeve length (if visible), main color (hue, lightness, "
    "vividness), pattern or print, any text. Note which parts are hidden by other clothing or accessories. Then describe "
    "Image 2 the same way. Then compare. It FAILS if any of these hold: a different garment type, neckline or visible "
    "sleeve length; a clearly different color (including noticeably more vivid, duller, lighter or darker); a different "
    "pattern, print or text; anything added that the highlighted garment does not have (tie, belt, necklace, scarf, brooch, "
    "the pattern of a jacket worn over it, colored trims, labels or lines). Parts hidden in Image 1 may be filled in "
    "plainly and plausibly. Finish with one line of JSON only: "
    '{"type": "ok"|"fail", "color": "ok"|"fail", "pattern": "ok"|"fail", "added": "ok"|"fail", "verdict": "pass"|"fail"}')


def reason_call(url: str, model: str, crop: Image.Image, result: Image.Image, max_tokens: int = 6000) -> dict:
    """생각 모드로 묘사 → 비교 → JSON 판정. 마지막 JSON 줄을 읽는다."""
    import os
    import re
    import urllib.request
    body = {"model": model, "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": True},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": ApiJudge._b64(crop)}},
                {"type": "image_url", "image_url": {"url": ApiJudge._b64(result)}},
                {"type": "text", "text": REASON_PROMPT}]}]}
    headers = {"Content-Type": "application/json"}
    if os.environ.get("JUDGE_API_KEY"):
        headers["Authorization"] = "Bearer " + os.environ["JUDGE_API_KEY"]
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=1800) as r:
        msg = json.loads(r.read())["choices"][0]["message"]
    text = msg.get("content") or ""
    found = re.findall(r"\{[^{}]*\"verdict\"[^{}]*\}", text)
    try:
        v = json.loads(found[-1])
    except (IndexError, json.JSONDecodeError):
        v = {"verdict": "fail", "parse_error": True}
    v["text"] = text[-1500:]
    return v


class Judge:
    """model_id 가 QIE 면 그 안의 Qwen2.5-VL 글 인코더를, 아니면 그 VLM 자체를 쓴다."""

    def __init__(self, model_id: str = QIE_ID, device: str = "cuda"):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        if model_id == QIE_ID:
            from transformers import Qwen2_5_VLForConditionalGeneration
            root = Path(snapshot_download(QIE_ID, allow_patterns=["text_encoder/*", "processor/*"]))
            self.proc = AutoProcessor.from_pretrained(root / "processor")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                root / "text_encoder", dtype=torch.bfloat16, device_map=device).eval()
        else:
            root = Path(snapshot_download(model_id))
            self.proc = AutoProcessor.from_pretrained(root)
            self.model = AutoModelForImageTextToText.from_pretrained(root, dtype=torch.bfloat16, device_map=device).eval()
        tok = self.proc.tokenizer
        self.yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")})
        self.no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")})

    def ask(self, crop: Image.Image, result: Image.Image, question: str) -> float:
        """Yes 쪽 로짓 - No 쪽 로짓 (양수면 Yes)."""
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": INTRO + question}]}]
        try:  # Qwen3.x 는 생각 모드를 꺼야 첫 토큰이 답이다
            text = self.proc.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = self.proc.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[crop, result], return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits[0, -1].float()
        return float(logits[self.yes].logsumexp(0) - logits[self.no].logsumexp(0))

    def __call__(self, crop: Image.Image, result: Image.Image) -> dict[str, float]:
        """항목별 '좋음' 점수(양수가 좋음). added 는 부호를 뒤집는다."""
        out = {}
        for k, q in QUESTIONS.items():
            s = self.ask(crop, result, q)
            out[k] = -s if k in BAD_IF_YES else s
        return out


def fit(img: Image.Image, side: int = 448) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((side, side), Image.LANCZOS)
    return img


def passes(scores: dict[str, float], thr: dict[str, float] | None = None) -> bool:
    thr = thr or {}
    return all(scores[k] > thr.get(k, 0.0) for k in QUESTIONS)


def calibrate(rows: list[dict], labels: dict[str, bool]) -> dict:
    """사람 판정과의 일치. 각 항목 문턱값은 사람이 OK 한 것을 거의 떨어뜨리지 않는 선에서 가장 엄하게 고르지 않고,
    그냥 0 과 몇 개 후보에서 정밀도·재현율을 보여 준다(문턱값 고르기는 따로 둔 분할에서)."""
    out = {}
    for t in (-2.0, -1.0, 0.0, 1.0, 2.0):
        thr = {k: t for k in QUESTIONS}
        pred = {r["file"]: passes(r["judge"], thr) for r in rows}
        tp = sum(pred[f] and labels[f] for f in pred)
        fp = sum(pred[f] and not labels[f] for f in pred)
        fn = sum((not pred[f]) and labels[f] for f in pred)
        out[str(t)] = {"pass": tp + fp, "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3)}
    # 항목별로 사람 판정 실패를 얼마나 잡는지
    per = {}
    for k in QUESTIONS:
        bad = [r["judge"][k] for r in rows if not labels[r["file"]]]
        good = [r["judge"][k] for r in rows if labels[r["file"]]]
        per[k] = {"good_median": round(float(np.median(good)), 2), "bad_median": round(float(np.median(bad)), 2)}
    out["per_question"] = per
    return out


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="엄격한 VLM 채점")
    p.add_argument("--run", required=True, help="zeroshot 실행 폴더 (cases.json, <variant>/)")
    p.add_argument("--variant", default="lora")
    p.add_argument("--k", type=int, default=1, help="사례마다 채점할 장 수")
    p.add_argument("--fashionpedia", default="data/fashionpedia")
    p.add_argument("--labels", help="사람 판정 json: 일치도를 출력")
    p.add_argument("--limit", type=int)
    p.add_argument("--model", default=QIE_ID)
    p.add_argument("--api", help="OpenAI 호환 서버 주소 (예: http://127.0.0.1:30000/v1). 주면 --model 은 서버의 모델 이름")
    p.add_argument("--tag", default="qwen25vl7b")
    p.add_argument("--reason", action="store_true", help="--api 와 함께: 생각 모드로 묘사·비교 후 JSON 판정")
    p.add_argument("--workers", type=int, default=6)
    args = p.parse_args(argv)

    from unpaired.layered import Annotations
    from unpaired.pointer import dim_crop

    run = Path(args.run)
    cases = json.loads((run / "cases.json").read_text())[: args.limit]
    ann = Annotations(Path(args.fashionpedia))
    if args.reason:
        reason_main(args, run, cases, ann, dim_crop)
        return
    judge = ApiJudge(args.api, args.model) if args.api else Judge(args.model)
    rows = []
    out_path = run / f"judge_{args.tag}_{args.variant}.jsonl"
    with out_path.open("w") as fh:
        for c in cases:
            photo = np.array(Image.open(ann.image_dir / c["file"]).convert("RGB"))
            crop = Image.fromarray(dim_crop(photo, ann.mask(c["file"], c["inner"]["ann_id"]), factor=0.25))
            stem = c["file"].rsplit(".", 1)[0]
            for k in range(args.k):
                res = run / args.variant / f"{stem}_k{k}.png"
                if not res.exists():
                    continue
                s = judge(fit(crop), fit(Image.open(res)))
                r = {"file": c["file"], "k": k, "judge": {a: round(b, 3) for a, b in s.items()}, "pass": passes(s)}
                rows.append(r)
                fh.write(json.dumps(r) + "\n")
                fh.flush()
                print(c["file"], k, r["judge"], flush=True)
    first = [r for r in rows if r["k"] == 0]
    print(json.dumps({"n": len(first), "pass": round(float(np.mean([r["pass"] for r in first])), 3)}))
    if args.labels:
        labels = {x["file"]: x["ok"] for x in json.loads(Path(args.labels).read_text())["labels"]}
        print(json.dumps(calibrate([r for r in first if r["file"] in labels], labels), indent=1))


def reason_main(args, run: Path, cases: list[dict], ann, dim_crop) -> None:
    from concurrent.futures import ThreadPoolExecutor

    jobs = []
    for c in cases:
        stem = c["file"].rsplit(".", 1)[0]
        for k in range(args.k):
            res = run / args.variant / f"{stem}_k{k}.png"
            if res.exists():
                jobs.append((c, k, res))

    def one(job):
        c, k, res = job
        photo = np.array(Image.open(ann.image_dir / c["file"]).convert("RGB"))
        crop = Image.fromarray(dim_crop(photo, ann.mask(c["file"], c["inner"]["ann_id"]), factor=0.25))
        v = reason_call(args.api, args.model, fit(crop, 640), fit(Image.open(res), 640))
        return {"file": c["file"], "k": k, "pass": v.get("verdict") == "pass", "verdict": v}

    out_path = run / f"judge_{args.tag}_{args.variant}.jsonl"
    rows = []
    with out_path.open("w") as fh, ThreadPoolExecutor(args.workers) as ex:
        for r in ex.map(one, jobs):
            rows.append(r)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            print(r["file"], r["k"], {a: b for a, b in r["verdict"].items() if a != "text"}, flush=True)
    first = [r for r in rows if r["k"] == 0]
    print(json.dumps({"n": len(first), "pass": round(float(np.mean([r["pass"] for r in first])), 3)}))
    if args.labels:
        labels = {x["file"]: x["ok"] for x in json.loads(Path(args.labels).read_text())["labels"]}
        f = [r for r in first if r["file"] in labels]
        tp = sum(r["pass"] and labels[r["file"]] for r in f)
        fp = sum(r["pass"] and not labels[r["file"]] for r in f)
        print(json.dumps({"pass": tp + fp, "precision": round(tp / max(tp + fp, 1), 3),
                          "recall": round(tp / max(sum(labels[r["file"]] for r in f), 1), 3)}))


if __name__ == "__main__":
    main()

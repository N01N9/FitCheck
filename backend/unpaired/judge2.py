"""근거를 모아 최종 판정하는 엄격 채점기.

근거
  color  사진 속 보이는 대상 옷과 결과물의 색 수치(gates: 팔레트 거리, 밝기 차, 채도 비, 없던 색) — score.py 결과
  text   옷의 글자를 System 2(Qwen3.8-27B 그대로)로 두 이미지에서 각각 받아 적고 문자열로 비교
  pvc    (있으면) PVC-Judge 가 같은 사진의 두 후보 중 어느 쪽이 더 일치한다고 봤는지

최종 판정 (같은 vLLM 서버: JEV-27B-VL NVFP4)
  s1     JEV System 1: 근거 + 두 이미지 → P(통과), 글 생성 없이 한 번의 전방 계산
  s2     Qwen3.8-27B 생각 모드: 근거 + 두 이미지 → 묘사·비교 뒤 JSON 판정

사용 (fitcheck-nas 컨테이너, vLLM 이 127.0.0.1:8000 에 jev / jev-decision 으로 떠 있을 때)
  python -m unpaired.judge2 --run results/unpaired/eval_report/lora_v2_distilled \\
      --labels results/unpaired/review/labels_v2d_strict.json
"""

from __future__ import annotations

import argparse
import base64
import difflib
import io
import json
import math
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

URL = "http://127.0.0.1:8000/v1"
JEV_DIR = Path("/nas/models/models/Atlas3D--JEV-27B-VL-NVFP4")
RAW_TEMPLATE = ("{%- for m in messages -%}{%- for c in m['content'] -%}{%- if c['type'] == 'text' -%}{{ c['text'] }}"
                "{%- else -%}<|vision_start|><|image_pad|><|vision_end|>{%- endif -%}{%- endfor -%}{%- endfor -%}")
CRITERIA = (
    "It is faithful only if ALL hold: same garment type, neckline and visible sleeve length; the same color (not "
    "noticeably more vivid, duller, lighter or darker); the same pattern, print and text; nothing added that the "
    "highlighted garment does not have (tie, belt, necklace, scarf, brooch, the pattern of a jacket worn over it). "
    "Colored or contrasting trims and neckbands, brand labels, side or hem tags, logos and pockets that the highlighted "
    "garment does not show are ALWAYS failures, even when they sit on a part that is hidden in the photo: a hidden part "
    "may only be filled with the same plain fabric. A plain inner neck label is the only exception.")
INTRO = ("Image 1 is a close-up of a photo where one garment is highlighted: it is shown bright and everything else is "
         "darkened. Image 2 is a product photo that is supposed to show exactly that garment on its own. ")


def b64(img: Image.Image, side: int = 640) -> str:
    img = img.convert("RGB")
    img.thumbnail((side, side), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


SERVER = {"url": URL, "model": "jev"}  # main() 에서 --url/--model 로 바꾼다 (예: 사용자 sglang 의 qwen3.8-27b)


def post(path: str, body: dict, timeout: int = 1800) -> dict:
    import os
    headers = {"Content-Type": "application/json"}
    if os.environ.get("JUDGE_API_KEY"):  # 키는 환경 변수로만 받는다
        headers["Authorization"] = "Bearer " + os.environ["JUDGE_API_KEY"]
    req = urllib.request.Request(SERVER["url"] + path, json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def chat(content: list, max_tokens: int, thinking: bool) -> str:
    out = post("/chat/completions", {"model": SERVER["model"], "max_tokens": max_tokens, "temperature": 0,
                                     "chat_template_kwargs": {"enable_thinking": thinking},
                                     "messages": [{"role": "user", "content": content}]})
    return out["choices"][0]["message"].get("content") or ""


def img_part(img: Image.Image) -> dict:
    return {"type": "image_url", "image_url": {"url": b64(img)}}


# ---- 근거: 글자 ----
TEXT_Q = ("Transcribe every piece of text, letters or numbers printed on the {what}, exactly as written, in reading "
          "order. Ignore small care or size labels. If there is no text, answer exactly: NONE. Answer with the text only.")


def read_text(img: Image.Image, what: str) -> str:
    t = chat([img_part(img), {"type": "text", "text": TEXT_Q.format(what=what)}], 200, False).strip()
    return "" if t.upper().startswith("NONE") else re.sub(r"\s+", " ", t)


def text_evidence(crop: Image.Image, result: Image.Image) -> dict:
    a = read_text(crop, "bright (highlighted) garment")
    b = read_text(result, "garment")
    norm = lambda s: re.sub(r"[^0-9a-z]", "", s.lower())
    sim = 1.0 if not a and not b else difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()
    return {"photo_text": a, "product_text": b, "similarity": round(sim, 3)}


def evidence_text(color: dict | None, text: dict, pvc: dict | None) -> str:
    lines = ["Automatic measurements (use them as hints; trust what you see when they disagree):"]
    if color:
        lines.append(f"- color: palette distance {color['palette_dist']:.1f} (CIEDE2000; under 5 is close, over 10 is "
                     f"clearly different), lightness gap {color['lightness_gap']:.1f}, chroma ratio product/photo "
                     f"{color['chroma_ratio']:.2f}, colors not in the garment {color['extra_color']:.1f}")
    if text["photo_text"] or text["product_text"]:
        lines.append(f"- text read from the photo garment: \"{text['photo_text'] or '(none)'}\"; from the product: "
                     f"\"{text['product_text'] or '(none)'}\"; similarity {text['similarity']:.2f}")
    else:
        lines.append("- no text found on either garment")
    if pvc:
        lines.append(f"- a consistency judge preferred this product photo over another candidate with probability "
                     f"{pvc['p_win']:.2f}")
    return "\n".join(lines)


# ---- 최종: JEV System 1 ----
class System1:
    def __init__(self, root: Path = JEV_DIR):
        self.dh = json.loads((root / "adapter_vllm" / "decision_head.json").read_text())
        self.temp = json.loads((root / "calibration.json").read_text())["per_kind"]

    def noul(self, parts: list, question: str) -> float:
        options = ["false", "true"]
        content = [{"type": "text", "text": "[kind] noul\n[state] "}]
        content += [p if isinstance(p, dict) else {"type": "text", "text": p} for p in parts]
        content.append({"type": "text", "text": f"\n[question] {question}\n[options]\n" + "\n".join(options) + "\n[decision]:"})
        s = self.dh["slots"]["ranges"]["noul"][0]
        ids = self.dh["verbalizer_ids"][s: s + 2]
        r = post("/chat/completions", {
            "model": "jev-decision", "messages": [{"role": "user", "content": content}], "chat_template": RAW_TEMPLATE,
            "add_generation_prompt": False, "add_special_tokens": False, "max_tokens": 1, "temperature": 1.0,
            "top_k": 0, "top_p": 1.0, "logprobs": True, "top_logprobs": 2, "allowed_token_ids": ids,
            "return_tokens_as_token_ids": True})
        lp = {int(t["token"].split(":")[1]): t["logprob"] for t in r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]}
        z = [(lp.get(t, -1e9) + self.dh["bias"][s + i]) / self.temp["noul"] for i, t in enumerate(ids)]
        e = [math.exp(x - max(z)) for x in z]
        return e[1] / sum(e)


def s1_decide(s1: System1, crop, result, ev: str) -> float:
    parts = ["Close-up of the photo, target garment bright: ", img_part(crop), "\nProduct photo: ", img_part(result),
             "\n" + ev + "\n" + CRITERIA]
    return s1.noul(parts, "Is the product photo a faithful product photo of the bright garment in the close-up?")


S2_PROMPT = INTRO + (
    "Judge strictly whether Image 2 is a faithful product photo of the highlighted garment. Describe the highlighted "
    "garment (type, neckline, sleeves if visible, main color, pattern, text; note hidden parts), then Image 2, then "
    "compare. " + CRITERIA + "\n\n{ev}\n\nFinish with one line of JSON only: "
    '{{"type": "ok"|"fail", "color": "ok"|"fail", "pattern": "ok"|"fail", "added": "ok"|"fail", "verdict": "pass"|"fail"}}')


def s2_decide(crop, result, ev: str) -> dict:
    text = chat([img_part(crop), img_part(result), {"type": "text", "text": S2_PROMPT.format(ev=ev)}], 16000, True)
    found = re.findall(r"\{[^{}]*\"verdict\"[^{}]*\}", text)
    try:
        v = json.loads(found[-1])
    except (IndexError, json.JSONDecodeError):
        v = {"verdict": "fail", "parse_error": True}
    v["text"] = text[-1200:]
    return v


def precision_recall(rows: list[dict], labels: dict[str, bool], key) -> dict:
    tp = sum(key(r) and labels[r["file"]] for r in rows)
    fp = sum(key(r) and not labels[r["file"]] for r in rows)
    pos = sum(labels[r["file"]] for r in rows)
    return {"pass": tp + fp, "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(pos, 1), 3)}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="근거 + 최종 판정 채점기")
    p.add_argument("--run", required=True)
    p.add_argument("--variant", default="lora")
    p.add_argument("--k", type=int, default=1)
    p.add_argument("--fashionpedia", default="data/fashionpedia")
    p.add_argument("--labels")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int)
    p.add_argument("--no-s2", action="store_true")
    p.add_argument("--no-s1", action="store_true", help="System 1 은 따로(judge_s1.py) 돌린다")
    p.add_argument("--url", default=URL)
    p.add_argument("--model", default="jev", help="System 2·글자 받아 적기에 쓸 서버 모델 이름")
    p.add_argument("--tag", default="j2")
    args = p.parse_args(argv)

    from unpaired.layered import Annotations
    from unpaired.pointer import dim_crop

    run = Path(args.run)
    cases = json.loads((run / "cases.json").read_text())[: args.limit]
    ann = Annotations(Path(args.fashionpedia))
    scores = {}
    if (run / "scores.jsonl").exists():
        for line in (run / "scores.jsonl").read_text().splitlines():
            r = json.loads(line)
            if r["variant"] == args.variant:
                scores[(r["file"], r["k"])] = r["scores"]
    SERVER.update(url=args.url.rstrip("/"), model=args.model)
    s1 = None if args.no_s1 else System1()

    jobs = [(c, k) for c in cases for k in range(args.k)
            if (run / args.variant / f"{c['file'].rsplit('.', 1)[0]}_k{k}.png").exists()]

    def one(job):
        c, k = job
        photo = np.array(Image.open(ann.image_dir / c["file"]).convert("RGB"))
        crop = Image.fromarray(dim_crop(photo, ann.mask(c["file"], c["inner"]["ann_id"]), factor=0.25))
        result = Image.open(run / args.variant / f"{c['file'].rsplit('.', 1)[0]}_k{k}.png")
        sc = scores.get((c["file"], k))
        color = None
        if sc and all(x in sc for x in ("palette_dist", "lightness_gap", "chroma_ratio", "extra_color")):
            color = {x: float(sc[x]) for x in ("palette_dist", "lightness_gap", "chroma_ratio", "extra_color")}
            color["chroma_ratio"] = min(color["chroma_ratio"], 9.99)
            color["neutral"] = float(sc.get("target_chroma", 99)) < 3
        text = text_evidence(crop, result)
        ev = evidence_text(color, text, None)
        row = {"file": c["file"], "k": k, "color": color, "text": text, "evidence": ev}
        if s1 is not None:
            row["p_s1"] = round(s1_decide(s1, crop, result, ev), 4)
        if not args.no_s2:
            v = s2_decide(crop, result, ev)
            row["s2"] = v
            row["pass_s2"] = v.get("verdict") == "pass"
        return row

    out = run / f"judge_{args.tag}_{args.variant}.jsonl"
    rows = []
    with out.open("w") as fh, ThreadPoolExecutor(args.workers) as ex:
        for r in ex.map(one, jobs):
            rows.append(r)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            print(r["file"], r["k"], r.get("p_s1"), r.get("pass_s2"), r["text"]["similarity"], flush=True)
    if args.labels:
        labels = {x["file"]: x["ok"] for x in json.loads(Path(args.labels).read_text())["labels"]}
        first = [r for r in rows if r["k"] == 0 and r["file"] in labels]
        report = {}
        if s1 is not None:
            report = {f"s1>{t}": precision_recall(first, labels, lambda r, t=t: r["p_s1"] > t) for t in (0.5, 0.7, 0.9, 0.97)}
        if not args.no_s2:
            report["s2"] = precision_recall(first, labels, lambda r: r["pass_s2"])
        print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()

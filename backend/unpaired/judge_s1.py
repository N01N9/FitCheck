"""System 1 판정(글 생성 없이 확률 하나): JEV-27B-VL 또는 Cloudflare Clef 를 transformers 로 직접 돌린다.

judge2.py 가 남긴 근거(색 수치·글자 비교)를 같은 문장으로 넣고, 두 이미지(대상만 밝은 확대, 상품 사진)를 보고
"충실한 상품 사진인가" 의 P(true) 를 낸다. 컨테이너의 vLLM(0.27)은 JEV 어댑터의 lm_head LoRA 를 못 올려서
transformers + peft 로 돌린다.

사용 (fitcheck-nas)
  python -m unpaired.judge_s1 --model jev  --run results/unpaired/eval_report/lora_v2_distilled --evidence judge_j2_lora.jsonl
  python -m unpaired.judge_s1 --model clef --run ... --evidence judge_j2_lora.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from unpaired.judge2 import CRITERIA, precision_recall

JEV = Path("/nas/models/models/Atlas3D--JEV-27B-VL-NVFP4")
CLEF = Path("data/clef_local")  # NAS 의 kurcontko--clef-NVFP4 를 가리키는 링크 + 헤드 파일을 맨 위로
QUESTION = "Is the product photo a faithful product photo of the bright garment in the close-up?"
SIDE = 640


def fit(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((SIDE, SIDE), Image.LANCZOS)
    return img


class JevS1:
    def __init__(self, root: Path = JEV):
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.proc = AutoProcessor.from_pretrained(root)
        from transformers import CompressedTensorsConfig
        # 4비트 층은 .weight 가 없어 peft LoRA 를 못 붙인다 → bf16 으로 풀어서 올린다(약 54GB)
        base = AutoModelForImageTextToText.from_pretrained(
            root, dtype=torch.bfloat16, device_map="cuda", quantization_config=CompressedTensorsConfig(run_compressed=False))
        self.model = PeftModel.from_pretrained(base, root / "adapter_vllm").eval()
        self.dh = json.loads((root / "adapter_vllm" / "decision_head.json").read_text())
        self.temp = json.loads((root / "calibration.json").read_text())["per_kind"]["noul"]
        s = self.dh["slots"]["ranges"]["noul"][0]
        self.ids, self.bias = self.dh["verbalizer_ids"][s: s + 2], self.dh["bias"][s: s + 2]

    def __call__(self, crop: Image.Image, result: Image.Image, ev: str) -> float:
        img = "<|vision_start|><|image_pad|><|vision_end|>"
        text = (f"[kind] noul\n[state] Close-up of the photo, target garment bright: {img}\nProduct photo: {img}\n"
                f"{ev}\n{CRITERIA}\n[question] {QUESTION}\n[options]\nfalse\ntrue\n[decision]:")
        inputs = self.proc(text=[text], images=[fit(crop), fit(result)], return_tensors="pt",
                           add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        z = [(float(torch.log_softmax(logits, -1)[t]) + b) / self.temp for t, b in zip(self.ids, self.bias)]
        e = np.exp(np.array(z) - max(z))
        return float(e[1] / e.sum())


class ClefS1:
    def __init__(self, root: Path = CLEF):
        sys.path.insert(0, str(root))
        from joint_schema_model import collate_records, encode_record, load_release_model

        self.enc, self.col = encode_record, collate_records
        from transformers import CompressedTensorsConfig
        self.model, self.proc = load_release_model(str(root), device="cuda",
                                                   quantization_config=CompressedTensorsConfig(run_compressed=False))

    def __call__(self, crop: Image.Image, result: Image.Image, ev: str) -> float:
        record = {"state": {"image_1": "close-up of the photo, target garment bright",
                            "image_2": "product photo", "measurements": ev, "rule": CRITERIA},
                  "images": [fit(crop), fit(result)],
                  "questions": {"faithful": {"type": "noul", "instructions": QUESTION}}}
        e = self.enc(self.proc.tokenizer, record, processor=self.proc)
        batch = self.col([e], self.proc.tokenizer.pad_token_id, torch.device("cuda"))
        with torch.inference_mode():
            logits = self.model(batch)[0][0].float().softmax(-1).tolist()
        opts = list(e.questions[0].option_ids)
        return float(logits[opts.index("true")]) if "true" in opts else float(logits[-1])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="System 1 판정 (JEV / Clef)")
    p.add_argument("--model", choices=("jev", "clef"), required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--variant", default="lora")
    p.add_argument("--evidence", required=True, help="judge2 결과 파일 이름 (run 폴더 안)")
    p.add_argument("--fashionpedia", default="data/fashionpedia")
    p.add_argument("--labels")
    p.add_argument("--limit", type=int)
    args = p.parse_args(argv)

    from unpaired.layered import Annotations
    from unpaired.pointer import dim_crop

    torch.backends.cudnn.enabled = False  # GB10 + cuDNN 9.25: 합성곱 결과가 틀리거나 엔진이 없다(selftest.py)
    run = Path(args.run)
    ev_rows = [json.loads(l) for l in (run / args.evidence).read_text().splitlines()][: args.limit]
    cases = {c["file"]: c for c in json.loads((run / "cases.json").read_text())}
    ann = Annotations(Path(args.fashionpedia))
    judge = JevS1() if args.model == "jev" else ClefS1()
    out = run / f"judge_s1_{args.model}_{args.variant}.jsonl"
    rows = []
    with out.open("w") as fh:
        for r in ev_rows:
            c = cases[r["file"]]
            photo = np.array(Image.open(ann.image_dir / c["file"]).convert("RGB"))
            crop = Image.fromarray(dim_crop(photo, ann.mask(c["file"], c["inner"]["ann_id"]), factor=0.25))
            result = Image.open(run / args.variant / f"{c['file'].rsplit('.', 1)[0]}_k{r['k']}.png")
            row = {"file": r["file"], "k": r["k"], "p": round(judge(crop, result, r["evidence"]), 4)}
            rows.append(row)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            print(row, flush=True)
    if args.labels:
        labels = {x["file"]: x["ok"] for x in json.loads(Path(args.labels).read_text())["labels"]}
        first = [r for r in rows if r["k"] == 0 and r["file"] in labels]
        pos = [r["p"] for r in first if labels[r["file"]]]
        neg = [r["p"] for r in first if not labels[r["file"]]]
        auc = float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])) if pos and neg else None
        rep = {f">{t}": precision_recall(first, labels, lambda r, t=t: r["p"] > t) for t in (0.3, 0.5, 0.7, 0.9, 0.97)}
        rep["auroc"] = round(auc, 3) if auc is not None else None
        print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()

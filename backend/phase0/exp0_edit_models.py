"""실험 0: 편집 모델(Qwen-Image-Edit-2511, FLUX.2-klein-4B)을 GB10 에서 돌려 속도·메모리를 재고,
벗기기(착용 -> 상품) / 입히기(상품 -> 착용) / 바닥 사진 펴기 결과를 눈으로 비교할 수 있게 저장한다.

  - 같은 착용 사진을 "사진 전체 + 지시" 와 "그 옷 부분만 잘라낸 조각" 두 방식으로 넣어 비교한다
    (가려진 옷은 사진 전체 맥락이 필요하다는 가설 확인용).
  - 모델은 둘 다 Apache-2.0. 결과는 results/exp0/<모델>/ 에 저장하고 timings.json 에 시간·메모리를 남긴다.

사용 (컨테이너 안)
  python -m phase0.exp0_edit_models --model qie --cases cases.json --out results/exp0
  python -m phase0.exp0_edit_models --model klein --cases cases.json --out results/exp0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

QIE_ID = "Qwen/Qwen-Image-Edit-2511"
QIE_LIGHTNING = ("lightx2v/Qwen-Image-Edit-2511-Lightning", "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors")
KLEIN_ID = "black-forest-labs/FLUX.2-klein-4B"

NEGATIVE = "person, body, hands, mannequin, hanger, text overlay, watermark, blurry, deformed"


def load_image(path: str, max_side: int = 1024) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_side, max_side))
    # 모델이 요구하는 크기 단위(16 의 배수)로 맞춘다
    w, h = (img.width // 16) * 16, (img.height // 16) * 16
    return img.crop((0, 0, w, h))


def crop_box(img: Image.Image, box: list[float]) -> Image.Image:
    """box 는 원본 대비 비율 [x0, y0, x1, y1]."""
    w, h = img.size
    c = img.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
    cw, ch = (c.width // 16) * 16, (c.height // 16) * 16
    return c.crop((0, 0, max(cw, 256), max(ch, 256)))


class Qie:
    name = "qie2511"

    def __init__(self, lightning: bool):
        import torch
        from diffusers import QwenImageEditPlusPipeline
        from huggingface_hub import hf_hub_download

        self.torch = torch
        # GB10 은 CPU·GPU 가 메모리를 공유한다. CPU 에 먼저 올리고 .to("cuda") 하면 모델(약 58GB)을
        # 두 번 들게 되어 메모리가 바닥나 시스템이 멈췄다(2026-10-02). GPU 로 바로 올린다.
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(QIE_ID, torch_dtype=torch.bfloat16, device_map="cuda")
        self.lightning = lightning
        if lightning:
            self.pipe.load_lora_weights(hf_hub_download(*QIE_LIGHTNING))
            self.name = "qie2511_lightning4"

    def __call__(self, images, prompt, seed):
        steps, cfg = (4, 1.0) if self.lightning else (30, 4.0)
        g = self.torch.Generator("cuda").manual_seed(seed)
        return self.pipe(image=images, prompt=prompt, negative_prompt=NEGATIVE, true_cfg_scale=cfg,
                         num_inference_steps=steps, generator=g).images[0]


class Klein:
    name = "flux2_klein4b"

    def __init__(self):
        import torch
        from diffusers import Flux2KleinPipeline

        self.torch = torch
        self.pipe = Flux2KleinPipeline.from_pretrained(KLEIN_ID, torch_dtype=torch.bfloat16, device_map="cuda")

    def __call__(self, images, prompt, seed):
        g = self.torch.Generator("cuda").manual_seed(seed)
        # klein-4B 는 단계 증류 모델이라 4 스텝, guidance 1.0 이 권장 설정이다
        return self.pipe(image=images, prompt=prompt, num_inference_steps=4, guidance_scale=1.0,
                         generator=g).images[0]


def require_free_memory(gb: float) -> None:
    """불러오기 전에 사용 가능한 메모리를 확인한다. 부족하면 시스템을 멈추게 하지 말고 바로 끝낸다."""
    avail_kb = next(int(line.split()[1]) for line in open("/proc/meminfo") if line.startswith("MemAvailable"))
    if avail_kb / 1e6 < gb:
        raise SystemExit(f"사용 가능한 메모리 {avail_kb / 1e6:.0f}GB < 필요 {gb:.0f}GB. 다른 GPU 작업을 먼저 끄세요.")


def disable_broken_cudnn() -> None:
    """GB10(sm_121) + NGC vllm:26.08(cuDNN 9.25)에서 cuDNN 합성곱이 입력 384px 이상일 때 틀린 값을 낸다
    (FLUX.2 VAE 왕복 오차: cuDNN 켬 111%, 끔 0%, 2026-10-02 확인). 그래서 cuDNN 을 끈다."""
    import torch

    torch.backends.cudnn.enabled = False


def run(model, cases: list[dict], out: Path, seed: int) -> list[dict]:
    import torch

    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        images = [load_image(p) for p in case["images"]]
        if case.get("crop"):
            images = [crop_box(images[0], case["crop"])]
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = model(images, case["prompt"], seed)
        torch.cuda.synchronize()
        sec = time.perf_counter() - start
        result.save(out / f"{case['id']}.png")
        # 입력도 나란히 볼 수 있게 저장한다
        for k, im in enumerate(images):
            im.save(out / f"{case['id']}_in{k}.jpg", quality=90)
        row = {"id": case["id"], "task": case["task"], "seconds": round(sec, 1),
               "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
               "in_sizes": [im.size for im in images], "out_size": result.size}
        print(json.dumps(row), flush=True)
        rows.append(row)
    return rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="실험 0: 편집 모델 속도·결과 비교")
    p.add_argument("--model", choices=["qie", "qie-lightning", "klein"], required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    disable_broken_cudnn()
    require_free_memory(30 if args.model == "klein" else 75)
    t0 = time.perf_counter()
    model = Klein() if args.model == "klein" else Qie(lightning=args.model == "qie-lightning")
    load_s = time.perf_counter() - t0
    cases = json.loads(Path(args.cases).read_text())
    # 첫 호출은 커널 준비 시간이 섞이므로 같은 첫 케이스를 한 번 먼저 돌려 버린다(warm-up)
    run(model, cases[:1], Path(args.out) / model.name / "_warmup", args.seed)
    rows = run(model, cases, Path(args.out) / model.name, args.seed)
    (Path(args.out) / model.name / "timings.json").write_text(
        json.dumps({"model": model.name, "load_seconds": round(load_s, 1), "cases": rows}, indent=2))


if __name__ == "__main__":
    main()

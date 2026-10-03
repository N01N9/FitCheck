"""GB10 컨테이너 자가 점검: 합성곱을 쓰는 모델이 GPU 에서 CPU 와 같은 값을 내는지 본다.

NGC vllm:26.08(cuDNN 9.25)에서는 cuDNN 합성곱이 입력 384px 이상에서 틀린 값을 냈다(2026-10-02).
VAE·DINOv2·BiRefNet 을 cuDNN 켬/끔 GPU 와 CPU 로 돌려 상대 오차를 비교한다. 끔 쪽 오차가 작아야 쓴다.

사용 (컨테이너 안)
  python -m unpaired.selftest --out results/unpaired/selftest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def rel_err(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def check(name: str, build, make_input, forward) -> dict:
    import torch

    import time

    model = build()
    x = make_input()
    res = {"model": name, "input": list(x.shape)}
    with torch.inference_mode():
        cpu = forward(model.to("cpu").float(), x.to("cpu").float())
        model = model.to("cuda")
        for flag in (True, False):
            torch.backends.cudnn.enabled = flag
            forward(model, x.to("cuda"))  # 첫 호출은 준비 시간이 섞인다
            torch.cuda.synchronize()
            start = time.perf_counter()
            out = forward(model, x.to("cuda"))
            torch.cuda.synchronize()
            key = "cudnn_on" if flag else "cudnn_off"
            res[f"{key}_rel_err"] = rel_err(out, cpu)
            res[f"{key}_seconds"] = round(time.perf_counter() - start, 3)
    torch.backends.cudnn.enabled = False
    print(json.dumps(res), flush=True)
    del model
    torch.cuda.empty_cache()
    return res


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="합성곱 모델 GPU/CPU 일치 점검")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    import torch
    from diffusers import AutoencoderKLFlux2
    from transformers import AutoModel, AutoModelForImageSegmentation

    from phase0.exp0_edit_models import KLEIN_ID

    torch.manual_seed(0)
    img = lambda s: torch.rand(1, 3, s, s) * 2 - 1  # noqa: E731
    vae = lambda: AutoencoderKLFlux2.from_pretrained(KLEIN_ID, subfolder="vae").eval()  # noqa: E731
    results = [
        check("flux2_vae_encode", vae, lambda: img(768), lambda m, x: m.encode(x).latent_dist.mode().float().cpu()),
        check("flux2_vae_decode", vae, lambda: torch.randn(1, 32, 96, 96),
              lambda m, z: m.decode(z).sample.float().cpu()),
        check("dinov2_large", lambda: AutoModel.from_pretrained("facebook/dinov2-large").eval(),
              lambda: img(448), lambda m, x: m(pixel_values=x).last_hidden_state.float().cpu()),
        check("birefnet", lambda: AutoModelForImageSegmentation.from_pretrained("ZhengPeng7/BiRefNet", trust_remote_code=True).eval(),
              lambda: img(1024), lambda m, x: m(x)[-1].sigmoid().float().cpu()),
    ]
    Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

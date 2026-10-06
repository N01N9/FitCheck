"""Qwen-Image-Edit-2511(20B, Apache-2.0)에 LoRA 를 학습한다. train_lora.py(klein-4B)와 같은 쌍 파일을 쓴다.

klein-4B 로는 엄격 기준 한 장 통과율이 52%(v3)에서 막혔다 — 목선·비대칭 밑단·글자 같은 세부와 겉옷/이너 구분이 약하다.
같은 데이터를 큰 편집 모델에 학습시켜 본다.

목적함수: rectified flow. x_σ = (1-σ)·x0 + σ·ε, 모델은 v = ε - x0 를 맞힌다(diffusers FlowMatchEuler 와 같은 정의).
입력: [잡음 섞인 정답 토큰, 참조 이미지 토큰들] 을 이어 붙이고, 손실은 정답 토큰에서만 계산한다.
글 인코더(Qwen2.5-VL)는 지시문 + 참조 이미지(384² 넓이)를 읽고, VAE 참조는 768² 넓이로 줄인다(추론 때도 같게:
`set_ref_area`).

사용 (컨테이너 안)
  python -m unpaired.train_qie --benchmark
  python -m unpaired.train_qie --pairs data/unpaired/pairs/v4/pairs.jsonl --out data/unpaired/runs/qie_v1 --steps 3000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

from phase0.exp0_edit_models import QIE_ID, disable_broken_cudnn
from unpaired.train_lora import latest_checkpoint, parse_weights, sampler

QIE_LORA_TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]
REF_AREA = 768 * 768


def set_ref_area(area: int = REF_AREA) -> None:
    """QIE 파이프라인이 참조 이미지를 VAE 에 넣는 넓이(기본 1024²)를 바꾼다 — 학습과 추론을 같게."""
    import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as m
    m.VAE_IMAGE_SIZE = area


class QieTrainer:
    def __init__(self, rank: int = 32, lr: float = 1e-4, grad_ckpt: bool = True, model_id: str = QIE_ID):
        import torch
        from diffusers import QwenImageEditPlusPipeline
        from peft import LoraConfig

        self.torch = torch
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
        self.tr = self.pipe.transformer
        self.tr.requires_grad_(False)
        self.pipe.text_encoder.requires_grad_(False)
        self.pipe.vae.requires_grad_(False)
        self.tr.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                                       target_modules=QIE_LORA_TARGETS))
        self.params = [p for p in self.tr.parameters() if p.requires_grad]
        for p in self.params:
            p.data = p.data.float()
        if grad_ckpt:
            self.tr.enable_gradient_checkpointing()
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=1e-4)
        self.vsf = self.pipe.vae_scale_factor
        self.c = self.tr.config.in_channels // 4

    def latents(self, img: Image.Image, w: int, h: int):
        t = self.pipe.image_processor.preprocess(img, h, w).unsqueeze(2).to("cuda", self.torch.bfloat16)
        lat = self.pipe._encode_vae_image(t, None)  # (1, C, 1, h/8, w/8), 정규화 포함
        return self.pipe._pack_latents(lat, 1, self.c, lat.shape[3], lat.shape[4])

    def step(self, pair: dict, rng: random.Random) -> float:
        import torch
        from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import CONDITION_IMAGE_SIZE, calculate_dimensions

        refs = [Image.open(p).convert("RGB") for p in pair["refs"]]
        w, h = pair["size"]
        cond, ref_sizes = [], []
        for r in refs:
            cw, ch = calculate_dimensions(CONDITION_IMAGE_SIZE, r.width / r.height)
            cond.append(self.pipe.image_processor.resize(r, ch, cw))
            ref_sizes.append(calculate_dimensions(REF_AREA, r.width / r.height))
        with torch.no_grad():
            emb, emb_mask = self.pipe.encode_prompt(image=cond, prompt=pair["prompt"], device="cuda")
            ref_lat = torch.cat([self.latents(r, rw, rh) for r, (rw, rh) in zip(refs, ref_sizes)], dim=1)
            x0 = self.latents(Image.open(pair["target"]).convert("RGB"), w, h)
        # 시그모이드-정규 표집(가운데 σ 를 더 자주) — FLUX/Qwen-Image 학습 관례
        sigma = 1 / (1 + math.exp(-rng.gauss(0, 1)))
        eps = torch.randn_like(x0)
        xt = (1 - sigma) * x0 + sigma * eps
        img_shapes = [[(1, h // self.vsf // 2, w // self.vsf // 2),
                       *[(1, rh // self.vsf // 2, rw // self.vsf // 2) for rw, rh in ref_sizes]]]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = self.tr(hidden_states=torch.cat([xt, ref_lat], dim=1),
                           timestep=torch.tensor([sigma], device="cuda", dtype=torch.bfloat16),
                           encoder_hidden_states=emb, encoder_hidden_states_mask=emb_mask,
                           img_shapes=img_shapes, return_dict=False)[0][:, : x0.shape[1]]
        loss = torch.nn.functional.mse_loss(pred.float(), (eps - x0).float())
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        self.opt.step()
        return float(loss)

    def save(self, path: Path) -> None:
        from diffusers import QwenImageEditPlusPipeline
        from peft.utils import get_peft_model_state_dict

        path.mkdir(parents=True, exist_ok=True)
        QwenImageEditPlusPipeline.save_lora_weights(path, transformer_lora_layers=get_peft_model_state_dict(self.tr))
        self.torch.save(self.opt.state_dict(), path / "optimizer.pt")

    def load(self, path: Path) -> None:
        from peft.utils import set_peft_model_state_dict
        from safetensors.torch import load_file

        sd = {k.replace("transformer.", "", 1): v for k, v in load_file(path / "pytorch_lora_weights.safetensors").items()}
        set_peft_model_state_dict(self.tr, sd)
        for p in self.params:
            p.data = p.data.float()
        if (path / "optimizer.pt").exists():
            self.opt.load_state_dict(self.torch.load(path / "optimizer.pt"))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="QIE-2511 LoRA 학습")
    p.add_argument("--pairs")
    p.add_argument("--out")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--save-every", type=int, default=250)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source-weights")
    p.add_argument("--benchmark", action="store_true")
    args = p.parse_args(argv)

    disable_broken_cudnn()
    set_ref_area()
    import torch

    tr = QieTrainer(args.rank, args.lr)
    if args.benchmark:
        pairs = [json.loads(l) for l in Path("data/unpaired/pairs/v3/pairs.jsonl").read_text().splitlines()[:6]]
        rng = random.Random(0)
        for i, pr in enumerate(pairs):
            t = time.time()
            loss = tr.step(pr, rng)
            torch.cuda.synchronize()
            print(json.dumps({"i": i, "loss": round(loss, 4), "sec": round(time.time() - t, 2),
                              "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}), flush=True)
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(l) for l in Path(args.pairs).read_text().splitlines()]
    start = 0
    last = latest_checkpoint(out)
    if last:
        start, path = last
        tr.load(path)
        print(f"resume from {path}", flush=True)
    rng = random.Random(args.seed + start)
    draw = sampler(pairs, parse_weights(args.source_weights), rng)
    t0, losses = time.time(), []
    with (out / "train_log.jsonl").open("a") as log:
        for step in range(start + 1, args.steps + 1):
            losses.append(tr.step(draw(), rng))
            if step % 10 == 0:
                log.write(json.dumps({"step": step, "loss": round(float(np.mean(losses)), 5),
                                      "sec": round(time.time() - t0, 1)}) + "\n")
                log.flush()
                losses = []
            if step % args.save_every == 0 or step == args.steps:
                tr.save(out / f"step{step:05d}")


if __name__ == "__main__":
    main()

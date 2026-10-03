"""FLUX.2-klein-base-4B(Apache-2.0)에 LoRA 를 학습한다. 서비스는 4스텝 klein-4B 에 같은 LoRA 를 얹는다.

목적함수는 rectified flow 다: x_t = (1-t)·x0 + t·ε 에서 속도 v = ε - x0 를 맞힌다(FLUX 표준).
참조 이미지(사진 전체 + 표시, 보조 크롭)는 토큰으로 이어 붙이고, 손실은 정답 토큰에서만 계산한다.
정답 토큰마다 가중치를 줄 수 있다(입력에서 안 보였던 로고·글자 자리는 0 — 지어내기를 배우지 않게).

쌍 파일(jsonl) 한 줄: {"target": 정답 경로, "refs": [참조 경로...], "prompt": 지시문,
                     "weight": 가중치 마스크 경로(선택, 흰색=1), "size": [w, h]}

사용 (컨테이너 안)
  python -m unpaired.train_lora --benchmark                       # 스텝 시간·메모리만 잰다
  python -m unpaired.train_lora --pairs pairs.jsonl --out runs/lora1 --steps 3000
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

from phase0.exp0_edit_models import KLEIN_ID, disable_broken_cudnn, require_free_memory

KLEIN_BASE_ID = "black-forest-labs/FLUX.2-klein-base-4B"
# 이중 블록의 주의(attention) 투영 + 단일 블록의 묶음 투영. 단일 블록 to_out 은 Linear, 이중 블록은 to_out.0
LORA_TARGETS = (r".*\.(to_q|to_k|to_v|to_out\.0|add_q_proj|add_k_proj|add_v_proj|to_add_out|to_qkv_mlp_proj)$"
                r"|.*single_transformer_blocks\.\d+\.attn\.to_out$")


class Trainer:
    def __init__(self, rank: int = 32, lr: float = 1e-4, grad_ckpt: bool = True):
        import torch
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel
        from peft import LoraConfig

        self.torch = torch
        transformer = Flux2Transformer2DModel.from_pretrained(
            KLEIN_BASE_ID, subfolder="transformer", torch_dtype=torch.bfloat16, device_map="cuda")
        self.pipe = Flux2KleinPipeline.from_pretrained(KLEIN_ID, transformer=transformer, torch_dtype=torch.bfloat16,
                                                       device_map="cuda")
        self.tr = self.pipe.transformer
        self.tr.requires_grad_(False)
        self.tr.add_adapter(LoraConfig(r=rank, lora_alpha=rank, init_lora_weights="gaussian",
                                       target_modules=LORA_TARGETS))
        self.params = [p for p in self.tr.parameters() if p.requires_grad]
        for p in self.params:
            p.data = p.data.float()  # LoRA 가중치는 fp32 로 두고 본체는 bf16
        if grad_ckpt:
            self.tr.enable_gradient_checkpointing()
        self.opt = torch.optim.AdamW(self.params, lr=lr, weight_decay=1e-4)
        self.text_cache: dict[str, tuple] = {}

    # --- 입력 준비 -------------------------------------------------------------
    def encode_text(self, prompt: str):
        if prompt not in self.text_cache:
            with self.torch.no_grad():
                emb, ids = self.pipe.encode_prompt(prompt=prompt, device="cuda", num_images_per_prompt=1,
                                                   max_sequence_length=512)
            self.text_cache[prompt] = (emb, ids)
        return self.text_cache[prompt]

    def encode_image(self, img: Image.Image, size: tuple[int, int]):
        """(1, 128, H/16, W/16) 정규화 latent. size 는 16 의 배수."""
        x = self.pipe.image_processor.preprocess(img.convert("RGB"), height=size[1], width=size[0], resize_mode="crop")
        with self.torch.no_grad():
            return self.pipe._encode_vae_image(x.to("cuda", self.torch.bfloat16), generator=None)

    def pack_refs(self, ref_latents: list):
        ids = self.pipe._prepare_image_ids(ref_latents).to("cuda")
        tokens = self.torch.cat([self.pipe._pack_latents(r).squeeze(0) for r in ref_latents], dim=0).unsqueeze(0)
        return tokens, ids

    # --- 학습 한 스텝 ------------------------------------------------------------
    def sample_t(self, n_tokens: int):
        """logit-normal 로 뽑고, 해상도에 맞춰 시간 이동(FLUX 추론과 같은 mu)을 준다."""
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu

        torch = self.torch
        t = torch.sigmoid(torch.randn(1, device="cuda"))
        shift = math.exp(compute_empirical_mu(n_tokens, 50))
        return shift * t / (1 + (shift - 1) * t)

    def step(self, target_latent, ref_tokens, ref_ids, text, weight=None) -> float:
        torch = self.torch
        x0 = self.pipe._pack_latents(target_latent).float()          # (1, N, 128)
        lat_ids = self.pipe._prepare_latent_ids(target_latent).to("cuda")
        noise = torch.randn_like(x0)
        t = self.sample_t(x0.shape[1])
        xt = (1 - t) * x0 + t * noise
        emb, txt_ids = text
        hidden = torch.cat([xt.to(torch.bfloat16), ref_tokens], dim=1)
        ids = torch.cat([lat_ids, ref_ids], dim=1)
        pred = self.tr(hidden_states=hidden, timestep=t.to(torch.bfloat16), guidance=None, encoder_hidden_states=emb,
                       txt_ids=txt_ids, img_ids=ids, return_dict=False)[0][:, : x0.shape[1]]
        err = ((pred.float() - (noise - x0)) ** 2).mean(-1)          # (1, N)
        loss = err.mean() if weight is None else (err * weight).sum() / weight.sum().clamp(min=1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 1.0)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        return float(loss.detach())

    def save(self, path: Path) -> None:
        from diffusers import Flux2KleinPipeline
        from peft.utils import get_peft_model_state_dict

        path.mkdir(parents=True, exist_ok=True)
        state = {k: v.to(self.torch.bfloat16) for k, v in get_peft_model_state_dict(self.tr).items()}
        Flux2KleinPipeline.save_lora_weights(path, transformer_lora_layers=state)


def weight_tokens(path: str | None, size: tuple[int, int]):
    """가중치 마스크(흰색=1)를 정답 토큰 격자(16px 단위)로 줄인다."""
    if not path:
        return None
    import torch

    m = Image.open(path).convert("L").resize((size[0] // 16, size[1] // 16), Image.BOX)
    return torch.from_numpy(np.asarray(m, np.float32) / 255).reshape(1, -1).to("cuda")


def benchmark(trainer: Trainer, out: Path) -> list[dict]:
    torch = trainer.torch
    text = trainer.encode_text("[EXTRACT] t-shirt; layer=inner under jacket")
    rows = []
    for side in (768, 1024):
        for n_refs in (1, 2, 3):
            lat = torch.randn(1, 128, side // 16, side // 16, device="cuda", dtype=torch.bfloat16)
            refs = [torch.randn_like(lat) for _ in range(n_refs)]
            ref_tokens, ref_ids = trainer.pack_refs(refs)
            torch.cuda.reset_peak_memory_stats()
            times = []
            for i in range(6):
                torch.cuda.synchronize()
                start = time.perf_counter()
                trainer.step(lat, ref_tokens, ref_ids, text)
                torch.cuda.synchronize()
                if i >= 1:  # 첫 스텝은 준비 시간이 섞인다
                    times.append(time.perf_counter() - start)
            row = {"side": side, "refs": n_refs, "tokens": (1 + n_refs) * (side // 16) ** 2,
                   "sec_per_step": round(float(np.median(times)), 2),
                   "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
            print(json.dumps(row), flush=True)
            rows.append(row)
    out.write_text(json.dumps(rows, indent=2))
    return rows


def parse_weights(text: str | None) -> dict[str, float]:
    """"swap=3,flatlay=1" → {"swap": 3.0, "flatlay": 1.0}. 비우면 모든 출처 1."""
    if not text:
        return {}
    return {k: float(v) for k, v in (item.split("=") for item in text.split(","))}


def sampler(pairs: list[dict], weights: dict[str, float], rng: random.Random):
    """출처를 먼저 가중치로 고르고, 그 안에서 쌍을 고른다(수가 많은 출처가 학습을 다 차지하지 않게)."""
    by_source: dict[str, list[dict]] = {}
    for pair in pairs:
        by_source.setdefault(pair.get("source", "?"), []).append(pair)
    names = sorted(by_source)
    w = [weights.get(n, 1.0) for n in names]

    def draw() -> dict:
        return rng.choice(by_source[rng.choices(names, weights=w)[0]])

    return draw


def train(trainer: Trainer, pairs: list[dict], steps: int, out: Path, save_every: int, seed: int,
          weights: dict[str, float] | None = None) -> None:
    rng = random.Random(seed)
    draw = sampler(pairs, weights or {}, rng)
    log = (out / "train_log.jsonl").open("a")
    start = time.perf_counter()
    for step in range(1, steps + 1):
        pair = draw()
        size = tuple(pair["size"])
        target = trainer.encode_image(Image.open(pair["target"]), size)
        refs = [trainer.encode_image(Image.open(r), _ref_size(Image.open(r))) for r in pair["refs"]]
        ref_tokens, ref_ids = trainer.pack_refs(refs)
        loss = trainer.step(target, ref_tokens, ref_ids, trainer.encode_text(pair["prompt"]),
                            weight_tokens(pair.get("weight"), size))
        if step % 10 == 0:
            log.write(json.dumps({"step": step, "loss": round(loss, 5),
                                  "sec": round(time.perf_counter() - start, 1)}) + "\n")
            log.flush()
        if step % save_every == 0 or step == steps:
            trainer.save(out / f"step{step:05d}")


def _ref_size(img: Image.Image, max_side: int = 1024) -> tuple[int, int]:
    s = min(1.0, max_side / max(img.size))
    return int(img.width * s) // 16 * 16, int(img.height * s) // 16 * 16


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="klein-base-4B LoRA 학습")
    p.add_argument("--benchmark", action="store_true")
    p.add_argument("--pairs")
    p.add_argument("--out", default="results/unpaired/lora_benchmark.json")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source-weights", help='출처별 표집 비율, 예: "swap=3,flatlay=1"')
    args = p.parse_args(argv)

    disable_broken_cudnn()
    require_free_memory(40)
    trainer = Trainer(rank=args.rank, lr=args.lr)
    if args.benchmark:
        benchmark(trainer, Path(args.out))
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = [json.loads(line) for line in Path(args.pairs).read_text().splitlines() if line]
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    train(trainer, pairs, args.steps, out, args.save_every, args.seed, parse_weights(args.source_weights))


if __name__ == "__main__":
    main()

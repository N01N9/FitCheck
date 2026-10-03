"""이미지 편집 모델을 같은 모양으로 감싼다: (참조 이미지들, 지시문, 시드 K개, 출력 크기) → 결과 K장.

모델 상수·GB10 안전장치는 실험 0(phase0.exp0_edit_models)의 것을 그대로 쓴다.
"""

from __future__ import annotations

from PIL import Image

from phase0.exp0_edit_models import KLEIN_ID, NEGATIVE, QIE_ID, QIE_LIGHTNING, disable_broken_cudnn, require_free_memory

KLEIN_BASE_ID = "black-forest-labs/FLUX.2-klein-base-4B"


class Klein:
    """FLUX.2-klein-4B. lora 를 주면 학습한 LoRA 를 얹는다.

    base=True 면 같은 부품에 klein-base-4B 본체를 끼우고 CFG 로 여러 스텝 돈다(LoRA 를 학습한 그 모델).
    2026-10-03 진단: base 에서 학습한 LoRA 가 새 지시문 형식("[EXTRACT] ...")을 쓰면 4스텝 증류 모델에서는
    입력 사진을 그대로 다시 그렸고, base(30스텝, CFG 4)에서는 5건 중 4건이 맞는 상품 사진이었다.
    """

    def __init__(self, lora: str | None = None, base: bool = False, steps: int | None = None,
                 guidance: float | None = None):
        import torch
        from diffusers import Flux2KleinPipeline, Flux2Transformer2DModel

        self.torch = torch
        if base:
            tr = Flux2Transformer2DModel.from_pretrained(KLEIN_BASE_ID, subfolder="transformer", torch_dtype=torch.bfloat16,
                                                         device_map="cuda")
            self.pipe = Flux2KleinPipeline.from_pretrained(KLEIN_ID, transformer=tr, torch_dtype=torch.bfloat16,
                                                           device_map="cuda", is_distilled=False)
        else:
            self.pipe = Flux2KleinPipeline.from_pretrained(KLEIN_ID, torch_dtype=torch.bfloat16, device_map="cuda")
        self.steps = steps or (20 if base else 4)
        self.guidance = guidance or (4.0 if base else 1.0)
        self.name = "klein4b_base" if base else "klein4b"
        if lora:
            self.pipe.load_lora_weights(lora)
            self.name += "_lora"

    def __call__(self, images: list[Image.Image], prompt: str, seeds: list[int], size: tuple[int, int]) -> list[Image.Image]:
        gens = [self.torch.Generator("cuda").manual_seed(s) for s in seeds]
        return self.pipe(image=images, prompt=prompt, width=size[0], height=size[1], num_inference_steps=self.steps,
                         guidance_scale=self.guidance, num_images_per_prompt=len(seeds), generator=gens).images


class Qie:
    """Qwen-Image-Edit-2511 + Lightning 4스텝 LoRA."""

    def __init__(self):
        import torch
        from diffusers import QwenImageEditPlusPipeline
        from huggingface_hub import hf_hub_download

        self.torch = torch
        self.pipe = QwenImageEditPlusPipeline.from_pretrained(QIE_ID, torch_dtype=torch.bfloat16, device_map="cuda")
        self.pipe.load_lora_weights(hf_hub_download(*QIE_LIGHTNING), adapter_name="lightning")
        self.pipe.set_adapters(["lightning"], adapter_weights=[1.0])
        self.name = "qie2511_lightning4"

    def __call__(self, images, prompt, seeds, size):
        out = []
        # QIE 는 메모리가 빠듯해서 한 장씩 만든다
        for s in seeds:
            g = self.torch.Generator("cuda").manual_seed(s)
            out += self.pipe(image=images, prompt=prompt, negative_prompt=NEGATIVE, true_cfg_scale=1.0,
                             width=size[0], height=size[1], num_inference_steps=4, generator=g).images
        return out


def load(name: str, lora: str | None = None, base: bool = False):
    disable_broken_cudnn()
    require_free_memory(30 if name == "klein" else 75)
    return Klein(lora, base=base) if name == "klein" else Qie()

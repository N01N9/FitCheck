"""이미지 편집 모델을 같은 모양으로 감싼다: (참조 이미지들, 지시문, 시드 K개, 출력 크기) → 결과 K장.

모델 상수·GB10 안전장치는 실험 0(phase0.exp0_edit_models)의 것을 그대로 쓴다.
"""

from __future__ import annotations

from PIL import Image

from phase0.exp0_edit_models import KLEIN_ID, NEGATIVE, QIE_ID, QIE_LIGHTNING, disable_broken_cudnn, require_free_memory


class Klein:
    """FLUX.2-klein-4B(4스텝 증류). lora 를 주면 학습한 LoRA 를 얹는다."""

    def __init__(self, lora: str | None = None):
        import torch
        from diffusers import Flux2KleinPipeline

        self.torch = torch
        self.pipe = Flux2KleinPipeline.from_pretrained(KLEIN_ID, torch_dtype=torch.bfloat16, device_map="cuda")
        self.name = "klein4b"
        if lora:
            self.pipe.load_lora_weights(lora)
            self.name += "_lora"

    def __call__(self, images: list[Image.Image], prompt: str, seeds: list[int], size: tuple[int, int]) -> list[Image.Image]:
        gens = [self.torch.Generator("cuda").manual_seed(s) for s in seeds]
        return self.pipe(image=images, prompt=prompt, width=size[0], height=size[1], num_inference_steps=4,
                         guidance_scale=1.0, num_images_per_prompt=len(seeds), generator=gens).images


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


def load(name: str, lora: str | None = None):
    disable_broken_cudnn()
    require_free_memory(30 if name == "klein" else 75)
    return Klein(lora) if name == "klein" else Qie()

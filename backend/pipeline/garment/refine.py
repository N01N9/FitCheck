"""3단계. 경계 다듬기(refine).

2단계의 거친 마스크를 기준으로 아이템을 여유 있게 잘라낸 뒤, 배경 제거 모델로
경계를 다시 정밀하게 딴다. 얇은 끈·니트·레이스·비치는 소재가 살아야 한다.

2단계 마스크를 조금 부풀린 것을 **울타리**로 써서, 옷걸이나 손이 다시 들어오지 않게 막는다.

backend
  birefnet : BiRefNet (MIT). general / HR / matting / lite 변형을 --model-id 로 고른다. 1순위
  ben2     : BEN2 Base (MIT). 비교 후보
  border   : 가장자리 색을 배경으로 보는 고전 방식. GPU 없이 도는 기준선
  fake     : 테스트용. 입력 마스크를 그대로 알파로 쓴다

Phase 0 의 T01(배경 제거 비교) 테스트도 여기 있는 모델 래퍼를 그대로 쓴다.
"""

from __future__ import annotations

import time

import numpy as np
from PIL import Image

BIREFNET_VARIANTS = {
    "general": ("ZhengPeng7/BiRefNet", 1024),
    "hr": ("ZhengPeng7/BiRefNet_HR", 2048),
    "matting": ("ZhengPeng7/BiRefNet-matting", 1024),
    "lite": ("ZhengPeng7/BiRefNet_lite", 1024),
}


# --------------------------------------------------------------------------
# 배경 제거 모델들 (Phase 0 T01 과 공용)
# --------------------------------------------------------------------------
class BorderRemover:
    """사진 가장자리의 중앙값 색을 배경으로 보고, 그 색과 먼 픽셀을 옷으로 본다."""

    name = "border"
    model_id = None

    def __init__(self, threshold: float = 40.0):
        self.threshold = threshold

    def predict_mask(self, img: Image.Image) -> Image.Image:
        a = np.asarray(img.convert("RGB"), dtype=np.float32)
        edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
        bg = np.median(edge, axis=0)
        dist = np.linalg.norm(a - bg, axis=-1)
        return Image.fromarray(((dist > self.threshold) * 255).astype(np.uint8))


class BiRefNetRemover:
    """BiRefNet (MIT). NGC 컨테이너의 torch 를 쓰고 ARM64 에서는 SDPA 로 돈다."""

    name = "birefnet"

    def __init__(self, model_id: str = "ZhengPeng7/BiRefNet", resolution: int | None = None,
                 half: bool = True):
        import torch
        from torchvision import transforms
        from transformers import AutoModelForImageSegmentation

        self.torch = torch
        self.model_id = model_id
        self.cuda = torch.cuda.is_available()
        self.half = half and self.cuda
        if resolution is None:
            resolution = next((r for mid, r in BIREFNET_VARIANTS.values() if mid == model_id), 1024)
        self.resolution = resolution
        self.model = AutoModelForImageSegmentation.from_pretrained(model_id, trust_remote_code=True)
        self.model.eval().to("cuda" if self.cuda else "cpu")
        if self.cuda:
            torch.set_float32_matmul_precision("high")
            if self.half:
                self.model.half()
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def predict_mask(self, img: Image.Image) -> Image.Image:
        x = self.transform(img.convert("RGB")).unsqueeze(0)
        if self.cuda:
            x = x.to("cuda")
            if self.half:
                x = x.half()
        with self.torch.inference_mode():
            pred = self.model(x)[-1].sigmoid().float().cpu()[0, 0].numpy()
        return Image.fromarray((pred * 255).astype(np.uint8)).resize(img.size, Image.BILINEAR)


class Ben2Remover:
    """BEN2 Base (MIT). transformers 모델이 아니라 저장소의 BEN2.py 를 직접 불러 쓴다.

    유료 Refiner 는 쓰지 않는다. 가중치를 받기 전에는 생성 시점에 바로 실패한다.
    """

    name = "ben2"

    def __init__(self, repo_id: str = "PramaLLC/BEN2", resolution: int = 1024):
        import importlib.util

        import torch
        from huggingface_hub import hf_hub_download

        self.torch = torch
        self.model_id = repo_id
        self.resolution = resolution
        code = hf_hub_download(repo_id, "BEN2.py")
        weights = hf_hub_download(repo_id, "BEN2_Base.pth")
        spec = importlib.util.spec_from_file_location("ben2_module", code)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.cuda = torch.cuda.is_available()
        self.model = module.BEN_Base().eval()
        self.model.loadcheckpoints(weights)
        self.model.to("cuda" if self.cuda else "cpu")

    def predict_mask(self, img: Image.Image) -> Image.Image:
        out = self.model.inference(img.convert("RGB"))
        if isinstance(out, Image.Image) and out.mode == "RGBA":
            return out.split()[-1].resize(img.size, Image.BILINEAR)
        return Image.fromarray(np.asarray(out).astype(np.uint8)).resize(img.size, Image.BILINEAR)


class FakeRemover:
    """테스트용. 넘겨받은 울타리 마스크를 그대로 알파로 쓴다."""

    name = "fake"
    model_id = "fake"

    def predict_mask(self, img: Image.Image) -> Image.Image:
        return Image.new("L", img.size, 255)


def build_remover(backend: str, **kw):
    if backend == "birefnet":
        variant = kw.get("variant", "general")
        model_id, res = BIREFNET_VARIANTS.get(variant, (kw.get("model_id", variant), None))
        return BiRefNetRemover(model_id, kw.get("resolution", res))
    if backend == "ben2":
        return Ben2Remover(kw.get("model_id", "PramaLLC/BEN2"))
    if backend == "fake":
        return FakeRemover()
    return BorderRemover(kw.get("threshold", 40.0))


# --------------------------------------------------------------------------
# 마스크 도구
# --------------------------------------------------------------------------
def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    """마스크를 px 만큼 부풀린다. scipy 가 있으면 쓰고 없으면 밀어서 합친다."""
    if px <= 0:
        return mask
    try:
        from scipy import ndimage

        return ndimage.binary_dilation(mask, iterations=px)
    except ImportError:
        out = mask.copy()
        for _ in range(px):
            shifted = out.copy()
            shifted[1:] |= out[:-1]
            shifted[:-1] |= out[1:]
            shifted[:, 1:] |= out[:, :-1]
            shifted[:, :-1] |= out[:, 1:]
            out = shifted
        return out


def bbox_of(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return (0, 0, mask.shape[1], mask.shape[0])
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def expand_box(box: tuple[int, int, int, int], size: tuple[int, int], margin: float) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    w, h = size
    mx, my = round((x1 - x0) * margin), round((y1 - y0) * margin)
    return max(0, x0 - mx), max(0, y0 - my), min(w, x1 + mx), min(h, y1 + my)


# --------------------------------------------------------------------------
# 3단계 본체
# --------------------------------------------------------------------------
class Refiner:
    def __init__(self, remover, margin: float = 0.10, guard_px: int = 6,
                 min_alpha: int = 8, use_guard: bool = True):
        self.remover = remover
        self.margin = margin          # 자를 때 두는 여유
        self.guard_px = guard_px      # 울타리를 얼마나 부풀릴지
        self.min_alpha = min_alpha    # 이보다 작은 알파는 0 으로 (먼지 제거)
        self.use_guard = use_guard

    @property
    def name(self) -> str:
        return self.remover.name

    @property
    def model_id(self) -> str | None:
        return getattr(self.remover, "model_id", None)

    def refine(self, img: Image.Image, mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
        """아이템 하나의 정밀 알파(uint8, 원본 크기)와 bbox, 걸린 시간을 돌려준다."""
        start = time.perf_counter()
        box = expand_box(bbox_of(mask), img.size, self.margin)
        x0, y0, x1, y1 = box
        crop = img.convert("RGB").crop(box)

        alpha_crop = np.asarray(self.remover.predict_mask(crop).convert("L")).astype(np.uint16)
        if self.use_guard:
            # 2단계 마스크를 조금 부풀린 울타리 밖은 0 으로. 옷걸이·손이 다시 들어오지 못한다.
            guard = dilate(mask[y0:y1, x0:x1], self.guard_px)
            alpha_crop = np.where(guard, alpha_crop, 0)
        alpha_crop[alpha_crop < self.min_alpha] = 0

        alpha = np.zeros(mask.shape, dtype=np.uint8)
        alpha[y0:y1, x0:x1] = alpha_crop.astype(np.uint8)
        return alpha, box, time.perf_counter() - start


def occlusion_ratio(alpha: np.ndarray) -> float:
    """구멍(팔·머리카락에 가려진 부분) 비율로 가려짐 정도를 어림잡는다."""
    solid = alpha > 127
    if not solid.any():
        return 0.0
    filled = fill(solid)
    area = filled.sum()
    return float(1.0 - solid.sum() / area) if area else 0.0


def fill(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy import ndimage

        return ndimage.binary_fill_holes(mask)
    except ImportError:
        return mask

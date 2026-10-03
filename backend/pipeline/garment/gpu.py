"""GPU 별 우회 설정.

DGX Spark(GB10, sm_121) + NGC vllm:26.08(cuDNN 9.25)에서는 cuDNN 합성곱이 틀린 값을 낸다.
2026-10-03 점검(unpaired.selftest): cuDNN 을 켜면 CPU 대비 상대 오차가 FLUX.2 VAE 디코더 96%,
BiRefNet 60% 였고, 끄면 0.0003% 이하였다. 다른 GPU 에서는 cuDNN 이 정상이고 빠르므로 GB10 에서만 끈다.
"""

from __future__ import annotations

BROKEN_CUDNN_CAPABILITIES = {(12, 1)}


def guard_cudnn() -> bool:
    """문제 있는 GPU 면 cuDNN 을 끄고 True 를 돌려준다."""
    import torch

    if torch.cuda.is_available() and torch.cuda.get_device_capability() in BROKEN_CUDNN_CAPABILITIES:
        torch.backends.cudnn.enabled = False
        return True
    return False

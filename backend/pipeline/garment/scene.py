"""1단계. 장면 판단(scene).

사진이 case1/2/3 중 무엇이고 어떤 아이템이 몇 개 들어 있는지 알아낸다.
결과는 **힌트**다. 2단계는 이 값이 없거나 틀려도 스스로 아이템을 찾아야 한다.

backend
  vlm       : vLLM 등 OpenAI 호환 서버의 VLM 에게 JSON 으로 물어본다 (1순위)
  heuristic : 배경 단색도·살색 비율로 어림잡는다 (VLM 이 없을 때)
  none      : 판단하지 않는다 (2단계가 알아서 한다)
"""

from __future__ import annotations

import base64
import io
import json
import time
import urllib.request

import numpy as np
from PIL import Image

from pipeline.garment.types import CASES, CATEGORIES, SceneInfo

SCENE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["case", "confidence", "items"],
    "properties": {
        "case": {"type": "string", "enum": CASES},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "items": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["category"],
                "properties": {
                    "category": {"type": "string", "enum": CATEGORIES},
                    "note": {"type": "string"},
                },
            },
        },
    },
}

PROMPT = f"""이 사진을 보고 아래 JSON 형식으로만 답한다.

case 는 다음 중 하나다.
- case1: 단색·스튜디오 배경의 상품 사진. 보통 옷 한 벌만 있다.
- case2: 사람이 옷을 입고 찍은 사진.
- case3: 바닥·침대·이불 위에 펼쳐 놓거나 옷걸이에 건 사진.

items 에는 사진에서 **옷·패션 아이템만** 빠짐없이 적는다.
사람, 얼굴, 손, 머리카락, 옷걸이, 가구, 배경 소품은 적지 않는다.
한 켤레 신발은 1개로 센다. 겹쳐 입은 아우터와 안에 입은 상의는 따로 센다.
category 는 {", ".join(CATEGORIES)} 중 하나만 쓴다.
note 에는 색·소재처럼 구분에 도움이 되는 짧은 말을 적는다."""


def _b64(img: Image.Image, max_side: int = 1024) -> str:
    small = img.copy()
    small.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    small.convert("RGB").save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


class VlmScene:
    """OpenAI 호환 서버(vLLM 권장)의 VLM 에게 물어본다."""

    name = "vlm"

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY",
                 timeout: float = 120, max_side: int = 1024, thinking: bool = False):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.api_key, self.timeout = model, api_key, timeout
        self.max_side, self.thinking = max_side, thinking

    def analyze(self, img: Image.Image) -> SceneInfo:
        body = {
            "model": self.model, "temperature": 0, "max_tokens": 768,
            "messages": [
                {"role": "system", "content": "너는 패션 사진 분석기다. 지정된 JSON 형식으로만 답한다."},
                {"role": "user", "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": _b64(img, self.max_side)}},
                ]},
            ],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "scene", "schema": SCENE_SCHEMA, "strict": True}},
            "chat_template_kwargs": {"enable_thinking": self.thinking},
        }
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.load(resp)
        text = data["choices"][0]["message"].get("content") or ""
        seconds = time.perf_counter() - start

        parsed = _parse_json(text)
        if parsed is None:
            return SceneInfo(case="case3", confidence=0.0, backend=self.name,
                             model=self.model, seconds=round(seconds, 3), raw=text[:2000])
        return SceneInfo(
            case=parsed.get("case", "case3"),
            confidence=float(parsed.get("confidence", 0.0)),
            expected_items=[i for i in parsed.get("items", []) if i.get("category") in CATEGORIES],
            backend=self.name, model=self.model, seconds=round(seconds, 3), raw=text[:2000],
        )


def _parse_json(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


class HeuristicScene:
    """VLM 없이 case 만 어림잡는다. 아이템 목록은 비워 둔다(2단계가 찾는다)."""

    name = "heuristic"

    def __init__(self, plain_bg_std: float = 18.0, skin_ratio: float = 0.06):
        self.plain_bg_std, self.skin_ratio = plain_bg_std, skin_ratio

    def analyze(self, img: Image.Image) -> SceneInfo:
        start = time.perf_counter()
        a = np.asarray(img.convert("RGB").resize((256, 256)), dtype=np.float32)
        border = np.concatenate([a[:8].reshape(-1, 3), a[-8:].reshape(-1, 3),
                                 a[:, :8].reshape(-1, 3), a[:, -8:].reshape(-1, 3)])
        bg_std = float(border.std(axis=0).mean())
        skin = _skin_ratio(a)

        if skin >= self.skin_ratio:
            case, conf = "case2", min(1.0, 0.5 + skin)
        elif bg_std <= self.plain_bg_std:
            case, conf = "case1", 0.6
        else:
            case, conf = "case3", 0.5
        return SceneInfo(case=case, confidence=round(conf, 3), backend=self.name,
                         seconds=round(time.perf_counter() - start, 3),
                         raw=f"bg_std={bg_std:.1f} skin={skin:.3f}")


def _skin_ratio(rgb: np.ndarray) -> float:
    """YCbCr 기준의 흔한 살색 범위. 대충이지만 사람이 있는지 정도는 가른다."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b
    y = 0.299 * r + 0.587 * g + 0.114 * b
    mask = (y > 60) & (cb > 77) & (cb < 127) & (cr > 133) & (cr < 173)
    return float(mask.mean())


class NoScene:
    name = "none"

    def analyze(self, img: Image.Image) -> SceneInfo:
        return SceneInfo(backend=self.name, confidence=0.0)


def build_scene(backend: str, **kw):
    if backend == "vlm":
        return VlmScene(kw["base_url"], kw["model"], kw.get("api_key", "EMPTY"))
    if backend == "heuristic":
        return HeuristicScene()
    return NoScene()

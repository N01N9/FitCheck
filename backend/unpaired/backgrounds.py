"""바닥·침대 합성(flatlay)에 쓸 배경을 klein-4B 로 만든다. 옷·사람이 없는 "위에서 본 표면" 사진.

배경은 입력 쪽 재료라서 생성해도 된다(정답은 여전히 실제 상품 사진). Commons 배경은 방 전경·그림이
많아 옷을 얹으면 떠 보였다. 여기서는 침대 시트·이불·마루·러그·소파 등을 위에서 내려다본 사진만 만든다.

사용 (컨테이너 안)
  python -m unpaired.backgrounds --n 400 --out data/unpaired/backgrounds
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

SURFACES = [
    "an unmade bed with rumpled white cotton sheets", "a bed with a light grey duvet", "a bed with a navy blue duvet",
    "a bed with a beige linen bedspread", "a bed with a striped bedspread", "a bed with a floral quilt",
    "a light oak wooden floor", "a dark walnut wooden floor", "a grey carpet", "a beige shaggy rug",
    "a patterned kilim rug", "white bathroom tiles", "a grey fabric sofa seat", "a brown leather sofa seat",
    "a light wooden table top", "white bed sheets in morning sunlight", "a pink bedspread", "a green velvet blanket",
    "a concrete floor", "a laminate floor with soft window light",
]
LIGHT = ["soft daylight from a window", "warm evening lamp light", "flat overcast light", "bright phone flash"]
PROMPT = ("Top-down phone photo looking straight down at {surface}, {light}. Only the empty surface fills the frame: "
          "no clothes, no people, no hands, no text.")


def prompts(n: int) -> list[str]:
    combos = list(itertools.product(SURFACES, LIGHT))
    return [PROMPT.format(surface=s, light=lt) for s, lt in (combos[i % len(combos)] for i in range(n))]


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="flatlay 배경 생성")
    p.add_argument("--n", type=int, default=400)
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--out", default="data/unpaired/backgrounds")
    args = p.parse_args(argv)

    from unpaired import editors

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = editors.load("klein")
    with (out / "backgrounds.jsonl").open("a") as fh:
        for i, text in enumerate(prompts(args.n)):
            path = out / f"bg{i:05d}.jpg"
            if path.exists():
                continue
            model.pipe(prompt=text, width=args.size, height=args.size, num_inference_steps=4, guidance_scale=1.0,
                       generator=model.torch.Generator("cuda").manual_seed(i)).images[0].save(path, quality=93)
            fh.write(json.dumps({"file": path.name, "prompt": text, "seed": i}) + "\n")


if __name__ == "__main__":
    main()

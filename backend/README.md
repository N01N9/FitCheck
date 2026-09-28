# FitCheck 백엔드 — Phase 0 기술 검증

앱에 들어갈 AI 기술이 실제로 잘 작동하는지 하나씩 확인한다. 각 테스트는 DGX Spark에서 돌리고, 결과(`report.md`, `report.json`, 검수 파일)를 보고 다음 단계로 넘어간다. 기획서: [../docs/PLAN.md](../docs/PLAN.md)

## 테스트 목록

| # | 기술 | 후보 모델 | 합격 기준 | 상태 |
|---|---|---|---|---|
| T01 | 옷 누끼(배경 제거) | BiRefNet (MIT) | 중앙값 1초/장 이내, 눈 검수 합격 90% 이상 | 스크립트 준비됨 |
| T02 | 옷 태깅(종류·색·핏 등) | Qwen3.6-35B-A3B (vLLM) | 카테고리 95%, 주요 속성 85% | 스크립트 준비됨 |
| T03 | 옷 임베딩·유사 옷 찾기 | Marqo-FashionSigLIP | 같은 옷 찾기 상위 1개 정확도 | 예정 |
| T04 | 한 장에 여러 벌 분리 | SAM 3 | 옷 개수·경계 정확도 | 예정 |
| T05 | 사진으로 체형 치수 재기 | SAM 3D Body → MHR | 가슴·허리·엉덩이 평균 오차 3cm 이하 | 예정 |
| T06 | 단품 가상 착용 | FASHN VTON v1.5 | 눈 검수 합격률, 장당 시간 | 예정 |
| T07 | 코디 착용·아바타·옷 복원 | Qwen-Image-Edit | 눈 검수 합격률, 장당 시간 | 예정 |
| T08 | 폰 엣지 모델 | Gemma 4 E2B 등 | 폰에서 속도·정확도 | 예정 |

## Claude가 gx10에서 직접 작업하게 하기

클라우드 세션은 집 네트워크(Tailscale)에 닿지 않으므로, Claude Code를 gx10 위에서 실행하고 Remote Control로 연결한다.

```bash
ssh gx10
curl -fsSL https://claude.ai/install.sh | bash      # Claude Code 설치 (ARM64 Linux 지원)
git clone https://github.com/N01N9/FitCheck.git && cd FitCheck
git checkout claude/clever-bardeen-tzem48
groups | grep -q docker || echo "docker 그룹에 없음: sudo usermod -aG docker $USER 후 재로그인"
tmux new -s fitcheck                                 # SSH가 끊겨도 세션 유지
claude remote-control                                # 처음에는 로그인
```

Claude Code 앱에 새로 뜬 gx10 세션에서 "Phase 0 이어서 진행"이라고 보내면 된다.

## DGX Spark에서 실행하기

Spark는 ARM64라서 PyTorch를 pip로 설치하지 않고 NVIDIA NGC 컨테이너에 들어 있는 것을 쓴다.

```bash
git clone https://github.com/N01N9/FitCheck.git && cd FitCheck/backend
git checkout claude/clever-bardeen-tzem48

# NGC PyTorch 컨테이너 (태그는 NVIDIA DGX Spark 플레이북에 있는 최신 버전으로, 예: 26.05-py3)
docker run --gpus all -it --rm --ipc=host \
  -v "$PWD":/work -w /work \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  nvcr.io/nvidia/pytorch:26.05-py3 bash

# 컨테이너 안에서
pip install -r requirements-phase0.txt
python -m pytest -q tests          # 코드 점검 (GPU 없이도 통과해야 함)
```

## T01 옷 누끼

1. **사진 준비**: `backend/data/garments/`에 내 옷 사진 30장 정도를 넣는다. `data/`는 git에 올라가지 않는다.
   - 쉬운 것과 어려운 것을 섞는다: 바닥에 펼친 옷, 옷걸이에 건 옷, 침대 위(무늬 있는 이불), 흰 옷을 흰 바닥에, 검은 옷을 어두운 바닥에, 끈·레이스·니트처럼 경계가 복잡한 옷.
2. **실행**
   ```bash
   python -m phase0.t01_bg_removal --images data/garments --out results/t01
   # 비교 기준선(고전 방식)
   python -m phase0.t01_bg_removal --images data/garments --out results/t01_border --model border
   ```
3. **검수**: `results/t01/contact_sheet.jpg`를 보고 `results/t01/review.csv`에 장마다 1(합격)/0(불합격)과 메모를 적는다.
4. **공유**: `results/t01/report.md`, `review.csv`, 그리고 불합격 사진 몇 장의 결과를 공유하면 분석해서 다음 조치(해상도 조정, 다른 모델 비교, 촬영 가이드 보완)를 정한다.

선택: 옷 경계를 직접 칠한 정답 마스크가 있으면 `--masks data/garments_masks`로 IoU도 계산한다.

## T02 옷 태깅

VLM을 OpenAI 호환 서버(vLLM 권장)로 띄워 두고 사진을 보낸다. 모델이 스키마(`phase0/garment_schema.py`)에 맞는 JSON만 내도록 강제한다.

1. **모델 서버**: NVIDIA DGX Spark 플레이북의 vLLM 컨테이너로 `Qwen/Qwen3.6-35B-A3B`(또는 NVFP4 버전)를 띄운다. 기본 주소는 `http://localhost:8000/v1`.
2. **정답 라벨**: 처음 실행하면 `results/t02/label_template.csv`가 생긴다. 한국어 값으로 채워 `data/garment_labels.csv`로 저장한다. 허용 값은 `garment_schema.py`에 있고, 모르는 칸은 비워 두면 채점에서 빠진다.
3. **실행**
   ```bash
   python -m phase0.t02_tagging --images data/garments --labels data/garment_labels.csv \
       --base-url http://localhost:8000/v1 --model Qwen/Qwen3.6-35B-A3B --out results/t02
   ```
   T01 누끼 결과(`results/t01/cutouts`)로도 돌려서 배경 제거가 태깅에 도움이 되는지 비교한다.
4. **결과**: 필드별 정확도, 스키마 통과율, 장당 시간, 출력 토큰/초가 `report.md`에 나온다. 틀린 사례는 `predictions.jsonl`에서 본다.

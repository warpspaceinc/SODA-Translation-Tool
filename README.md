# SODA Translation Tool

SODA 데이터셋을 다양한 언어로 번역하는 도구입니다. 원본 [allenai/soda](https://huggingface.co/datasets/allenai/soda) 데이터셋을 LLM을 활용하여 원하는 언어로 번역하고 HuggingFace에 배포할 수 있습니다.

## 라이센스

이 프로젝트는 원본 SODA 데이터셋과 동일한 **CC-BY-4.0 (Creative Commons Attribution 4.0)** 라이센스를 따릅니다.

- 원본 데이터셋: [allenai/soda](https://huggingface.co/datasets/allenai/soda) (CC-BY-4.0)
- 번역 데이터셋: CC-BY-4.0
- 출처 표기 필수: Allen Institute for AI (원본), 번역자 정보

## 프로젝트 구조

```
soda-kr/
├── src/
│   ├── translator.py      # 번역 로직
│   ├── dataset_loader.py  # HuggingFace 데이터셋 로드/저장
│   ├── processor.py       # 청크 단위 병렬 처리
│   └── utils.py          # 유틸리티 함수
├── index.py              # CLI 진입점
├── .env                  # API 키 설정
├── pyproject.toml        # uv 패키지 관리
└── README.md
```

## 설치 및 설정

### 1. uv 설치

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 프로젝트 초기화

```bash
# 가상환경 생성 및 의존성 설치
uv sync

# 또는 수동으로
uv venv
uv pip install -e .
```

### 3. 환경 변수 설정

`.env` 파일을 생성하고 API 키를 설정합니다:

```env
OPENROUTER_API_KEY=your_api_key_here
HF_TOKEN=your_huggingface_token_here
```

## 사용법

### 빠른 시작 (권장)

번역 + 검증 + 업로드를 한 번에 실행:

```bash
# 한국어로 10개 샘플 테스트
uv run python index.py quick-run --model moonshotai/kimi-k2-0905 --samples 10 --target-lang Korean --repo-name username/soda-kr

# 일본어로 100개 샘플 번역
uv run python index.py quick-run --model moonshotai/kimi-k2-0905 --samples 100 --chunk-size 20 --target-lang Japanese --repo-name username/soda-ja

# 중국어로 전체 데이터셋 번역
uv run python index.py quick-run --model moonshotai/kimi-k2-0905 --samples 100000 --chunk-size 20 --target-lang Chinese --repo-name username/soda-zh
```

### 데이터셋 미리보기

```bash
uv run python index.py preview --samples 5
```

### 번역만 실행

```bash
# 기본 번역 (청크 10개씩 동시 처리)
uv run python index.py translate --model moonshotai/kimi-k2-0905 --max-samples 100 --target-lang Korean

# 병렬 처리 속도 향상 (20개씩 동시 처리)
uv run python index.py translate --model moonshotai/kimi-k2-0905 --max-samples 1000 --chunk-size 20 --target-lang Korean

# 다른 언어로 번역
uv run python index.py translate --model moonshotai/kimi-k2-0905 --max-samples 100 --target-lang Japanese
uv run python index.py translate --model moonshotai/kimi-k2-0905 --max-samples 100 --target-lang Spanish
uv run python index.py translate --model moonshotai/kimi-k2-0905 --max-samples 100 --target-lang "Simplified Chinese"

# 출력 경로 지정
uv run python index.py translate --model moonshotai/kimi-k2-0905 --max-samples 100 --target-lang Korean --output ./temp/output/my_translation.json
```

### 번역 결과 검증

```bash
uv run python index.py validate --input ./temp/output/translated.json
```

### HuggingFace에 업로드

```bash
uv run python index.py upload --input ./temp/output/translated.json --repo-name username/soda-kr
```

### 비용 및 시간 추정

전체 번역 전 비용과 시간을 미리 예측:

```bash
# 10개 샘플로 비용 추정 (자동으로 전체 데이터셋 크기 감지)
uv run python index.py estimate --model moonshotai/kimi-k2-0905 --samples 10

# 특정 데이터셋 크기로 추정
uv run python index.py estimate --model moonshotai/kimi-k2-0905 --samples 10 --total-size 100000
```

### 체크포인트에서 번역된 항목 추출 및 업로드

번역 작업이 진행 중이거나 일부 완료된 경우, 이미 번역된 항목만 먼저 업로드할 수 있습니다:

```bash
# 1. 체크포인트에서 번역된 항목 추출
uv run python index.py export-checkpoint --checkpoint ./temp/checkpoints/checkpoint.json --output ./temp/output/translated_partial.json

# 2. 추출된 데이터 검증
uv run python index.py validate --input ./temp/output/translated_partial.json

# 3. HuggingFace에 업로드
uv run python index.py upload --input ./temp/output/translated_partial.json --repo-name username/soda-kr-partial
```

### 중단된 작업 재개

```bash
# translate 명령 중단 시
uv run python index.py resume --checkpoint ./temp/checkpoints/checkpoint.json --model moonshotai/kimi-k2-0905 --chunk-size 20 --target-lang Korean

# quick-run 명령 중단 시 (다른 언어도 가능)
uv run python index.py resume --checkpoint ./temp/checkpoints/quick_run.json --model moonshotai/kimi-k2-0905 --chunk-size 20 --target-lang Japanese
```

## CLI 명령어

| 명령어 | 설명 | 주요 옵션 |
|--------|------|-----------|
| `quick-run` | 번역 + 검증 + 업로드 한번에 | `--model`, `--samples`, `--chunk-size`, `--target-lang`, `--repo-name` |
| `preview` | 데이터셋 샘플 확인 | `--samples` |
| `translate` | 번역만 실행 | `--model`, `--max-samples`, `--chunk-size`, `--target-lang`, `--output` |
| `estimate` | 비용 및 시간 추정 | `--model`, `--samples`, `--total-size` |
| `export-checkpoint` | 체크포인트에서 번역된 항목 추출 | `--checkpoint`, `--output` |
| `validate` | 번역 결과 검증 | `--input` |
| `upload` | HuggingFace 업로드 | `--input`, `--repo-name`, `--private` |
| `resume` | 중단된 작업 재개 | `--checkpoint`, `--model`, `--chunk-size`, `--target-lang` |

## 주요 기능

- ✅ **진짜 병렬 처리**: `asyncio.gather()`로 청크 내 모든 샘플을 동시에 번역 (4-5배 속도 향상)
- ✅ **체크포인트 시스템**: 중단되어도 `resume` 명령으로 이어서 작업 가능
- ✅ **다국어 지원**: `--target-lang` 옵션으로 Korean, Japanese, English 등 다양한 언어로 번역
- ✅ **자동 재시도**: API 오류 시 지수 백오프로 자동 재시도 (최대 3회)
- ✅ **검증 기능**: 업로드 전 데이터 구조 및 품질 자동 검증
- ✅ **진행률 표시**: tqdm으로 실시간 진행 상황 모니터링
- ✅ **빠른 실행**: `quick-run` 명령으로 번역부터 업로드까지 원스톱

## 개발

```bash
# 개발 모드 실행
uv run python index.py --help

# 의존성 추가
uv add package-name

# 의존성 업데이트
uv sync
```

## 참조

- 원본 데이터셋: https://huggingface.co/datasets/allenai/soda
- OpenRouter API: https://openrouter.ai/

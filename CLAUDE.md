# CLAUDE.md - 개발자 가이드

이 문서는 Claude Code를 사용하여 프로젝트를 개발하는 개발자를 위한 가이드입니다.

## 프로젝트 개요

**목적**: SODA 대화 데이터셋을 한국어로 번역하여 HuggingFace에 공개

**핵심 요구사항**:
- 원본 데이터셋: `allenai/soda` (CC-BY-4.0 라이센스)
- 번역 모델: OpenRouter API를 통한 다양한 LLM
- 병렬 처리: 10개 청크씩 동시 번역
- 패키지 관리: uv (가상환경 + 의존성 관리)
- 라이센스: CC-BY-4.0 (원본과 동일)

## 아키텍처 설계

### 1. 모듈 구조

```
src/
├── translator.py       # 번역 API 호출 및 재시도 로직
├── dataset_loader.py   # HuggingFace datasets 로드/저장
├── processor.py        # 청크 분할 및 병렬 처리 (asyncio/concurrent.futures)
└── utils.py           # 비용 계산, 검증, 체크포인트 관리
```

### 2. 데이터 흐름

```
[HuggingFace Load] → [Filter & Preprocess] → [Chunk (n=10)] → [Parallel Translate]
                                                                       ↓
[HuggingFace Upload] ← [Validate] ← [Checkpoint Save] ← [Merge Results]
```

### 3. 핵심 함수 설계

#### `translator.py`

```python
async def translate_json_async(
    input_json: dict,
    model: str,
    target_lang: str = "Korean",
    max_retries: int = 3
) -> tuple[str, dict]:
    """
    비동기 번역 함수
    - 반환: (request_id, translated_dict)
    - 재시도 로직 포함
    - rate limit 처리
    """
```

#### `processor.py`

```python
async def process_chunk(
    chunk: List[Dict],
    model: str,
    session: aiohttp.ClientSession,
    target_lang: str = "Korean",
    checkpoint_manager: CheckpointManager = None,
    pbar: tqdm = None
) -> List[Dict]:
    """
    청크 단위 병렬 처리
    - asyncio.gather(*tasks)로 청크 내 모든 샘플을 진짜 동시 번역
    - 각 항목 완료 시 체크포인트 자동 저장
    - 에러 발생 시 해당 항목만 스킵하고 계속 진행
    """

async def process_dataset(
    samples: List[Dict],
    model: str,
    chunk_size: int = 10,
    checkpoint_path: str = "./temp/checkpoints/checkpoint.json",
    target_lang: str = "Korean"
) -> List[Dict]:
    """
    전체 데이터셋 처리
    - 청크 분할 및 순차적으로 청크 실행 (각 청크 내부는 병렬)
    - 이미 번역된 항목은 자동으로 건너뛰기
    - 진행률 표시 (tqdm)
    - 청크 간 1초 대기 (rate limit 방지)
    """

def run_translation(
    samples: List[Dict],
    model: str,
    chunk_size: int = 10,
    checkpoint_path: str = "./temp/checkpoints/checkpoint.json",
    target_lang: str = "Korean"
) -> List[Dict]:
    """
    동기 버전 번역 실행 (asyncio.run 래퍼)
    - CLI에서 호출하기 쉽도록 동기 함수로 제공
    """
```

#### `dataset_loader.py`

```python
def load_soda_dataset(split: str = "train", cache_dir: str = "./temp/hf_cache") -> Dataset:
    """SODA 데이터셋 로드 및 필터링"""

def save_to_huggingface(
    dataset: Dataset,
    repo_name: str,
    license: str = "cc-by-4.0",
    token: str = None
):
    """HuggingFace Hub에 업로드"""
```

#### `utils.py`

```python
class CheckpointManager:
    """체크포인트 저장/로드 관리"""
    def save(self, idx: int, data: dict)
    def load(self, checkpoint_path: str) -> tuple[int, list[dict]]
    def cleanup(self)

def estimate_cost(model: str, sample_size: int) -> dict:
    """비용 추정 (샘플 번역 후 extrapolate)"""

def validate_dataset(dataset: Dataset) -> dict:
    """번역 품질 및 구조 검증"""
```

## CLI 구현 세부사항

### index.py 구조

```python
import click
from src.processor import process_dataset
from src.dataset_loader import load_soda_dataset, save_to_huggingface
from src.utils import CheckpointManager, estimate_cost, validate_dataset

@click.group()
def cli():
    """SODA-KR Translation CLI"""
    pass

@cli.command()
@click.option("--samples", default=5, help="Number of samples to preview")
def preview(samples):
    """데이터셋 샘플 확인"""
    pass

@cli.command()
@click.option("--model", required=True, help="Translation model")
@click.option("--chunk-size", default=10, help="Chunk size (동시 처리 개수)")
@click.option("--max-samples", default=None, type=int, help="최대 샘플 수")
@click.option("--target-lang", default="Korean", help="대상 언어")
@click.option("--checkpoint", default="./temp/checkpoints/checkpoint.json")
@click.option("--output", default="./temp/output/translated.json")
def translate(model, chunk_size, max_samples, target_lang, checkpoint, output):
    """번역만 실행"""
    pass

@cli.command()
@click.option("--input", "input_path", required=True)
def validate(input_path):
    """번역 결과 검증"""
    pass

@cli.command()
@click.option("--input", "input_path", required=True)
@click.option("--repo-name", required=True)
@click.option("--private", is_flag=True)
def upload(input_path, repo_name, private):
    """HuggingFace 업로드"""
    pass

@cli.command()
@click.option("--checkpoint", required=True)
@click.option("--model", required=True)
@click.option("--chunk-size", default=10)
@click.option("--target-lang", default="Korean")
@click.option("--output", default="./temp/output/translated.json")
def resume(checkpoint, model, chunk_size, target_lang, output):
    """중단된 작업 재개"""
    pass

@cli.command()
@click.option("--model", default="moonshotai/kimi-k2-0905")
@click.option("--samples", default=10)
@click.option("--chunk-size", default=10)
@click.option("--target-lang", default="Korean")
@click.option("--repo-name", default="CaveduckAI/simplified_soda_kr")
def quick_run(model, samples, chunk_size, target_lang, repo_name):
    """빠른 실행: 번역 + 검증 + 업로드를 한번에"""
    pass

if __name__ == "__main__":
    cli()
```

## 중요 고려사항

### 1. 라이센스 준수

- **CC-BY-4.0 요구사항**:
  - ✅ 원본 출처 명시 (allenai/soda)
  - ✅ 라이센스 명시
  - ✅ 변경 사항 표시 (번역 정보)
  - ✅ 동일 라이센스 유지 (CC-BY-4.0)

- **HuggingFace 업로드 시 메타데이터**:
```python
dataset_card = """
# SODA-KR

## Dataset Description
Korean translation of the SODA dataset.

## Source Dataset
- Original: allenai/soda
- License: CC-BY-4.0
- Citation: [SODA paper]

## Translation
- Translated by: [Your Name]
- Translation Model: [Model Name]
- Date: [Date]

## License
CC-BY-4.0 (same as original)
"""
```

### 2. 병렬 처리 전략 (실제 구현)

**중요**: `asyncio.gather(*tasks)`를 사용하여 진짜 동시 처리 구현

```python
# process_chunk 함수 내부
async def process_chunk(chunk, model, session, target_lang, ...):
    # 1. 모든 작업 생성
    tasks = []
    for sample in chunk:
        task = translate_json_async(to_translate, model, target_lang, session)
        tasks.append(task)

    # 2. 모든 작업을 동시에 실행 (핵심!)
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # 3. 결과 처리
    for idx, response in zip(indices, responses):
        if isinstance(response, Exception):
            # 에러 처리
        else:
            # 성공 처리
```

**성능**:
- 순차 처리: 10개 샘플 약 54-77초
- 병렬 처리: 10개 샘플 약 15초 (4-5배 향상)

### 3. 체크포인트 포맷

```json
{
  "version": "1.0",
  "model": "qwen/qwen3-next-80b-a3b-instruct",
  "last_index": 1234,
  "total_items": 10000,
  "start_time": "2025-10-06T00:00:00",
  "translated_data": [
    {"index": 0, "original": {...}, "translated": {...}, "request_id": "..."},
    {"index": 1, "original": {...}, "translated": {...}, "request_id": "..."}
  ],
  "stats": {
    "total_cost": 12.34,
    "avg_duration": 2.5,
    "errors": 3
  }
}
```

### 4. 에러 처리

```python
# 재시도 로직
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(aiohttp.ClientError)
)
async def translate_with_retry(...):
    pass

# Rate limit 처리
from asyncio import Semaphore
semaphore = Semaphore(10)  # 최대 10개 동시 요청

async with semaphore:
    result = await translate_json_async(...)
```

## uv 패키지 관리

### pyproject.toml

```toml
[project]
name = "soda-kr"
version = "0.1.0"
description = "Korean translation of SODA dataset"
requires-python = ">=3.10"
license = {text = "CC-BY-4.0"}
dependencies = [
    "datasets>=2.14.0",
    "requests>=2.31.0",
    "python-dotenv>=1.0.0",
    "click>=8.1.0",
    "aiohttp>=3.9.0",
    "tqdm>=4.66.0",
    "tenacity>=8.2.0",
    "huggingface-hub>=0.19.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]

[project.scripts]
soda-kr = "index:cli"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 명령어

```bash
# 초기 설정
uv init
uv add datasets requests python-dotenv click aiohttp tqdm tenacity huggingface-hub

# 개발 의존성
uv add --dev pytest black ruff

# 실행
uv run python index.py translate --model qwen/qwen3-next-80b-a3b-instruct
```

## 개발 워크플로우

1. **초기 샘플 테스트**: `preview` 명령으로 데이터 확인
2. **비용 추정**: `estimate` 명령으로 전체 비용 예측
3. **소규모 번역**: `--start 0 --end 100`으로 100개만 테스트
4. **검증**: `validate` 명령으로 품질 확인
5. **전체 번역**: 체크포인트 활용하여 점진적 처리
6. **최종 검증 및 업로드**: 라이센스 정보 포함하여 HuggingFace에 업로드

## 성능 목표

- **처리 속도**: 청크 10개 병렬 처리로 순차 대비 10배 향상
- **안정성**: 체크포인트로 언제든 재개 가능
- **비용 효율**: 사전 추정으로 예산 관리


import os
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import HfApi, create_repo


def load_soda_dataset(split: str = "train", cache_dir: str = "./temp/hf_cache") -> Dataset:
    """
    SODA 데이터셋 로드 및 필터링

    Args:
        split: 데이터셋 분할 ("train", "test", "validation")
        cache_dir: 캐시 디렉토리

    Returns:
        필터링된 Dataset
    """
    ds = load_dataset("allenai/soda", split=split, cache_dir=cache_dir)
    return ds


def generate_filtered_samples(dataset: Dataset, max_samples: int = None):
    """
    데이터셋에서 유효한 샘플만 필터링하여 생성

    Args:
        dataset: 원본 데이터셋
        max_samples: 최대 샘플 수 (None이면 전체)

    Yields:
        필터링된 샘플 딕셔너리
    """
    count = 0
    for idx in range(len(dataset)):
        data = dataset[idx]
        # 화자 최소 2명, 대화 최소 1개, narrative 필수
        if len(data["speakers"]) < 2 or len(data["dialogue"]) < 1:
            continue
        narrative = data["narrative"] or ""
        if narrative == "":
            continue

        yield {
            "index": idx,
            "speakers": data["speakers"][:2],
            "narrative": data["narrative"],
            "dialogue": data["dialogue"][:1]
        }

        count += 1
        if max_samples and count >= max_samples:
            break


def create_dataset_from_samples(samples: list) -> Dataset:
    """
    샘플 리스트로부터 Dataset 생성

    Args:
        samples: 샘플 딕셔너리 리스트

    Returns:
        Dataset 객체
    """
    # 데이터 구조 변환
    data_dict = {
        "index": [],
        "speakers": [],
        "narrative": [],
        "dialogue": []
    }

    for sample in samples:
        data_dict["index"].append(sample["index"])
        data_dict["speakers"].append(sample["speakers"])
        data_dict["narrative"].append(sample["narrative"])
        data_dict["dialogue"].append(sample["dialogue"])

    return Dataset.from_dict(data_dict)


def save_to_huggingface(
    dataset: Dataset,
    repo_name: str,
    token: str = None,
    private: bool = False
):
    """
    HuggingFace Hub에 데이터셋 업로드

    Args:
        dataset: 업로드할 데이터셋
        repo_name: 레포지토리 이름 (예: "username/dataset-name")
        token: HuggingFace 토큰 (None이면 환경변수 사용)
        private: 비공개 여부
    """
    if token is None:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ValueError("HF_TOKEN 환경변수를 설정하거나 token 파라미터를 제공하세요")

    # 레포지토리 생성 (이미 존재하면 무시)
    try:
        create_repo(
            repo_id=repo_name,
            repo_type="dataset",
            token=token,
            private=private,
            exist_ok=True
        )
    except Exception as e:
        print(f"Repository creation note: {e}")

    # 데이터셋 카드 생성
    dataset_card = f"""---
license: cc-by-4.0
task_categories:
- text-generation
language:
- ko
tags:
- dialogue
- translation
- korean
- conversational
size_categories:
- n<1K
---

# SODA-KR (Simplified)

Korean translation of the SODA dataset (simplified version with {len(dataset)} samples).

## Dataset Description

This is a Korean-translated version of the [allenai/soda](https://huggingface.co/datasets/allenai/soda) dataset.
Each sample contains speakers, narrative context, and dialogue translated to Korean.

## Source Dataset

- **Original Dataset**: [allenai/soda](https://huggingface.co/datasets/allenai/soda)
- **Original License**: CC-BY-4.0
- **Citation**: Please cite the original SODA paper

## Translation Details

- **Translation Model**: moonshotai/kimi-k2-0905
- **Translation Date**: 2025-10-06
- **Translated by**: CaveduckAI
- **Number of Samples**: {len(dataset)}

## Dataset Structure

```python
{{
    "index": int,           # Original dataset index
    "speakers": list[str],  # List of speaker names (max 2)
    "narrative": str,       # Narrative context in Korean
    "dialogue": list[str]   # Dialogue lines in Korean (first line only)
}}
```

## License

CC-BY-4.0 (same as original dataset)

## Attribution

This dataset is a derivative work of SODA by Allen Institute for AI, used under CC-BY-4.0.
"""

    # 데이터셋 업로드
    dataset.push_to_hub(
        repo_id=repo_name,
        token=token,
        private=private,
        commit_message="Upload translated SODA dataset"
    )

    # README 업로드
    api = HfApi()
    api.upload_file(
        path_or_fileobj=dataset_card.encode(),
        path_in_repo="README.md",
        repo_id=repo_name,
        repo_type="dataset",
        token=token,
        commit_message="Add dataset card"
    )

    print(f"Dataset uploaded to https://huggingface.co/datasets/{repo_name}")

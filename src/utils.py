import os
import json
import time
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set


class CheckpointManager:
    """체크포인트 관리 클래스"""

    def __init__(self, checkpoint_path: str, model: str):
        """
        Args:
            checkpoint_path: 체크포인트 파일 경로
            model: 사용 중인 모델명
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.model = model
        self.data = self._load_or_create()

    def _load_or_create(self) -> dict:
        """체크포인트 파일 로드 또는 새로 생성"""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 모델이 다르면 새로 시작
                if data.get("model") != self.model:
                    print(f"⚠️  Model changed from {data.get('model')} to {self.model}. Starting fresh.")
                    return self._create_new()
                return data
        else:
            return self._create_new()

    def _create_new(self) -> dict:
        """새 체크포인트 데이터 생성"""
        # 디렉토리 생성
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        return {
            "version": "1.0",
            "model": self.model,
            "start_time": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "items": []
        }

    def save_item(self, item: dict):
        """번역된 항목 저장"""
        # 기존 항목 업데이트 또는 추가
        existing_indices = {i["index"]: idx for idx, i in enumerate(self.data["items"])}

        if item["index"] in existing_indices:
            self.data["items"][existing_indices[item["index"]]] = item
        else:
            self.data["items"].append(item)

        self.data["last_updated"] = datetime.now().isoformat()
        self._save()

    def _save(self):
        """체크포인트 파일에 저장"""
        with open(self.checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    def get_processed_indices(self) -> Set[int]:
        """처리된 인덱스 집합 반환"""
        return {item["index"] for item in self.data["items"]}

    def load_all_items(self) -> List[Dict]:
        """모든 저장된 항목 반환"""
        return self.data["items"]

    def get_stats(self) -> dict:
        """통계 정보 반환"""
        items = self.data["items"]
        errors = [item for item in items if "error" in item]

        return {
            "total_processed": len(items),
            "errors": len(errors),
            "success": len(items) - len(errors),
            "model": self.model,
            "start_time": self.data.get("start_time"),
            "last_updated": self.data.get("last_updated")
        }

    def cleanup(self):
        """체크포인트 파일 삭제"""
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()


def validate_dataset(items: List[Dict]) -> dict:
    """
    번역된 데이터셋 검증

    Args:
        items: 검증할 항목 리스트

    Returns:
        검증 결과 딕셔너리
    """
    total = len(items)
    errors = []
    warnings = []

    for item in items:
        idx = item.get("index", "unknown")

        # 필수 필드 체크
        required_fields = ["speakers", "narrative", "dialogue"]
        for field in required_fields:
            if field not in item:
                errors.append(f"Index {idx}: Missing field '{field}'")

        # 에러 항목 체크
        if "error" in item:
            errors.append(f"Index {idx}: Translation error - {item['error']}")

        # speakers 검증
        if "speakers" in item:
            if not isinstance(item["speakers"], list):
                errors.append(f"Index {idx}: 'speakers' is not a list")
            elif len(item["speakers"]) < 2:
                warnings.append(f"Index {idx}: Less than 2 speakers")

        # narrative 검증
        if "narrative" in item:
            if not isinstance(item["narrative"], str):
                errors.append(f"Index {idx}: 'narrative' is not a string")
            elif len(item["narrative"]) == 0:
                warnings.append(f"Index {idx}: Empty narrative")

        # dialogue 검증
        if "dialogue" in item:
            if not isinstance(item["dialogue"], list):
                errors.append(f"Index {idx}: 'dialogue' is not a list")
            elif len(item["dialogue"]) == 0:
                warnings.append(f"Index {idx}: Empty dialogue")

    result = {
        "total": total,
        "errors": len(errors),
        "warnings": len(warnings),
        "error_details": errors[:10],  # 최대 10개만
        "warning_details": warnings[:10],
        "valid": len(errors) == 0
    }

    return result


def print_validation_result(result: dict):
    """검증 결과 출력"""
    print("\n" + "="*50)
    print("Validation Result")
    print("="*50)
    print(f"Total items: {result['total']}")
    print(f"Valid: {result['valid']}")
    print(f"Errors: {result['errors']}")
    print(f"Warnings: {result['warnings']}")

    if result['error_details']:
        print("\nError details:")
        for error in result['error_details']:
            print(f"  - {error}")

    if result['warning_details']:
        print("\nWarning details:")
        for warning in result['warning_details']:
            print(f"  - {warning}")

    print("="*50 + "\n")


def get_generation_cost(generation_id: str) -> float:
    """
    OpenRouter API에서 특정 generation의 비용 조회

    Args:
        generation_id: OpenRouter generation ID (예: "gen-xxx")

    Returns:
        비용 (USD)
    """
    time.sleep(1.0)  # Rate limit 방지
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY 환경변수를 설정하세요")

    url = "https://openrouter.ai/api/v1/generation"
    headers = {"Authorization": f"Bearer {api_key}"}
    params = {"id": generation_id}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        total_cost = response.json()['data']['total_cost']
        return total_cost
    except Exception as e:
        print(f"Warning: Could not get cost for {generation_id}: {e}")
        return 0.0


def estimate_translation_cost(
    model: str,
    sample_count: int,
    total_dataset_size: int
) -> dict:
    """
    샘플 번역 후 전체 비용 추정

    Args:
        model: 사용할 모델명
        sample_count: 번역한 샘플 수
        total_dataset_size: 전체 데이터셋 크기

    Returns:
        비용 추정 정보 딕셔너리
    """
    from src.dataset_loader import load_soda_dataset, generate_filtered_samples
    from src.processor import run_translation
    import time as time_module

    print(f"Estimating cost for model: {model}")
    print(f"Sample size: {sample_count}")
    print(f"Total dataset size: {total_dataset_size}\n")

    # 샘플 데이터 로드
    dataset = load_soda_dataset()
    samples = list(generate_filtered_samples(dataset, max_samples=sample_count))

    print(f"Translating {len(samples)} samples for cost estimation...")

    # 샘플 번역 실행
    start_time = time_module.time()
    checkpoint_path = "./temp/checkpoints/estimate.json"

    translated = run_translation(
        samples=samples,
        model=model,
        chunk_size=min(10, sample_count),
        checkpoint_path=checkpoint_path,
        target_lang="Korean"
    )
    duration = time_module.time() - start_time

    # 비용 계산 (request_id에서 generation_id 추출)
    costs = []
    for item in translated:
        if "request_id" in item and item["request_id"]:
            cost = get_generation_cost(item["request_id"])
            if cost > 0:
                costs.append(cost)

    if not costs:
        return {
            "error": "Could not retrieve cost information",
            "sample_count": sample_count,
            "model": model
        }

    # 통계 계산
    import statistics
    avg_cost = statistics.mean(costs)
    total_estimated_cost = avg_cost * total_dataset_size
    avg_duration = duration / len(samples)
    total_estimated_duration_hours = (avg_duration * total_dataset_size) / 3600

    result = {
        "model": model,
        "sample_count": len(samples),
        "total_dataset_size": total_dataset_size,
        "sample_costs": costs,
        "avg_cost_per_sample": avg_cost,
        "total_estimated_cost_usd": total_estimated_cost,
        "avg_duration_per_sample_seconds": avg_duration,
        "total_estimated_duration_hours": total_estimated_duration_hours,
        "sample_duration_seconds": duration
    }

    return result


def print_cost_estimation(result: dict):
    """비용 추정 결과 출력"""
    if "error" in result:
        print(f"\nError: {result['error']}")
        return

    print("\n" + "="*60)
    print("Cost Estimation Result")
    print("="*60)
    print(f"Model: {result['model']}")
    print(f"Sample count: {result['sample_count']}")
    print(f"Total dataset size: {result['total_dataset_size']}")
    print()
    print(f"Average cost per sample: ${result['avg_cost_per_sample']:.6f}")
    print(f"Total estimated cost: ${result['total_estimated_cost_usd']:.2f}")
    print()
    print(f"Average duration per sample: {result['avg_duration_per_sample_seconds']:.2f} seconds")
    print(f"Total estimated duration: {result['total_estimated_duration_hours']:.2f} hours")
    print(f"Sample translation took: {result['sample_duration_seconds']:.2f} seconds")
    print("="*60 + "\n")

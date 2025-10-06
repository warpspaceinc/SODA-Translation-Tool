import asyncio
import aiohttp
from typing import List, Dict
from tqdm import tqdm
from src.translator import translate_json_async
from src.utils import CheckpointManager


async def process_chunk(
    chunk: List[Dict],
    model: str,
    session: aiohttp.ClientSession,
    target_lang: str = "Korean",
    checkpoint_manager: CheckpointManager = None,
    pbar: tqdm = None
) -> List[Dict]:
    """
    청크 단위로 샘플들을 병렬 번역

    Args:
        chunk: 번역할 샘플 리스트
        model: 사용할 모델명
        session: aiohttp 세션
        target_lang: 대상 언어
        checkpoint_manager: 체크포인트 관리자
        pbar: 진행률 표시 바

    Returns:
        번역된 샘플 리스트
    """
    # 모든 샘플에 대한 비동기 작업 생성
    tasks = []
    indices = []
    for sample in chunk:
        # 번역할 데이터 추출
        to_translate = {
            "speakers": sample["speakers"],
            "narrative": sample["narrative"],
            "dialogue": sample["dialogue"]
        }
        task = translate_json_async(to_translate, model, target_lang=target_lang, session=session)
        tasks.append(task)
        indices.append(sample["index"])

    # 모든 작업을 동시에 실행 (진짜 병렬 처리)
    responses = await asyncio.gather(*tasks, return_exceptions=True)

    # 결과 처리
    results = []
    for idx, response in zip(indices, responses):
        if isinstance(response, Exception):
            # 에러 발생
            print(f"\nError translating index {idx}: {response}")
            result = {
                "index": idx,
                "speakers": [],
                "narrative": "",
                "dialogue": [],
                "error": str(response)
            }
        else:
            # 성공
            req_id, translated = response
            result = {
                "index": idx,
                "speakers": translated.get("speakers", []),
                "narrative": translated.get("narrative", ""),
                "dialogue": translated.get("dialogue", []),
                "request_id": req_id
            }

        results.append(result)

        # 체크포인트 저장
        if checkpoint_manager:
            checkpoint_manager.save_item(result)

        # 진행률 업데이트
        if pbar:
            pbar.update(1)

    return results


async def process_dataset(
    samples: List[Dict],
    model: str,
    chunk_size: int = 10,
    checkpoint_path: str = "./temp/checkpoints/checkpoint.json",
    target_lang: str = "Korean"
) -> List[Dict]:
    """
    전체 데이터셋 병렬 처리

    Args:
        samples: 번역할 샘플 리스트
        model: 사용할 모델명
        chunk_size: 청크 크기 (동시 처리 개수)
        checkpoint_path: 체크포인트 저장 경로
        target_lang: 대상 언어

    Returns:
        번역된 전체 샘플 리스트
    """
    checkpoint_manager = CheckpointManager(checkpoint_path, model)

    # 이미 번역된 항목 확인
    processed_indices = checkpoint_manager.get_processed_indices()
    remaining_samples = [s for s in samples if s["index"] not in processed_indices]

    if not remaining_samples:
        print("All samples already translated. Loading from checkpoint...")
        return checkpoint_manager.load_all_items()

    print(f"Total samples: {len(samples)}")
    print(f"Already processed: {len(processed_indices)}")
    print(f"Remaining: {len(remaining_samples)}")
    print(f"Model: {model}")
    print(f"Chunk size: {chunk_size}")

    # 청크로 분할
    chunks = [
        remaining_samples[i:i + chunk_size]
        for i in range(0, len(remaining_samples), chunk_size)
    ]

    all_results = []

    # 진행률 표시
    with tqdm(total=len(remaining_samples), desc="Translating") as pbar:
        async with aiohttp.ClientSession() as session:
            for chunk in chunks:
                results = await process_chunk(
                    chunk,
                    model,
                    session,
                    target_lang,
                    checkpoint_manager,
                    pbar
                )
                all_results.extend(results)

                # 청크 처리 후 잠시 대기 (rate limit 방지)
                await asyncio.sleep(1)

    # 기존 처리된 항목과 합치기
    existing_items = checkpoint_manager.load_all_items()
    all_results.extend([item for item in existing_items if item["index"] not in [r["index"] for r in all_results]])

    # 인덱스 순으로 정렬
    all_results.sort(key=lambda x: x["index"])

    print(f"\nTranslation completed! Total: {len(all_results)} samples")
    return all_results


def run_translation(
    samples: List[Dict],
    model: str,
    chunk_size: int = 10,
    checkpoint_path: str = "./temp/checkpoints/checkpoint.json",
    target_lang: str = "Korean"
) -> List[Dict]:
    """
    동기 버전 번역 실행 (asyncio.run 래퍼)

    Args:
        samples: 번역할 샘플 리스트
        model: 사용할 모델명
        chunk_size: 청크 크기
        checkpoint_path: 체크포인트 경로
        target_lang: 대상 언어

    Returns:
        번역된 샘플 리스트
    """
    return asyncio.run(process_dataset(samples, model, chunk_size, checkpoint_path, target_lang))

import os
import json
import asyncio
import aiohttp
from typing import Tuple, Dict
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError))
)
async def translate_json_async(
    input_json: dict,
    model: str,
    target_lang: str = "Korean",
    session: aiohttp.ClientSession = None
) -> Tuple[str, dict]:
    """
    비동기 JSON 번역 함수

    Args:
        input_json: 번역할 JSON 데이터
        model: 사용할 모델명
        target_lang: 대상 언어
        session: aiohttp 세션 (None이면 새로 생성)

    Returns:
        (request_id, translated_dict) 튜플
    """
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY 환경변수를 설정하세요")

    input_str = json.dumps(input_json, ensure_ascii=False, indent=2)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a JSON translator. Translate only the values, keep all keys unchanged. Return valid JSON only."
            },
            {
                "role": "user",
                "content": f"Translate all values in this JSON to {target_lang}:\n\n{input_str}. do not change order of arrays."
            }
        ],
        "temperature": 0.3,
        "response_format": {"type": "json_object"}
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True

    try:
        async with session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=60)
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise Exception(f"API Error: {response.status} - {text}")

            result = await response.json()
            req_id = result['id']
            translated = json.loads(result["choices"][0]["message"]["content"])
            return req_id, translated
    finally:
        if close_session:
            await session.close()


def translate_json_sync(
    input_json: dict,
    model: str,
    target_lang: str = "Korean"
) -> Tuple[str, dict]:
    """
    동기 버전 번역 함수 (기존 코드 호환성)
    """
    return asyncio.run(translate_json_async(input_json, model, target_lang))

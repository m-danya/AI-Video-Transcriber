import asyncio
import functools
import logging
import os
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

DEFAULT_MAX_PARALLEL_LLM_REQUESTS = 4


def _read_max_parallel_llm_requests() -> int:
    raw_value = (os.getenv("MAX_PARALLEL_LLM_REQUESTS") or "").strip()
    if not raw_value:
        return DEFAULT_MAX_PARALLEL_LLM_REQUESTS

    try:
        value = int(raw_value)
    except ValueError:
        logger.warning(
            "MAX_PARALLEL_LLM_REQUESTS=%r is invalid, using %s",
            raw_value,
            DEFAULT_MAX_PARALLEL_LLM_REQUESTS,
        )
        return DEFAULT_MAX_PARALLEL_LLM_REQUESTS

    if value < 1:
        logger.warning(
            "MAX_PARALLEL_LLM_REQUESTS=%r must be >= 1, using %s",
            raw_value,
            DEFAULT_MAX_PARALLEL_LLM_REQUESTS,
        )
        return DEFAULT_MAX_PARALLEL_LLM_REQUESTS

    return value


MAX_PARALLEL_LLM_REQUESTS = _read_max_parallel_llm_requests()
_llm_request_semaphore: Optional[asyncio.Semaphore] = None
_llm_request_loop: Optional[asyncio.AbstractEventLoop] = None


def get_max_parallel_llm_requests() -> int:
    return MAX_PARALLEL_LLM_REQUESTS


def _get_llm_request_semaphore() -> asyncio.Semaphore:
    global _llm_request_loop
    global _llm_request_semaphore

    loop = asyncio.get_running_loop()
    if _llm_request_semaphore is None or _llm_request_loop is not loop:
        _llm_request_loop = loop
        _llm_request_semaphore = asyncio.Semaphore(MAX_PARALLEL_LLM_REQUESTS)
        logger.info(
            "LLM request parallelism limit: %s",
            MAX_PARALLEL_LLM_REQUESTS,
        )
    return _llm_request_semaphore


async def run_llm_request(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    semaphore = _get_llm_request_semaphore()
    call = functools.partial(func, *args, **kwargs)

    async with semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, call)


async def create_chat_completion(client: Any, **kwargs: Any) -> Any:
    return await run_llm_request(client.chat.completions.create, **kwargs)

import asyncio
import json
import logging
from asyncio import Task
from typing import Any, Callable, Dict, List

import aiohttp
from steamship import SteamshipError
from tenacity import stop_after_attempt, wait_exponential_jitter, before_sleep_log, retry_if_exception_type, after_log, \
    retry


async def _json_post(session: aiohttp.ClientSession, url: str, body: Dict, service_name: str) -> Task:

    @retry(
        reraise=True,
        stop=stop_after_attempt(8),
        wait=wait_exponential_jitter(jitter=5),
        before_sleep=before_sleep_log(logging.root, logging.INFO),
        retry=(
                retry_if_exception_type(SteamshipError)
        ),
        after=after_log(logging.root, logging.INFO),
    )
    async def _inner_json_post():
        async with session.post(url, data=json.dumps(body)) as resp:
            if not resp.ok:
                raise SteamshipError(
                    message=f"Request to {service_name} failed. URL={url}, Code={resp.status}. Body={await resp.text()}"
                )

            output = await resp.json()
            if not output:
                raise SteamshipError(
                    message=f"Request from {service_name} could not be interpreted as JSON. URL={url}"
                )
            return output

    result = _inner_json_post()
    logging.info("Retry statistics: " + json.dumps(_inner_json_post.retry.statistics))
    return await result



def list_batches(l: List, batch_size: int):
    """Chunk a list into batches of size `batch_size`."""
    for i in range(0, len(l), batch_size):
        yield l[i:i + batch_size]


async def async_concurrent_json_posts(
        url: str,
        headers: Dict,
        items: List[Any],
        batch_size: int,
        items_to_body: Callable[[List[Any]], Dict],
        service_name: str
) -> List[Dict]:
    """Helper function around a concurrent set of JSON->JSON posts.

    * The list of items is split into batches of size `batch_size`
    * Each batch is transformed into a post body
    * Those post bodies are concurrently run as json_post(url, headers, body)
    """
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = []
        for batch in list_batches(items, batch_size):
            body = items_to_body(batch)
            tasks.append(asyncio.ensure_future(_json_post(session, url, body, service_name)))

        result_bodies = await asyncio.gather(*tasks)
        return result_bodies

def concurrent_json_posts(
        url: str,
        headers: Dict,
        items: List[Any],
        batch_size: int,
        items_to_body: Callable[[List[Any]], Dict],
        service_name: str
) -> List[Dict]:
    return asyncio.run(async_concurrent_json_posts(url, headers, items, batch_size, items_to_body, service_name))

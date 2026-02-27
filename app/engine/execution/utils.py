import logging
import asyncio
import functools
from typing import Dict, Any, List, Optional
import httpx

logger = logging.getLogger(__name__)

def async_retry(max_retries: int = 3, initial_delay: float = 0.1, backoff_factor: float = 2.0):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except httpx.HTTPStatusError as e:
                    if 400 <= e.response.status_code < 500:
                        raise e
                    last_exception = e
                except (httpx.RequestError, Exception) as e:
                    last_exception = e
                
                if attempt < max_retries - 1:
                    logger.warning(f"Retry attempt {attempt + 1}/{max_retries} for {func.__name__} after {delay}s due to {last_exception}")
                    await asyncio.sleep(delay)
                    delay *= backoff_factor
            
            logger.error(f"All {max_retries} retries failed for {func.__name__}")
            raise last_exception
        return wrapper
    return decorator

class PaginationHandler:
    @staticmethod
    async def auto_fetch_all(client: httpx.AsyncClient, url: str, headers: Dict[str, str], params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        all_items = []
        current_url = url
        
        while current_url:
            response = await client.get(current_url, headers=headers, params=params if current_url == url else None)
            response.raise_for_status()
            data = response.json()
            
            items = data.get("items", []) or data.get("invoices", [])
            all_items.extend(items)
            
            links = data.get("links", [])
            next_link = next((l["href"] for l in links if l["rel"] == "next"), None)
            current_url = next_link
            
        return all_items

class SanitizationLayer:
    SENSITIVE_KEYS = ["email_address", "phone_number", "billing_info", "payer_info", "client_secret", "access_token"]

    @staticmethod
    def sanitize(data: Any) -> Any:
        if isinstance(data, list):
            return [SanitizationLayer.sanitize(item) for item in data]
        if isinstance(data, dict):
            return {
                k: (SanitizationLayer.sanitize(v) if k not in SanitizationLayer.SENSITIVE_KEYS else "[REDACTED]")
                for k, v in data.items()
            }
        return data

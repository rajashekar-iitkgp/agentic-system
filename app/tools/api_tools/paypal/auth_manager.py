import httpx
import time
import logging
from typing import Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class PayPalAuthManager:
    def __init__(self):
        self.client_id = settings.PAYPAL_CLIENT_ID
        self.client_secret = settings.PAYPAL_CLIENT_SECRET
        self.mode = settings.PAYPAL_MODE
        self.base_url = "https://api-m.sandbox.paypal.com" if self.mode == "sandbox" else "https://api-m.paypal.com"
        
        self._access_token: Optional[str] = None
        self._expires_at: float = 0
        
    async def get_access_token(self) -> str:
        if self._access_token and time.time() < self._expires_at - 60:
            return self._access_token
            
        return await self.refresh_access_token()

    async def refresh_access_token(self) -> str:
        if not self.client_id or not self.client_secret:
            raise ValueError("PayPal credentials are not configured in .env")

        url = f"{self.base_url}/v1/oauth2/token"
        headers = {
            "Accept": "application/json",
            "Accept-Language": "en_US",
        }
        data = {"grant_type": "client_credentials"}
        
        logger.info("Refreshing PayPal Access Token...")
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, 
                auth=(self.client_id, self.client_secret),
                headers=headers,
                data=data
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to refresh PayPal token: {response.text}")
                response.raise_for_status()
                
            token_data = response.json()
            self._access_token = token_data["access_token"]
            self._expires_at = time.time() + token_data["expires_in"]
            
            logger.info("Successfully refreshed PayPal Access Token.")
            return self._access_token
paypal_auth = PayPalAuthManager()
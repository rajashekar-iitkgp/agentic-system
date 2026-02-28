import httpx
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class PayPalAuthManager:
    def __init__(self):
        self.access_token = None
        
    async def get_access_token(self):
        if self.access_token:
            return self.access_token
        
        logger.info("Refreshing PayPal Access Token...")
        url = "https://api-m.sandbox.paypal.com/v1/oauth2/token"
        if settings.PAYPAL_MODE == "live":
             url = "https://api-m.paypal.com/v1/oauth2/token"
             
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    url,
                    data={"grant_type": "client_credentials"},
                    auth=(settings.PAYPAL_CLIENT_ID, settings.PAYPAL_CLIENT_SECRET),
                    timeout=10.0
                )
                response.raise_for_status()
                data = response.json()
                self.access_token = data["access_token"]
                logger.info("Successfully refreshed PayPal Access Token.")
                return self.access_token
            except Exception as e:
                logger.error(f"PayPal Auth Error: {e}")
                raise

paypal_auth = PayPalAuthManager()

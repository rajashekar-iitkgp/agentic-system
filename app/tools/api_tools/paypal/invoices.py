import httpx
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from app.tools.api_tools.paypal.auth_manager import paypal_auth
from app.engine.execution.utils import async_retry, SanitizationLayer, PaginationHandler

logger = logging.getLogger(__name__)

class CreateInvoiceSchema(BaseModel):
    customer_email: str = Field(..., description="The email address of the customer to send the invoice to.")
    currency_code: str = Field(default="USD", description="The currency code (e.g., USD, EUR).")
    amount_value: float = Field(..., description="The total amount of the invoice.")
    item_name: str = Field(..., description="A short name for the product or service.")

class SendInvoiceSchema(BaseModel):
    invoice_id: str = Field(..., description="The unique ID of the invoice to send.")

class CancelInvoiceSchema(BaseModel):
    invoice_id: str = Field(..., description="The unique ID of the invoice to cancel.")
    reason: Optional[str] = Field(default="Canceled by system", description="The reason for canceling the invoice.")

@async_retry()
async def create_paypal_invoice(customer_email: str, amount_value: float, item_name: str, currency_code: str = "USD") -> Dict[str, Any]:
    token = await paypal_auth.get_access_token()
    url = f"{paypal_auth.base_url}/v2/invoicing/invoices"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "detail": {
            "currency_code": currency_code
        },
        "invoicer": {
            "email_address": "merchant@example.com"
        },
        "primary_recipients": [{"billing_info": {"email_address": customer_email}}],
        "items": [{
            "name": item_name,
            "quantity": "1",
            "unit_amount": {"currency_code": currency_code, "value": str(amount_value)}
        }]
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return SanitizationLayer.sanitize(response.json())

@async_retry()
async def send_paypal_invoice(invoice_id: str) -> Dict[str, Any]:
    token = await paypal_auth.get_access_token()
    url = f"{paypal_auth.base_url}/v2/invoicing/invoices/{invoice_id}/send"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers)
        response.raise_for_status()
        return {"status": "SENT", "invoice_id": invoice_id}

@async_retry()
async def list_paypal_invoices() -> List[Dict[str, Any]]:
    token = await paypal_auth.get_access_token()
    url = f"{paypal_auth.base_url}/v2/invoicing/invoices"
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with httpx.AsyncClient() as client:
        return await PaginationHandler.auto_fetch_all(client, url, headers)

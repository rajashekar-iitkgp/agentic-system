import httpx
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from app.tools.api_tools.paypal.auth_manager import paypal_auth
from app.engine.execution.utils import async_retry, SanitizationLayer, PaginationHandler

logger = logging.getLogger(__name__)

class AcceptClaimSchema(BaseModel):
    dispute_id: str = Field(..., description="The unique ID of the dispute to accept.")

class ProvideEvidenceSchema(BaseModel):
    dispute_id: str = Field(..., description="The unique ID of the dispute.")
    evidence_type: str = Field(..., description="The type of evidence (e.g., PROOF_OF_FULFILLMENT).")
    notes: str = Field(..., description="Supporting documentation or notes.")

@async_retry()
async def list_paypal_disputes() -> List[Dict[str, Any]]:
    token = await paypal_auth.get_access_token()
    url = f"{paypal_auth.base_url}/v1/customer/disputes"
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with httpx.AsyncClient() as client:
        return await PaginationHandler.auto_fetch_all(client, url, headers)

@async_retry()
async def provide_dispute_evidence(dispute_id: str, evidence_type: str, notes: str) -> Dict[str, Any]:
    token = await paypal_auth.get_access_token()
    url = f"{paypal_auth.base_url}/v1/customer/disputes/{dispute_id}/provide-evidence"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "evidence_type": evidence_type,
        "notes": notes
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return {"status": "EVIDENCE_SUBMITTED", "dispute_id": dispute_id}

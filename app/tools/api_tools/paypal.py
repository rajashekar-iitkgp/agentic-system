import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def paypal_create_invoice(amount: float, customer_id: str) -> Dict[str, Any]:
    logger.info(f"Mock: Creating PayPal invoice for {customer_id} with amount {amount}")
    return {
        "status": "success",
        "invoice_id": f"INV-{customer_id[:3]}-882",
        "message": f"Successfully created invoice for {amount}"
    }

async def paypal_send_invoice(invoice_id: str) -> Dict[str, Any]:
    logger.info(f"Mock: Sending PayPal invoice {invoice_id}")
    return {
        "status": "success",
        "message": f"Invoice {invoice_id} has been dispatched to customer email."
    }

async def paypal_get_sales_volume(start_date: str, end_date: str) -> Dict[str, Any]:
    logger.info(f"Mock: Fetching sales volume from {start_date} to {end_date}")
    return {
        "status": "success",
        "total_volume": 125000.50,
        "currency": "USD",
        "period": f"{start_date} - {end_date}"
    }

async def paypal_check_dispute_status(transaction_id: str = None, customer_id: str = None) -> Dict[str, Any]:
    logger.info(f"Mock: Checking dispute status for transaction {transaction_id} or customer {customer_id}")
    return {
        "status": "success",
        "has_dispute": False,
        "details": "No open disputes found for the given identifiers."
    }

PAYPAL_TOOL_MAP = {
    "paypal_create_invoice": paypal_create_invoice,
    "paypal_send_invoice": paypal_send_invoice,
    "paypal_get_sales_volume": paypal_get_sales_volume,
    "paypal_check_dispute_status": paypal_check_dispute_status
}

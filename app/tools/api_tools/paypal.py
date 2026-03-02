import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

async def paypal_create_invoice(amount: float, customer_id: str) -> Dict[str, Any]:
    """
    Creates a new PayPal invoice for a specific customer. 
    IMPORTANT: Only 'amount' and 'customer_id' are required. 
    Do NOT ask the user for email, product name, or description; proceed immediately with these two fields.
    Args:
        amount: The total amount to be invoiced (e.g., 50.0).
        customer_id: The unique identifier of the customer (e.g., CUST-88).
    """
    logger.info(f"Mock: Creating PayPal invoice for {customer_id} with amount {amount}")
    return {
        "status": "success",
        "invoice_id": f"INV-{customer_id[:3]}-882",
        "message": f"Successfully created invoice for {amount}"
    }

async def paypal_send_invoice(invoice_id: str) -> Dict[str, Any]:
    """
    Dispatches a created invoice to the customer's email.
    IMPORTANT: Only 'invoice_id' is required. Do not ask for other details.
    Args:
        invoice_id: The ID of the invoice to send (e.g., INV-CUS-882).
    """
    logger.info(f"Mock: Sending PayPal invoice {invoice_id}")
    return {
        "status": "success",
        "message": f"Invoice {invoice_id} has been dispatched to customer email."
    }

async def paypal_get_sales_volume(start_date: str, end_date: str) -> Dict[str, Any]:
    """
    Retrieves the total sales volume for a given date range.
    Args:
        start_date: Start of the period (YYYY-MM-DD).
        end_date: End of the period (YYYY-MM-DD).
    """
    logger.info(f"Mock: Fetching sales volume from {start_date} to {end_date}")
    return {
        "status": "success",
        "total_volume": 125000.50,
        "currency": "USD",
        "period": f"{start_date} - {end_date}"
    }

async def paypal_check_dispute_status(transaction_id: str = None, customer_id: str = None) -> Dict[str, Any]:
    """
    Checks if there are any open disputes for a transaction or customer.
    Args:
        transaction_id: Optional transaction ID.
        customer_id: Optional customer ID.
    """
    logger.info(f"Mock: Checking dispute status for transaction {transaction_id} or customer {customer_id}")
    return {
        "status": "success",
        "has_dispute": False,
        "details": "No open disputes found for the given identifiers."
    }


async def paypal_get_shipping_address(order_id: str) -> Dict[str, Any]:
    """
    Retrieves the shipping address for a given PayPal order.

    This is a mocked helper for testing the agentic system. It does not call the real
    PayPal API but returns a representative shipping address payload so that we can
    validate tool selection, RBAC, and execution.

    Args:
        order_id: The PayPal order ID (e.g., 1DA59471B5379105V).
    """
    logger.info(f"Mock: Fetching shipping address for PayPal order {order_id}")
    return {
        "status": "success",
        "order_id": order_id,
        "shipping_address": {
            "full_name": "FooBuyer Jones",
            "address_line_1": "1 Main St",
            "city": "San Jose",
            "state": "CA",
            "postal_code": "95131",
            "country_code": "US",
        },
    }

PAYPAL_TOOL_MAP = {
    "paypal_create_invoice": paypal_create_invoice,
    "paypal_send_invoice": paypal_send_invoice,
    "paypal_get_sales_volume": paypal_get_sales_volume,
    "paypal_check_dispute_status": paypal_check_dispute_status,
    "paypal_get_shipping_address": paypal_get_shipping_address,
}

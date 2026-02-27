import asyncio
import logging
from typing import List

from app.models.tool import ToolMetadata
from app.tools.registry import registry_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_mock_paypal_tools() -> List[ToolMetadata]:
    tools = []
    tools.append(ToolMetadata(
        name="paypal_create_invoice",
        domain="payments",
        description="Creates a new invoice in PayPal for a customer with a specified amount and currency.",
        tags=["invoice", "create", "finance"],
        input_schema={"type": "object", "properties": {"amount": {"type": "number"}, "customer_id": {"type": "string"}}, "required": ["amount", "customer_id"]}
    ))
    tools.append(ToolMetadata(
        name="paypal_send_invoice",
        domain="payments",
        description="Dispatches an existing drafted invoice to the customer's email address.",
        tags=["invoice", "send", "email"],
        input_schema={"type": "object", "properties": {"invoice_id": {"type": "string"}}, "required": ["invoice_id"]}
    ))
    
    # Reporting Tools
    tools.append(ToolMetadata(
        name="paypal_get_sales_volume",
        domain="reporting",
        description="Calculates the total sales volume processed over a specific date range (e.g., 'last month').",
        tags=["sales", "volume", "report", "analytics"],
        input_schema={"type": "object", "properties": {"start_date": {"type": "string"}, "end_date": {"type": "string"}}, "required": ["start_date", "end_date"]}
    ))
    
    # Dispute Tools
    tools.append(ToolMetadata(
        name="paypal_check_dispute_status",
        domain="disputes",
        description="Checks if a specific customer or transaction has an open dispute or chargeback.",
        tags=["dispute", "chargeback", "status", "risk"],
        input_schema={"type": "object", "properties": {"transaction_id": {"type": "string"}, "customer_id": {"type": "string"}}}
    ))
    
    # System Tools (Just to show domain pollution)
    tools.append(ToolMetadata(
        name="system_get_tool_list",
        domain="system",
        description="Lists all the available capabilities and tools the agent has access to.",
        tags=["system", "tools", "capabilities"],
        input_schema={"type": "object", "properties": {}}
    ))
    
    # In a real scenario, we loop to generate 50+ unique descriptive tools here.
    # For this script, we'll auto-generate generic ones to hit the 50 mark.
    for i in range(5, 55):
        tools.append(ToolMetadata(
            name=f"paypal_mock_api_{i}",
            domain="payments",
            description=f"A generic mock PayPal API for handling low-level ledger transaction type {i}. Use this when specifically manipulating ledger {i}.",
            tags=["mock", "ledger"],
            input_schema={"type": "object", "properties": {"ledger_id": {"type": "string"}}}
        ))
        
    return tools

async def main():
    logger.info("Starting Tool Seeding Process...")
    mock_tools = generate_mock_paypal_tools()
    
    logger.info(f"Generated {len(mock_tools)} mock tools. Initiating registry upload.")
    await registry_manager.seed_tools(mock_tools)
    logger.info("Seeding complete. Vector Store is now populated.")

if __name__ == "__main__":
    asyncio.run(main())

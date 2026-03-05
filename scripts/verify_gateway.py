import asyncio
import logging
from app.engine.execution.gateway import execution_gateway
from app.engine.state import AgentState
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockSchema(BaseModel):
    amount: float
    target: str

async def test_gateway():
    print("\n--- Testing Execution Gateway ---")
    
    state: AgentState = {
        "trace_id": "test-trace-123",
        "request_metadata": {"role": "support"},
        "messages": [],
        "next_agent": None,
        "user_intent": None,
        "active_domain": "payments",
        "retrieved_tools": [],
        "tool_error": None
    }
    
    try:
        print("Test 1: Support role calling 'create_paypal_invoice'...")
        execution_gateway.validate_tool_call("create_paypal_invoice", {"amount": 10}, MockSchema, state)
    except ValueError as e:
        print(f"Blocked as expected: {e}")

    state["request_metadata"]["role"] = "merchant"
    try:
        print("Test 2: Merchant role calling with invalid args (amount as string)...")
        execution_gateway.validate_tool_call("create_paypal_invoice", {"amount": "not_a_number"}, MockSchema, state)
    except ValueError as e:
        print(f"Validated failure as expected: {e}")

    try:
        print("Test 3: Merchant role calling with valid args...")
        execution_gateway.validate_tool_call("create_paypal_invoice", {"amount": 50.0, "target": "user_1"}, MockSchema, state)
        print("Success as expected.")
    except Exception as e:
        print(f"Failed unexpectedly: {e}")

if __name__ == "__main__":
    asyncio.run(test_gateway())

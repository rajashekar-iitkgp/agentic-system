import asyncio
import logging
from typing import List, Dict

from app.engine.routing.semantic_router import router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EVAL_DATASET = [
    {
        "query": "I need to send out an invoice to customer abc@gmail for $150.",
        "expected_tool": "paypal_create_invoice",
        "domain_filter": "payments"
    },
    {
        "query": "Can you check if there are any chargebacks open for transaction TXN-123?",
        "expected_tool": "paypal_check_dispute_status",
        "domain_filter": "payments"
    },
    {
        "query": "What were my total sales last month?",
        "expected_tool": "paypal_get_sales_volume",
        "domain_filter": "reporting"
    },
    {
        "query": "Email that draft invoice to the client.",
        "expected_tool": "paypal_send_invoice",
        "domain_filter": "payments"
    }
]

async def run_tool_selection_eval():
    logger.info(f"Starting Tool Selection Evaluation over {len(EVAL_DATASET)} test queries.")
    total = len(EVAL_DATASET)
    successes = 0
    k = 5

    for item in EVAL_DATASET:
        query = item["query"]
        target = item["expected_tool"]
        domain = item["domain_filter"]
        
        try:
            retrieved = await router.retrieve_tools_for_intent(user_query=query, domain_filter=domain, k=k)
            retrieved_names = [t["name"] for t in retrieved]
            
            if target in retrieved_names:
                successes += 1
                logger.info(f"[PASS] Query: '{query}' -> Found '{target}' in Top-{k}.")
            else:
                logger.error(f"[FAIL] Query: '{query}' -> Expected '{target}' but got {retrieved_names}.")
                
        except Exception as e:
            logger.error(f"[ERROR] Evaluation failed on query '{query}': {e}")
            
    accuracy = (successes / total) * 100
    logger.info("====================================")
    logger.info(f"EVALUATION COMPLETE: {accuracy:.2f}% Accuracy (Top-{k} Retrieval)")
    logger.info("====================================")

if __name__ == "__main__":
    asyncio.run(run_tool_selection_eval())

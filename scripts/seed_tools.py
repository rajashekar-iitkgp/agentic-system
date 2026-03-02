import asyncio
import logging
from app.tools.registry import registry_manager
from app.models.tool import ToolMetadata
from app.tools.api_tools.paypal import PAYPAL_TOOL_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os

async def seed():
    # Clear old index to prevent "ghost" tools from prior runs
    # for f in ["data/hnsw_index.faiss", "data/tool_metadata.pkl"]:
    #     if os.path.exists(f):
    #         os.remove(f)
    #         logger.info(f"Deleted old index file: {f}")
            
    tools_to_seed = []
    
    for name, tool in PAYPAL_TOOL_MAP.items():
        domain = "payments"
        if "dispute" in name:
            domain = "disputes"
        elif "sales" in name or "volume" in name:
            domain = "reporting"
            
        tools_to_seed.append(ToolMetadata(
            name=name,
            description=tool.__doc__ or "No description provided.",
            domain=domain,
            tags=["paypal", domain],
            input_schema={} # Simplified for now as we use docstrings for OpenAI
        ))
    
    await registry_manager.seed_tools(tools_to_seed)
    logger.info("Universal Seed Completed.")

if __name__ == "__main__":
    asyncio.run(seed())

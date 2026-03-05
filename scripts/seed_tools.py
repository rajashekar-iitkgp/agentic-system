import asyncio
import logging
from app.tools.registry import registry_manager
from app.models.tool import ToolMetadata
from app.tools.api_tools.paypal import PAYPAL_TOOL_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os

async def seed():
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
            input_schema={} 
        ))
    
    await registry_manager.seed_tools(tools_to_seed)
    logger.info("Universal Seed Completed.")

if __name__ == "__main__":
    asyncio.run(seed())

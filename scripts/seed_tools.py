import asyncio
import logging
from app.tools.registry import registry_manager
from app.models.tool import ToolMetadata
from app.tools.api_tools.paypal import PAYPAL_TOOL_MAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def seed():
    tools_to_seed = []
    
    for name, tool in PAYPAL_TOOL_MAP.items():
        domain = "payments"
        if "dispute" in name:
            domain = "disputes"
            
        tools_to_seed.append(ToolMetadata(
            name=name,
            description=tool.description,
            domain=domain,
            tags=["paypal", domain],
            input_schema=tool.args_schema.schema() if hasattr(tool, "args_schema") and tool.args_schema else {}
        ))
    
    await registry_manager.seed_tools(tools_to_seed)
    logger.info("Universal Seed Completed.")

if __name__ == "__main__":
    asyncio.run(seed())

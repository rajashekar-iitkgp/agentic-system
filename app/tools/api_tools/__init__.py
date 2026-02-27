from .paypal import PAYPAL_TOOL_MAP

# Combine all tool domains into a single master map for the ToolExecutionAgent
ALL_TOOLS = {
    **PAYPAL_TOOL_MAP,
    # Add other domains here as they are implemented
}

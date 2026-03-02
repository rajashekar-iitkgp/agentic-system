from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class ToolMetadata(BaseModel):
    name: str = Field(..., description="The unique, programmatic name of the tool (e.g., 'paypal_create_invoice').")
    description: str = Field(..., description="A highly descriptive explanation of what the tool does. This is used for semantic search.")
    domain: str = Field(..., description="The broad domain the tool belongs to (e.g., 'payments', 'system', 'documentation').")
    tags: List[str] = Field(default_factory=list, description="Specific tags for hard filtering (e.g., ['finance', 'invoice', 'write']).")
    input_schema: Dict[str, Any] = Field(..., description="The JSON Schema representation of the expected arguments for this tool.")
    embedding: Optional[List[float]] = Field(default=None, description="The pre-computed vector embedding of the tool's description/name. Used for semantic retrieval.")
    compensating_tool: Optional[str] = Field(
        default=None,
        description="Optional name of a compensating (rollback) tool to call if workflows involving this tool fail.",
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "paypal_create_invoice",
                "description": "Creates a new invoice in PayPal for a specific customer amount.",
                "domain": "payments",
                "tags": ["paypal", "invoice", "create"],
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number"},
                        "customer_id": {"type": "string"}
                    },
                    "required": ["amount", "customer_id"]
                }
            }
        }

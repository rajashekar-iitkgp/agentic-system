import logging
import redis
from typing import Dict, Any, Optional, List
from pydantic import ValidationError
from app.core.config import settings
from app.engine.state import AgentState

logger = logging.getLogger(__name__)

try:
    redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    redis_client.ping()
except Exception as e:
    logger.warning(f"Failed to connect to Redis: {e}. Falling back to in-memory idempotency.")
    redis_client = None

MEM_IDEMPOTENCY = set()

class ExecutionGateway:
    def __init__(self, rbac_rules: Optional[Dict[str, List[str]]] = None):
        self.rbac_rules = rbac_rules or {
            "admin": ["*"],
            "merchant": ["payment", "invoice", "dispute", "reporting", "refund", "sales", "volume"],
            "support": ["reporting", "dispute", "sales", "volume"]
        }


    async def check_idempotency(self, request_id: str) -> bool:
        if redis_client:
            try:
                is_new = redis_client.set(f"idempotency:{request_id}", "EXEC", ex=86400, nx=True)
                return bool(is_new)
            except Exception as e:
                logger.error(f"Redis error during idempotency check: {e}")
        
        if request_id in MEM_IDEMPOTENCY:
            return False
        MEM_IDEMPOTENCY.add(request_id)
        return True

    def validate_tool_call(self, tool_name: str, args: Dict[str, Any], tool_schema: Any, state: AgentState) -> bool:
        user_metadata = state.get("request_metadata", {})
        user_role = user_metadata.get("role", "merchant")
        
        if not self._check_rbac(user_role, tool_name):
            logger.warning(f"RBAC Blocked: User {user_role} attempted to call {tool_name}")
            raise ValueError(f"Security Policy: Access to tool '{tool_name}' is denied for role '{user_role}'.")

        try:
            if tool_schema:
                tool_schema(**args)
            logger.info(f"Schema validated for tool {tool_name}")
        except ValidationError as e:
            logger.error(f"Schema validation failed for tool {tool_name}: {e}")
            raise ValueError(f"Invalid Arguments for {tool_name}: {str(e)}")

        logger.info(f"Audit: User={user_role}, Tool={tool_name}, TraceID={state.get('trace_id')}, Params={args}")
        
        return True

    def _check_rbac(self, role: str, tool_name: str) -> bool:
        allowed = self.rbac_rules.get(role, [])
        if "*" in allowed:
            return True
        for pattern in allowed:
            if pattern in tool_name:
                return True
        return False
         
execution_gateway = ExecutionGateway()

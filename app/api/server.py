import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from app.core.config import settings
from app.engine.graph import build_graph
import traceback
from app.tools.api_tools.auth_manager import paypal_auth

import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from app.db.db_logger import save_response_to_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from app.tools.api_tools import ALL_TOOLS

try:
    agent_graph = build_graph(ALL_TOOLS)
except Exception as e:
    logger.error(f"CRITICAL: Failed to build agent graph: {e}")
    raise

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.PROJECT_NAME} in async mode.")
    
    try:
        logger.info("Authenticating PayPal Developer credentials...")
        await paypal_auth.get_access_token()
        logger.info("PayPal authentication successful.")
    except Exception as e:
        logger.error(f"Failed to authenticate PayPal Developer credentials: {e}")
        
    if agent_graph._async_pool:
        await agent_graph._async_pool.open()
        logger.info("AsyncConnectionPool opened.")
    yield
    if agent_graph._async_pool:
        await agent_graph._async_pool.close()
        logger.info("AsyncConnectionPool closed.")
    logger.info("Shutting down API.")

app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    session_id: str

class ChatResponse(BaseModel):
    response: str
    active_domain: str
    retrieved_tools: List[str]

class TracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        trace_id = request.headers.get("X-Trace-ID", str(uuid.uuid4()))
        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers["X-Trace-ID"] = trace_id
        return response
        
app.add_middleware(TracingMiddleware)


def _infer_domain_from_text(text: str) -> str:
    lowered = (text or "").lower()
    if any(w in lowered for w in ["dispute", "chargeback", "case"]):
        return "disputes"
    if any(w in lowered for w in ["sales", "volume", "report", "revenue"]):
        return "reporting"
    return "payments"


def _coerce_message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # Gemini/LangChain can return structured content parts like:
    # [{'type': 'text', 'text': '...'}, ...]
    if isinstance(content, list):
        parts: List[str] = []
        for p in content:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                parts.append(p["text"])
            elif isinstance(p, str):
                parts.append(p)
        if parts:
            return "\n".join(parts)
    return str(content)

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, fastapi_request: Request):
    try:
        trace_id = fastapi_request.state.trace_id
        inferred_domain = _infer_domain_from_text(request.message)
        initial_state = {
            "messages": [HumanMessage(content=request.message)],
            "trace_id": trace_id,
            "request_metadata": {
                "session_id": request.session_id,
                "role": "merchant"
            },
            "user_intents": [request.message],
            "active_domain": inferred_domain,
            "retrieved_tools": [],
            "next_agent": "supervisor",
            "requires_reseed": False,
            "tool_error": None
        }
        
        # Use a per-request thread_id so each HTTP call runs in an isolated graph
        # thread, avoiding cross-request shortcut behavior from previous runs.
        thread_id = f"{request.session_id}:{trace_id}"
        final_state = await agent_graph.ainvoke(
            initial_state,
            config={"configurable": {"thread_id": thread_id}},
        )
        
        last_message = final_state["messages"][-1]
        
        raw_content = last_message.content if hasattr(last_message, "content") else str(last_message)
        content = _coerce_message_content_to_text(raw_content)
        domain = final_state.get("active_domain", "unknown")

        retrieved_tools_state = final_state.get("retrieved_tools", [])
        if not retrieved_tools_state:
            try:
                from app.engine.routing.semantic_router import router
                retrieved_tools_state = await router.retrieve_tools_for_intent(
                    request.message,
                    domain_filter=domain if domain and domain != "unknown" else "payments",
                    k=3,
                )
            except Exception as e:
                logger.error(f"Fallback semantic routing failed: {e}")
                retrieved_tools_state = []

        retrieved_tools = [t["name"] for t in retrieved_tools_state]

        if final_state.get("openai_quota_error"):
            quota_msg = (
                "The system identified the relevant tools for your request, "
                "but cannot complete the operation right now because the underlying LLM API quota "
                "or rate limit for this environment has been exceeded. "
                "Please try again later or update the API quota."
            )
            content = quota_msg
        
        logger.info(f"Final response sent to user: {content[:100]}...")
        
        # Mirror response to PostgreSQL in the background
        import asyncio
        asyncio.create_task(save_response_to_db(
            session_id=request.session_id,
            user_query=request.message,
            agent_response=content,
            domain=domain,
            tools=retrieved_tools
        ))

        return ChatResponse(
            response=content,
            active_domain=domain,
            retrieved_tools=retrieved_tools
        )
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"GRAPH ERROR: {e}\n{error_trace}")
        raise HTTPException(status_code=500, detail=f"Graph Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "ok", "system": settings.PROJECT_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.server:app", host="127.0.0.1", port=8000, reload=True)

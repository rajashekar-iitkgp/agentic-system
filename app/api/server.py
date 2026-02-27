import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from app.core.config import settings
from app.engine.graph import build_graph

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
    yield
    logger.info("Shutting down API.")

app = FastAPI(title=settings.PROJECT_NAME, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        initial_state = {"messages": [HumanMessage(content=request.message)]}
        final_state = await agent_graph.ainvoke(initial_state)
        last_message = final_state["messages"][-1]
        
        content = last_message.content if hasattr(last_message, "content") else str(last_message)
        retrieved_tools = [t["name"] for t in final_state.get("retrieved_tools", [])]
        domain = final_state.get("active_domain", "unknown")
        
        return ChatResponse(
            response=content,
            active_domain=domain,
            retrieved_tools=retrieved_tools
        )
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"Error executing graph: {e}\n{error_trace}")
        raise HTTPException(status_code=500, detail="Internal Server Error during execution.")

@app.get("/health")
async def health_check():
    return {"status": "ok", "system": settings.PROJECT_NAME}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.server:app", host="127.0.0.1", port=8000, reload=True)

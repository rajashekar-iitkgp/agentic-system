import os
import logging
from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from app.core.config import settings

logger = logging.getLogger(__name__)

class ToolEmbeddings(OpenAIEmbeddings):
    def __init__(self, model: str = "text-embedding-3-small"):
        super().__init__(
            model=model,
            openai_api_key=settings.OPENAI_API_KEY
        )


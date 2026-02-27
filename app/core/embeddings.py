import os
import logging
from typing import List, Optional
from google import genai
from langchain_core.embeddings import Embeddings
from app.core.config import settings

logger = logging.getLogger(__name__)

class GeminiEmbeddings(Embeddings):
    def __init__(self, model_name: str = "models/gemini-embedding-001"):
        self.api_key = settings.GEMINI_API_KEY
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        try:
            response = self.client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            return response.embeddings[0].values
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)

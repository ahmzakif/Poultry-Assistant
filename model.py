from __future__ import annotations

import os
from functools import lru_cache
from typing import Final

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.embeddings.base import Embeddings

import dotenv
dotenv.load_dotenv()

_MODEL_NAME: Final = "gemma-3-12b-it"
_EMBED_MODEL: Final = "models/embedding-001"


def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise ValueError(
            f"Environment variable {name!r} belum di-set. "
            "Export terlebih dahulu sebelum menjalankan aplikasi."
        )
    return val


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Mengembalikan instansi **tunggal** ChatGoogleGenerativeAI."""
    api_key = _require_env("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(model=_MODEL_NAME, api_key=api_key)


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Mengembalikan instansi **tunggal** Embedding Google."""
    _require_env("GOOGLE_API_KEY")  # validasi konsisten
    return GoogleGenerativeAIEmbeddings(model=_EMBED_MODEL)

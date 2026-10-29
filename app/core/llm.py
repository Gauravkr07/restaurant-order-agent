"""
Shared LLM client. One instance, built from validated settings
(see app/core/config.py) rather than reading os.environ here.

The API key is passed explicitly rather than left to ChatGroq's own
os.environ lookup: settings loads it from .env, which does not populate
the process environment, so the implicit lookup would fail even with a
perfectly valid key on disk.
"""
from langchain_groq import ChatGroq
from pydantic import SecretStr

from app.core.config import settings

llm = ChatGroq(
    model=settings.llm_model,
    api_key=SecretStr(settings.groq_api_key),
    temperature=settings.llm_temperature,
    streaming=True,
    timeout=settings.llm_timeout_seconds,
)

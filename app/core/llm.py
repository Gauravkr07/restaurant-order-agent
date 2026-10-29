"""
Shared LLM client. One instance, built from validated settings
(see app/core/config.py) rather than reading os.environ here.
"""
from langchain_groq import ChatGroq

from app.core.config import settings

llm = ChatGroq(
    model=settings.llm_model,
    temperature=settings.llm_temperature,
    streaming=True,
    timeout=settings.llm_timeout_seconds,
)

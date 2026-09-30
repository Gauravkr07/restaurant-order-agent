from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

LLM_TIMEOUT_SECONDS = 15

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    streaming=True,
    timeout=LLM_TIMEOUT_SECONDS,
)

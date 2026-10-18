import os

from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ["OPENROUTER_MODEL"]

"""
PROBLEM:
    `OpenAI` is LangChain's legacy TEXT-COMPLETION wrapper.
    It sends requests to the `/completions` endpoint using a single
    `prompt` string as the payload.

    However, the OpenRouter model configured in `OPENROUTER_MODEL`
    only supports the CHAT-COMPLETIONS endpoint (`/chat/completions`),
    which expects a `messages` list instead of a plain `prompt`.

    Since LangChain's `OpenAI` class cannot target `/chat/completions`,
    OpenRouter rejects the request and returns:

        "Input required: specify 'prompt'"

    This is NOT a Python syntax error, a bad API key, or a bad model name.
    It is an API SHAPE MISMATCH: wrong endpoint for the chosen model.

FIXES (pick one):
    1. Use `ChatOpenAI` instead of `OpenAI` (recommended).
    2. Pick an OpenRouter model that still supports `/completions`.
    3. Skip LangChain and call `/chat/completions` directly via `requests`.
"""

client = OpenAI(
    api_key=API_KEY,
    model=MODEL
)

response = client.invoke("What is the capital of France ?")

print(response)
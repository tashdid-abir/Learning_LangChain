import os

from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ["OPENROUTER_MODEL"]

"""
PROBLEMS IN THIS CODE:

1) WRONG CLASS FOR THE PROVIDER (the 404 / AnthropicModelNotFoundError)
   -----------------------------------------------------------------
   `ChatAnthropic` is Anthropic's official LangChain wrapper. It ONLY
   knows how to talk to Anthropic's own servers, using Anthropic's
   native API path:

       POST https://api.anthropic.com/v1/messages

   You are pointing it at OpenRouter via `base_url="https://openrouter.ai/api/v1"`.
   Even with the custom base_url, `ChatAnthropic` still appends its own
   Anthropic-style path (`/messages`) and sends Anthropic-shaped payloads.
   OpenRouter does not serve that endpoint, so it returns a 404 HTML page,
   which the Anthropic SDK cannot parse as JSON and surfaces as:

       anthropic.NotFoundError: ... 404: Not Found ...
       langchain_anthropic.chat_models.AnthropicModelNotFoundError

   The huge HTML dump in the traceback is just OpenRouter's 404 page.

   Also, you are passing `OPENROUTER_API_KEY` as the Anthropic API key.
   Anthropic's servers would reject that key too — so this is a
   double mismatch: wrong key AND wrong endpoint for this class.

   FIX (pick one):
     A. Use Anthropic directly:
            from langchain_anthropic import ChatAnthropic
            client = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                api_key=os.environ["ANTHROPIC_API_KEY"],
                temperature=0,
                max_tokens=10
            )
        (No base_url pointing at OpenRouter.)

     B. Keep OpenRouter but use the OpenAI-compatible class:
            from langchain_openai import ChatOpenAI
            client = ChatOpenAI(
                model=os.environ["OPENROUTER_MODEL"],   # e.g. "anthropic/claude-3.5-sonnet"
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
                temperature=0,
                max_tokens=10
            )

2) WRONG PARAMETER NAME: `max_token` (should be `max_tokens`)
   ----------------------------------------------------------
   You wrote `max_token=10`. The real parameter is `max_tokens`
   (plural). Because `max_token` is not a recognized argument,
   LangChain shoves it into `model_kwargs` and forwards it to the
   provider, which then raises:

       TypeError: Messages.create() got an unexpected keyword argument 'max_token'

   (Earlier you hit the same class of bug with `max_completion_tokens`,
   which is an OpenAI name — Anthropic only understands `max_tokens`.)

   FIX: rename `max_token=10`  ->  `max_tokens=10`.

3) (Related, already fixed earlier) `max_completion_tokens`
   --------------------------------------------------------
   If you ever see this warning:
       "max_completion_tokens is not default parameter ...
        transferred to model_kwargs"
   it means the same thing: that name belongs to OpenAI, not Anthropic.
   Anthropic chat models only accept `max_tokens`.

PARAMETER NAMING CHEAT SHEET (each provider has its own name — LangChain
does NOT translate them for you):

    Provider      Chat Models Use
    ----------    ------------------------
    OpenAI        max_completion_tokens
    Anthropic     max_tokens
    Google        max_output_tokens

LANGCHAIN ANTHROPIC ALIASES (max_tokens vs max_tokens_to_sample):

    Class                    Parameter to pass        Alias
    ---------------------    ---------------------    ---------------------
    ChatAnthropic            max_tokens               max_tokens_to_sample
    Anthropic (LLM)          max_tokens_to_sample     max_tokens

    - Use `max_tokens` for ChatAnthropic (chat-style, recommended).
    - Use `max_tokens_to_sample` for the older `Anthropic` LLM class.
    - Passing the wrong one (or an OpenAI-style name) is what triggers
      the TypeError / model_kwargs warning.

SUMMARY:
   - `ChatAnthropic` + OpenRouter base_url  -> wrong class for the provider (404).
   - `max_token` (singular)                 -> wrong parameter name (TypeError).
   - Fix both: either use Anthropic directly with `max_tokens`,
     or use `ChatOpenAI` with OpenRouter and `max_tokens`.
"""

client = ChatAnthropic(
    model=MODEL,
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_token=10
)

response = client.invoke('Write a 50 word paragraph.')

print(response.content)
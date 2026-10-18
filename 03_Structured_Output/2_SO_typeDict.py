import os

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ["OPENROUTER_MODEL"]

model = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

class Review(TypedDict):

    key_themes: Annotated[list[str], 'Write down all the key themes discussed in the review in a list.']

    pros: Annotated[Optional[list[str]], 'Write down all the pros inside a list']

    cons: Annotated[Optional[list[str]], 'Write down all the cons inside a list']

    summary: Annotated[Literal["pos", "neg"], 'A brief summary of the review.']

    sentiment: str

structured_model = model.with_structured_output(Review)

result = structured_model.invoke('''These earbuds punch way above their price. Sound is clear with decent bass, they fit comfortably for hours, and battery lasts about 6 hours per charge. Only downside is the plasticky case, but honestly, hard to beat for the money.
''')

print(result['summary'])
print(result['sentiment'])
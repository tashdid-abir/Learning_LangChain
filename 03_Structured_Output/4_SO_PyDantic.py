import os

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

from typing import Annotated, Literal, Optional

from pydantic import BaseModel, Field

load_dotenv()
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ["OPENROUTER_MODEL"]

model = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

class Review(BaseModel):
    key_themes: Annotated[
        list[str],
        Field(description="Write down all the key themes discussed in the review in a list.")
    ]
    pros: Annotated[
        Optional[list[str]],
        Field(default=None, description="Write down all the pros inside a list.")
    ]
    cons: Annotated[
        Optional[list[str]],
        Field(default=None, description="Write down all the cons inside a list.")
    ]
    summary: Annotated[
        Literal["pos", "neg"],
        Field(description="A brief summary of the review: pos or neg.")
    ]
    sentiment: Annotated[
        str,
        Field(description="The overall sentiment of the review.")
    ]

structured_model = model.with_structured_output(Review)

review = """
The earbuds offer excellent value for their price. The sound is clear, with
decent bass, and they remain comfortable during long listening sessions. The
battery lasts approximately six hours on a single charge. The main drawback
is that the charging case feels somewhat plasticky. Overall, these earbuds
are difficult to beat for the money.
"""

result = structured_model.invoke(review)

print("Key themes:", result.key_themes)
print("Pros:", result.pros)
print("Cons:", result.cons)
print("Summary:", result.summary)
print("Sentiment:", result.sentiment)
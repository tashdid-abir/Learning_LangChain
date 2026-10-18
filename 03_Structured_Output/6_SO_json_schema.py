import os

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ["OPENROUTER_MODEL"]

model = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

review_schema = {
    "title": "Review",
    "description": "A structured summary of a product review.",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The main themes discussed in the review."
        },
        "pros": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The positive aspects of the product."
        },
        "cons": {
            "type": "array",
            "items": {"type": "string"},
            "description": "The negative aspects or drawbacks of the product."
        },
        "summary": {
            "type": "string",
            "enum": ["pos", "neg"],
            "description": "A short classification of the review: pos or neg."
        },
        "sentiment": {
            "type": "string",
            "description": "The overall emotional sentiment of the review."
        }
    },
    "required": ["key_themes", "summary", "sentiment"],
    "additionalProperties": False
}

structured_model = model.with_structured_output(review_schema)

review = """
The earbuds offer excellent value for their price. The sound is clear, with
decent bass, and they remain comfortable during long listening sessions. The
battery lasts approximately six hours on a single charge. The main drawback
is that the charging case feels somewhat plasticky. Overall, these earbuds
are difficult to beat for the money.
"""

result = structured_model.invoke(review)

print("Key themes:", result["key_themes"])
print("Pros:", result.get("pros", []))
print("Cons:", result.get("cons", []))
print("Summary:", result["summary"])
print("Sentiment:", result["sentiment"])
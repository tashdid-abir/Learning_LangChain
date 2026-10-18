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


chat_history = []

while True:
    user_input = input('You : ')
    chat_history.append(user_input)

    if user_input.lower() == 'exit':
        break
    result = model.invoke(chat_history)
    chat_history.append(result.content)

    print('AI : ', result.content)


print(chat_history)

from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv

load_dotenv()

client = ChatGoogleGenerativeAI(model='')

response = client.invoke('Say anything')

print(response.content)


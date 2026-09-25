import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


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

prompt1 = PromptTemplate(
    template='Generate a detailed report on the {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 point summary on the {text}',
    input_variables=['text']
)

parser = StrOutputParser()


chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': ' Unemployment in India.'})

print(result)
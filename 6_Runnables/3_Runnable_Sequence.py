import os

from langchain_openai import ChatOpenAI
from langchain_classic.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableSequence

from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ["OPENROUTER_MODEL"]

prompt_1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template='Explain the given joke : {joke}',
    input_variables=['joke']
)

model = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

parser = StrOutputParser()

chain = RunnableSequence(prompt_1, model, parser, prompt_2, model, parser)

result = chain.invoke({'topic':'AI'})

print(result)
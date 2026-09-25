import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ["OPENROUTER_MODEL"]

prompt = PromptTemplate(
    template='Write 5 interesting facts on the {topic}',
    input_variables=['topic']
)

model = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

pareser = StrOutputParser()

chain = prompt | model | pareser

result = chain.invoke({'topic':'AI'})


chain.get_graph().print_ascii()
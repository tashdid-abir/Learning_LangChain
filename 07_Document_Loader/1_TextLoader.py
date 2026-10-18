import os

from langchain_community.document_loaders import TextLoader

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ["OPENROUTER_MODEL"]

model = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

prompt = PromptTemplate(
    template='Write a summary for the following text - \n {text}',
    input_variables=['text']
)

parser = StrOutputParser()

loader = TextLoader(
    'path to file',
    encoding='utf-8'
)

docs = loader.load()

chain = prompt | model | parser

chain.invoke({'text':docs[0].page_content})

print(type(docs))
print(docs[0].metadata)
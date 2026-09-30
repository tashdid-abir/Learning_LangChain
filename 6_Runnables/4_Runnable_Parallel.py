import os

from langchain_openai import ChatOpenAI
from langchain_classic.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import RunnableSequence, RunnableParallel

from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ["OPENROUTER_MODEL"]

model = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_completion_tokens=500
)

prompt_1 = PromptTemplate(
    template='Generate a tweet about {topic}',
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template='Generate a linkedin post about : {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'tweet': RunnableSequence(prompt_1, model, parser),
    'linkedin': RunnableSequence(prompt_2, model, parser)
})

result = parallel_chain.invoke({'topic':'AI'})

print(result)
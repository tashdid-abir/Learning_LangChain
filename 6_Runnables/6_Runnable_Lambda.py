import os

from langchain_openai import ChatOpenAI
from langchain_classic.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import ( 
    RunnableSequence, 
    RunnableParallel, 
    RunnablePassthrough,
    RunnableLambda
)

from dotenv import load_dotenv

# def word_count(text):
#     return len(text.split())

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

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count' : RunnableLambda(lambda x: len(x.split()))
})

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

result = final_chain.invoke({'topic':'AI'})

print(result)
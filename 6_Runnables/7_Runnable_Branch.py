import os

from langchain_openai import ChatOpenAI
from langchain_classic.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.schema.runnable import ( 
    RunnableSequence,  
    RunnablePassthrough,
    RunnableBranch
)

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
    template='write a detailed report on {topic}',
    input_variables=['topic']
)

prompt_2 = PromptTemplate(
    template='Summarize the following text {text}',
    input_variables=['text']
)

parser = StrOutputParser()

report_generation_chain = RunnableSequence(prompt_1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt_2, model, parser)),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_generation_chain, branch_chain)

result = final_chain.invoke({'topic':'Russia'})

print(result)
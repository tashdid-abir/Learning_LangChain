import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    provider="featherless-ai",
    task="text-generation",
    max_new_tokens=200,
    do_sample=False,
    huggingfacehub_api_token=HF_TOKEN,
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template= 'Give me the name, age, and city of a fictional person \n {format_instruction}',
    input_variables=[],
    partial_variables={
        'format_instruction' : parser.get_format_instructions()
    }
)

prompt = template.format()

print(prompt)

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(type(final_result))

print(final_result['name'])


# chain = template | model | parser
# chain.invoke({})
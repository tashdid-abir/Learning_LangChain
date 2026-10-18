import os

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate

from langchain_classic.output_parsers import (
    StructuredOutputParser,
    ResponseSchema,
)

load_dotenv()
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ["OPENROUTER_MODEL"]

model = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

schema = [
    ResponseSchema(name='fact_1', description='Fact 1 about the topic'),
    ResponseSchema(name='fact_2', description='Fact 2 about the topic'),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give 2 facts about the {topic}\n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction':parser.get_format_instructions()}
)

# chain = template | model | parser


prompt = template.invoke({'topic': 'Lionel Messi'})

result = model.invoke(prompt)

print(parser.parse(result.content))
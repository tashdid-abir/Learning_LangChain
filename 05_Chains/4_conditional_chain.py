import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableLambda

load_dotenv()
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ["OPENROUTER_MODEL"]


model = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

parser = StrOutputParser()


class Feedback(BaseModel):
    sentiment: Literal['Positive', 'Negative'] = Field(
        description='Give the sentiment of the feedback'
    )

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template=(
        'Classify the sentiment of the following feedback text. '
        '\n {feedback} \n {format_instruction}'
    ),
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()},
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate response to this positive {feedback}',
    input_variables=['feedback'],
)

prompt3 = PromptTemplate(
    template='Write an appropriate response to this negative {feedback}',
    input_variables=['feedback'],
)

positive_chain = prompt2 | model | parser
negative_chain = prompt3 | model | parser

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'Positive', positive_chain),
    (lambda x: x.sentiment == 'Negative', negative_chain),
    RunnableLambda(lambda x: "Could not find sentiment."),
)

final_chain = classifier_chain | branch_chain

result = final_chain.invoke({
    'feedback': (
        'The product arrived later than promised.\n'
        'The packaging was damaged, and one item was missing.\n'
        'Customer support took several days to respond.\n'
        'Overall, I am very disappointed with this experience.'
    )
})

print(result)
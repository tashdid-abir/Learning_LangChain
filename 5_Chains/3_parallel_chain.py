import os

from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from langchain_core.runnables import RunnableParallel

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL = os.environ["OPENROUTER_MODEL"]
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

model1 = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
    temperature=0
)

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0:featherless-ai",
    task='text-generation',
    do_sample=False,
    huggingfacehub_api_token=HF_TOKEN
)
model2 = ChatHuggingFace(llm=llm)


prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text : \n {text}',
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template='Generate 5 short Q/A from the following text : \n {text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and Q/A into a single document : \n {notes} and {quiz} ',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()


parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz'  : prompt2 | model2 | parser
})

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
LangChain is a framework for building applications powered by large language models.
Instead of treating a language model as an isolated text generator, LangChain helps
developers connect the model to prompts, documents, tools, databases, memory, and
other application components. This makes it possible to create useful workflows
such as question-answering systems, document summarizers, chatbots, research
assistants, and automated agents.

One important idea in LangChain is the runnable. A runnable is a component that
accepts an input and produces an output. Prompt templates, chat models, output
parsers, and custom Python functions can all be combined as runnable components.
The pipe operator makes it easy to connect these components into a sequence. For
example, an application can first format a prompt, send that prompt to a model,
and then convert the model response into a plain string. This approach keeps each
step separate and makes the complete workflow easier to read, test, and modify.

LangChain also supports parallel execution. A RunnableParallel can send the same
input to multiple independent chains at the same time. For example, one chain can
create short notes while another chain creates questions and answers. The results
are returned in a dictionary, where each output is identified by a key such as
notes or quiz. A later prompt can then use those keys to combine the independent
results into one final document.

Prompt templates are useful because they separate instructions from changing data.
Rather than building long strings manually, a developer can define placeholders
such as {text}, {notes}, or {quiz}. When the chain runs, LangChain replaces those
placeholders with the appropriate values. This reduces duplication and makes the
prompts easier to reuse across different inputs and models.

Output parsers provide another useful abstraction. Language models may return
messages or other structured response objects, while an application often needs
plain text, JSON, lists, or validated objects. A parser transforms the model output
into the format expected by the next step. Combining prompt templates, models, and
parsers creates a predictable pipeline that can be extended as the application
becomes more complex.

When building a production application, developers should also consider error
handling, API costs, response latency, authentication, logging, and evaluation.
The quality of the final result depends not only on the model but also on the input
data, prompt instructions, output format, and validation rules. A well-designed
LangChain workflow makes these concerns visible so that each part can be improved
independently.
"""

result = chain.invoke({'text':text})

print(result)
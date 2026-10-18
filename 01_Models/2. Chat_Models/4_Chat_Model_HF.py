import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0:featherless-ai",
    task='text-generation',
    max_new_tokens=200,
    do_sample=False,
    huggingfacehub_api_token=HF_TOKEN
)

model = ChatHuggingFace(llm=llm)

result = model.invoke('Who is the president of USA ?')

print(result.content)
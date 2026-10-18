from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

import os

os.environ['HF_HOME'] = 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id='',
    task='',
    pipeline_kwargs={
        "temperature": 0.5, # temperature may not work with every Hugging Face pipeline/model.
        "max_new_tokens": 200,
    },
)
model = ChatHuggingFace(
    llm=llm
)

response = model.invoke("Hello, how are you?")
print(response.content)
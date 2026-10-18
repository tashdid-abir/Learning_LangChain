import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


load_dotenv()

API_KEY = os.environ["OPENROUTER_API_KEY"]

documents = [
    Document(
        page_content="LangChain makes it easy to work with LLMs."
    ),
    Document(
        page_content="LangChain is used to build LLM-based applications."
    ),
    Document(
        page_content="Chroma is used to store and search document embeddings."
    ),
    Document(
        page_content="Embeddings are vector representations of text."
    ),
    Document(
        page_content="MMR helps you get diverse results during similarity search."
    ),
    Document(
        page_content="LangChain supports Chroma, FAISS, Pinecone, and more."
    ),
]

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=embedding_model,
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 3,
        "lambda_mult": 0.5,
    },
)

query = "What is LangChain?"

results = retriever.invoke(query)

for i, doc in enumerate(results, start=1):
    print(f"\n--- MMR Result {i} ---")
    print(doc.page_content)

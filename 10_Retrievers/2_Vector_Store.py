import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


load_dotenv()

API_KEY = os.environ["OPENROUTER_API_KEY"]

documents = [
    Document(
        page_content="LangChain helps developers build LLM applications easily."
    ),
    Document(
        page_content="Chroma is a vector database optimized for LLM-based search."
    ),
    Document(
        page_content="Embeddings convert text into high-dimensional vectors."
    ),
    Document(
        page_content="OpenAI provides powerful embedding models."
    ),
]

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding_model,
    collection_name="retriever_vector_store",
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2}
)

query = "What is Chroma used for?"

results = retriever.invoke(query)

for i, doc in enumerate(results, start=1):
    print(f"\n--- Retriever Result {i} ---")
    print(doc.page_content)

similarity_results = vectorstore.similarity_search(
    query=query,
    k=2,
)

for i, doc in enumerate(similarity_results, start=1):
    print(f"\n--- Similarity Search Result {i} ---")
    print(doc.page_content)

import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


load_dotenv()

API_KEY = os.environ["OPENROUTER_API_KEY"]
MODEL_NAME = os.environ["OPENROUTER_MODEL"]

documents = [
    Document(
        page_content=(
            "Regular walking boosts heart health and can reduce symptoms "
            "of depression."
        ),
        metadata={"source": "H1"},
    ),
    Document(
        page_content=(
            "Consuming leafy greens and fruits helps improve longevity."
        ),
        metadata={"source": "H2"},
    ),
    Document(
        page_content=(
            "Deep sleep is crucial for cellular repair and emotional regulation."
        ),
        metadata={"source": "H3"},
    ),
    Document(
        page_content=(
            "Mindfulness and controlled breathing lower cortisol and improve "
            "mental clarity."
        ),
        metadata={"source": "H4"},
    ),
    Document(
        page_content=(
            "Drinking sufficient water throughout the day helps maintain "
            "metabolism and energy."
        ),
        metadata={"source": "H5"},
    ),
    Document(
        page_content=(
            "Solar energy systems in modern homes help balance electricity demand."
        ),
        metadata={"source": "I1"},
    ),
    Document(
        page_content=(
            "Python balances readability with power, making it popular for "
            "system design."
        ),
        metadata={"source": "I2"},
    ),
    Document(
        page_content=(
            "Photosynthesis enables plants to produce energy by converting sunlight."
        ),
        metadata={"source": "I3"},
    ),
    Document(
        page_content=(
            "The 2022 FIFA World Cup was held in Qatar."
        ),
        metadata={"source": "I4"},
    ),
    Document(
        page_content=(
            "Black holes bend spacetime and contain immense gravitational energy."
        ),
        metadata={"source": "I5"},
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

similarity_retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(
        search_kwargs={"k": 5}
    ),
    llm=llm,
)

query = "How can I improve my energy levels and maintain balance?"

similarity_results = similarity_retriever.invoke(query)
multiquery_results = multiquery_retriever.invoke(query)

print("\n===== Similarity Retriever Results =====")

for i, doc in enumerate(similarity_results, start=1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)
    print(f"Source: {doc.metadata.get('source')}")

print("\n" + "*" * 100)

print("\n===== MultiQuery Retriever Results =====")

for i, doc in enumerate(multiquery_results, start=1):
    print(f"\n--- Result {i} ---")
    print(doc.page_content)
    print(f"Source: {doc.metadata.get('source')}")

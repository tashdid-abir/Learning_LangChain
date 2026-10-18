import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


load_dotenv()

API_KEY = os.environ["OPENROUTER_API_KEY"]

documents = [
    Document(
        page_content=(
            "Growing basil near a sunny kitchen window works best when the soil "
            "drains well and the plant receives several hours of light each day."
        ),
        metadata={"source": "urban_gardening"},
    ),
    Document(
        page_content=(
            "The northern lights are caused by charged particles from the Sun "
            "interacting with gases in Earth's upper atmosphere."
        ),
        metadata={"source": "astronomy_fact"},
    ),
    Document(
        page_content=(
            "A good sourdough starter needs flour, water, and regular feeding "
            "to maintain the yeast and bacteria that help bread rise."
        ),
        metadata={"source": "bread_baking"},
    ),
    Document(
        page_content=(
            "Honeybees communicate the location of flowers through a movement "
            "known as the waggle dance."
        ),
        metadata={"source": "nature_fact"},
    ),
    Document(
        page_content=(
            "The piano has eighty-eight keys, allowing musicians to play both "
            "very low bass notes and very high treble notes."
        ),
        metadata={"source": "music_fact"},
    ),
]

document_ids = [document.metadata["source"] for document in documents]

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=API_KEY,
    base_url="https://openrouter.ai/api/v1",
)

vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory="my_chroma_db",
    collection_name="sample",
)

vector_store.add_documents(documents, ids=document_ids)

stored_documents = vector_store.get(include=["documents", "metadatas"])
print(f"Stored documents: {len(stored_documents['documents'])}")

results = vector_store.similarity_search(
    query="How should I grow basil?", 
    k=2)
for document in results:
    print(f"Source: {document.metadata['source']}")
    print(f"Content: {document.page_content}\n")



results_with_scores = vector_store.similarity_search_with_score(
    query="How should I grow basil?",
    k=2,
    filter={"source": "urban_gardening"},
)
for document, score in results_with_scores:
    print(f"Filtered source: {document.metadata['source']}")
    print(f"Distance score: {score}")



updated_document = Document(
    page_content=(
        "Basil grows well in a warm, sunny spot with moist but well-drained "
        "soil. Pinching off flower buds encourages more leaves."
    ),
    metadata={"source": "urban_gardening", "updated": True},
)
vector_store.update_document(
    document_id="urban_gardening",
    document=updated_document,
)

updated_result = vector_store.similarity_search(
    query="How should I grow basil?",
    k=1,
    filter={"source": "urban_gardening"},
)
print(f"Updated content: {updated_result[0].page_content}\n")



vector_store.delete(ids=["music_fact"])
remaining_documents = vector_store.get(
    include=["documents", "metadatas"],
)
print(f"Documents after deletion: {len(remaining_documents['documents'])}")
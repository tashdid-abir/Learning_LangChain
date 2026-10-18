from langchain_community.retrievers import WikipediaRetriever


retriever = WikipediaRetriever(
    top_k_results=2,
    lang="en",
)

query = "The geopolitical history of India and Pakistan from a Chinese perspective"

docs = retriever.invoke(query)

for i, doc in enumerate(docs, start=1):
    print(f"\n--- Result {i} ---")
    print(f"Content:\n{doc.page_content}")
    print(f"Metadata: {doc.metadata}")

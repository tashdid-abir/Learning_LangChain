from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(
    model='sentence-transformers/all-MiniLM-L6-v2'
)

documents = [
    "LangChain makes it easier to build applications powered by language models.",
    "Embeddings represent text as numerical vectors for semantic comparison.",
    "Hugging Face provides pretrained models that can generate embeddings locally."
]

result = embedding.embed_documents(documents)

print(str(result))
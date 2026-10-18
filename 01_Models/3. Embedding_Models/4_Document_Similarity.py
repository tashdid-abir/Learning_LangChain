from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embedding = HuggingFaceEndpointEmbeddings(
    model='sentence-transformers/all-MiniLM-L6-v2'
)

documents = [
    "Lionel Messi was born on June 24, 1987, in Rosario, Argentina.",
    "Lionel Messi has won a record eight Ballon d'Or awards.",
    "Lionel Messi scored 672 goals for FC Barcelona, the most in the club's history.",
    "Lionel Messi captained Argentina to win the 2022 FIFA World Cup.",
    "Lionel Messi joined Inter Miami in 2023 after spells at PSG and Barcelona."
]

query = 'When was he born ?'

doc_embeddings   = embedding.embed_documents(documents)
query_embedding  = embedding.embed_query(query)

similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(enumerate(similarities), key=lambda x: x[1])[-1]

print(query)
print(documents[index])
print("Similarity score is : ", score)


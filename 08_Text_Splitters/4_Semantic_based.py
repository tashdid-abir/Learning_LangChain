from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings


load_dotenv()


text = (
    "LangChain is a framework for building applications powered by language models. "
    "It provides tools for prompts, chains, agents, memory, and document processing. "
    "Semantic chunking groups together sentences with similar meanings. "
    "This approach can create more meaningful chunks than splitting text at fixed "
    "character or token boundaries."
)


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
splitter = SemanticChunker(embeddings=embeddings,
                           breakpoint_threshold_type='standard_deviation',
                           breakpoint_threshold_amount=1)


chunks = splitter.split_text(text)

for index, chunk in enumerate(chunks, start=1):
    print(f"Chunk {index}:\n{chunk}\n")
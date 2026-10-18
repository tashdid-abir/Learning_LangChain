from langchain_classic.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


text = (
    "LangChain is a powerful framework for building applications with large language models. "
    "It provides tools for prompt management, chaining, memory, and text splitting. "
    "CharacterTextSplitter splits text into chunks by characters, useful for chunking documents."
)


splitter = CharacterTextSplitter(
    chunk_size=50,
    chunk_overlap=0,
    separator="",
)


# Split plain text into fixed-length chunks.
text_result = splitter.split_text(text=text)
print(text_result)


# Load and split documents into fixed-length chunks.
loader = PyPDFLoader(file_path="")
documents = loader.load()
document_result = splitter.split_documents(documents=documents)
print(document_result)
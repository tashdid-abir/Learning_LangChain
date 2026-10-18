from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import Language


text = (
    "from langchain_text_splitters import RecursiveCharacterTextSplitter\n\n"
    "splitter = RecursiveCharacterTextSplitter.from_language(\n"
    "    language=Language.PYTHON,\n"
    "    chunk_size=50,\n"
    "    chunk_overlap=0,\n"
    ")\n"
)


splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=50,
    chunk_overlap=0,
)


# Split Python source code using Python-aware separators.
text_result = splitter.split_text(text=text)
print(text_result)


# Load and split PDF documents using the same splitter.
loader = PyPDFLoader(file_path="")
documents = loader.load()
document_result = splitter.split_documents(documents=documents)
print(document_result)
from langchain_community.document_loaders import WebBaseLoader

url = ''

loader = WebBaseLoader(url) #returns one single document object

docs = loader.load()
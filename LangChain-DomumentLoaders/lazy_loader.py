from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader(
    path='LangChain-DomumentLoaders\Blockchain Modules',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# docs = loader.load()
docs = loader.lazy_load() # faster laoding than load() but it returns a generator instead of a list
for document in docs:
    print(document.metadata)
# print(docs[0].page_content)
# print(docs[0].metadata)
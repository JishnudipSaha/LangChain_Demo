from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader

loader = DirectoryLoader(
    path='LangChain-DomumentLoaders\Blockchain Modules',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()
print(docs[0].page_content)
print(docs[0].metadata)
# print(docs)
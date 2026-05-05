from langchain_community.document_loaders import TextLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()


poem_summary_prompt = PromptTemplate(
    template='Wtite a short Summary of the following poem: \n{poem}',
    input_variables=['poem'],
)


model = ChatOllama(
    model='gemma4:31b-cloud',
    temperature=0.7,
)

parser = StrOutputParser()

poem_summary_chain = poem_summary_prompt | model | parser


loader = TextLoader("LangChain-DomumentLoaders\cricket.txt", encoding="utf-8")
docs = loader.load()
# print(len(docs))
# print(docs[0].page_content)
# print(docs[0].metadata)
# print(type(docs[0]))

poem_content = docs[0].page_content

response = poem_summary_chain.invoke(
    {
    'poem': poem_content
    }
)

print('Poem:\n',poem_content)

print('\nSummary:\n',response)


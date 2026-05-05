from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()



url ='https://www.amazon.in/Apple-MacBook-Laptop-18%E2%80%91core-40%E2%80%91core/dp/B0GR1LB81D/ref=sr_1_2_sspa?crid=C17TJ2EI67KM&dib=eyJ2IjoiMSJ9.IH7ekYjEFZw8JIQA4mxDubaZ51xNwtmD0nLpvQ2Cw3oPZIgy0bmoJZDz9TqGVN_Wf_tdxWQrJw8MlmbQi0WAjZvfbjnxT-0Sy4Z-ZKcERrnK2sOpjMHqenh1fjMIxssQJP0QyFO32EhWilBCb766MAS1d28pqPcdiVvYIHla6HrWKHA8d6A09EhLEnS7jeGhOC50LvZJqA3HeSRtIz0ARsL4o8UIdLCltILemTmTbVI.N5W4dR1Y_O2r4vhuTAzJJfhywxx61q8iOp-WphxRNto&dib_tag=se&keywords=M5%2Bpro&qid=1778009710&sprefix=m5%2Bpro%2Caps%2C865&sr=8-2-spons&aref=lVLyuWKYTm&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&th=1'

loader = WebBaseLoader(url)
docs = loader.load()



prompt = PromptTemplate(
    template='Answer the following question based on the given document: \nDocument: {document}\nQuestion: {question}',
    input_variables=['document', 'question'],
)

parser = StrOutputParser()

model = ChatOllama(
    model='gemma4:31b-cloud',
    temperature=0.7,
)



chain = prompt | model | parser
response = chain.invoke(
    {
    'document': docs[0].page_content,
    'question': 'Lists all the specifications of the laptop mentioned in the document.'
    }
)
print(response)
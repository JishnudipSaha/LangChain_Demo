from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
load_dotenv()

model = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0.7
)

prompt = PromptTemplate(
    template='Write a 5 interesting facts on the following {topic}',
    input_variables=['topic']
)

# prompt = template.invoke(
#     {
#         'topic': 'Artificial Intelligence'
#     }
# )

parser = StrOutputParser()

# print(prompt)

chain = prompt | model | parser
result = chain.invoke({
        'topic': 'Black Hole'
    })
print(result)
chain.get_graph().print_ascii()


from typing_extensions import final

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.prompt import Prompt
load_dotenv()



model = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0.7
)

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(description='Age of the person', gt=18)
    city: str = Field(description='City of the person belongs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate unique name,age and city of a fictional {place} person \n {format_instructions}',
    input_variables=['place'],
    partial_variables={'format_instructions': parser.get_format_instructions()}
)

# prompt = template.format(place='Indian')
# prompt = template.invoke({'place': 'Indian'}) # better to use invoke instead of format
# print(prompt)

# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

chain = template | model | parser
final_result = chain.invoke({'place': 'Indian'})
print(final_result)
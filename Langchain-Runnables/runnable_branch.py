from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch, RunnableParallel, RunnableSequence, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv

load_dotenv()


prompt1 = PromptTemplate(
        template='Generate a detailed report on {topic}',
        input_variables=['topic']
)

prompt2 = PromptTemplate(
        template='Summarize the following text: \n{text}',
        input_variables=['text']
)

model = ChatOllama(
    model='gemma4:31b-cloud',
    temperature=0.7
)

parser = StrOutputParser()

# report generation chain
report_gen_chain = RunnableSequence(prompt1, model, parser)
# summary generation chain
# summary_gen_chain = RunnableSequence(prompt2, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)), # if length gt 500, the summary chain is executed
    RunnablePassthrough() # otherwise, the report chain is executed
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)
response = final_chain.invoke(
    {
        'topic': 'Russia VS Ukraine War'
    }
)

print('Report/Summary:', response)
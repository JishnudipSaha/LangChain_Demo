from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
load_dotenv()

model1 = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0.7
)

model2 = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0.7
)


prompt1 = PromptTemplate(
    template='Generate short and simple notes from the following text \n{text}',
    input_variables=['test']
)


prompt2 = PromptTemplate(
    template='Generate 5 short question answers for the following text \n{text}',
    input_variables=['text']
)

prompt3 = PromptTemplate(
    template='Merge the provided notes and quiz in a single document \n notes: {notes} \n quiz: {quiz}',
    input_variables=['notes', 'quiz']
)

parser = StrOutputParser()

# defining the parallel chain
parallel_chain = RunnableParallel(
    {
        'notes': prompt1 | model1 | parser,
        'quiz': prompt2 | model2 | parser
    }
)


# defining the merge chain
merge_chain = prompt3 | model1 | parser

# defining the final chain (parallel chain + merge chain)
final_chain = parallel_chain | merge_chain


# SVM text
text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""


response = final_chain.invoke(
    {
        'text': text,
    }
)

print(response)
final_chain.get_graph().print_ascii()

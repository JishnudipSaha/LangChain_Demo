import ollama
import numpy as np
response = ollama.embed(
    model='nomic-embed-text',
    input='The sky is blue because of Rayleigh scattering'
    # dimensions=72
)
result = response.embeddings
arr = np.array(result)
print(arr.ndim)
print(arr.shape)
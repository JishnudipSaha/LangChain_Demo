import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()

client = InferenceClient(
    api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
)

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V4-Pro:novita",
    messages=[
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ],
)

print(completion.choices[0].message)
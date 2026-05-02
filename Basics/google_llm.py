from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",   # ✅ from YOUR available models
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

response = llm.invoke("Explain AI in simple words")
print(response.content)
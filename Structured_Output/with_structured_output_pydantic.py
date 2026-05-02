from langchain_ollama import ChatOllama
from typing import Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from json import JSONDecodeError
from langchain_core.exceptions import OutputParserException
from langchain_core.utils.json import parse_json_markdown

load_dotenv()

# model using gemma4:31b-cloud
model = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0.7
    # format='json'
    )


# schema: a pydantic class
class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Return the sentiment of the review. It can be positive, negative, or neutral.")
    pros: Optional[list[str]] = Field(description="List all the pros of the product.", default=None)
    cons: Optional[list[str]] = Field(description="List all the cons of the product.", default=None)
    name: Optional[str] = Field(description="Name of the reviewer", default=None)

structured_model = model.with_structured_output(Review)

review_text = """The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this."""


def invoke_review_with_fallback(text: str) -> Review:
    try:
        return structured_model.invoke(text)
    except OutputParserException:
        strict_json_prompt = (
            "Extract the following product review into JSON.\n"
            "Return ONLY a valid JSON object with keys: key_themes, summary, sentiment, pros, cons, name.\n"
            "Rules:\n"
            "- sentiment must be one of: positive, negative, neutral\n"
            "- pros, cons, and name can be null when missing\n"
            "- Do not include markdown, code fences, or any extra text\n\n"
            f"Review text:\n{text}"
        )
        retry_json_prompt = (
            "Return ONLY valid JSON. No explanation.\n"
            "Required keys: key_themes, summary, sentiment, pros, cons, name.\n"
            "Sentiment must be exactly one of: positive, negative, neutral.\n"
            "pros, cons, and name can be null.\n\n"
            f"Review text:\n{text}"
        )
        last_error: Exception | None = None
        last_output = ""

        for prompt in (strict_json_prompt, retry_json_prompt):
            raw_response = model.invoke(prompt)
            last_output = raw_response.content
            try:
                parsed = parse_json_markdown(raw_response.content)
                if not isinstance(parsed, dict):
                    raise ValueError("Fallback response is not a JSON object.")
                return Review.model_validate(parsed)
            except (JSONDecodeError, ValueError) as err:
                last_error = err

        raise ValueError(
            f"Could not parse model output as Review JSON after retry. Last parser error: {last_error}. "
            f"Last raw output: {last_output[:300]!r}"
        ) from last_error


result = invoke_review_with_fallback(review_text)
print(result)

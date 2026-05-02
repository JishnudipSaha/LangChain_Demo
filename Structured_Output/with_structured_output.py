from langchain_ollama import ChatOllama
from typing import TypedDict, Annotated, Literal
from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from langchain_core.utils.json import parse_json_markdown

load_dotenv()

# model using gemma4:31b-cloud
model = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0.7
    )


# schema
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes in the review in a list"]
    summary: Annotated[str, "Write a concise summary of the review."]
    sentiment: Annotated[Literal["positive", "negative", "neutral"], "Return the sentiment of the review. It can be positive, negative, or neutral."]
    pros: Annotated[list[str], "List all the pros of the product."]
    cons: Annotated[list[str], "List all the cons of the product."]
    name: Annotated[str, "Name of the reviewer"]


structured_model = model.with_structured_output(Review)

review_text = """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful

Cons:
Bloatware and unnecessary Samsung apps
Overpriced for the features
Bulky and heavy design
Some zoom limitations beyond 30x

Overall, it’s a fantastic phone for power users and photography enthusiasts, but it’s not perfect.
Review by Jishnudip Saha"""


def _coerce_review(data: dict) -> Review:
    required_keys = {"key_themes", "summary", "sentiment", "pros", "cons", "name"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Missing required keys in output: {sorted(missing)}")

    sentiment = str(data["sentiment"]).lower().strip()
    if sentiment not in {"positive", "negative", "neutral"}:
        raise ValueError(f"Invalid sentiment value: {data['sentiment']}")

    return {
        "key_themes": [str(item) for item in data["key_themes"]],
        "summary": str(data["summary"]),
        "sentiment": sentiment,
        "pros": [str(item) for item in data["pros"]],
        "cons": [str(item) for item in data["cons"]],
        "name": str(data["name"]),
    }


def _invoke_with_fallback(text: str) -> Review:
    try:
        result = structured_model.invoke(text)
        if hasattr(result, "model_dump"):
            return _coerce_review(result.model_dump())
        if isinstance(result, dict):
            return _coerce_review(result)
        return _coerce_review(dict(result))
    except OutputParserException:
        json_prompt = (
            "Return ONLY a valid JSON object with keys: key_themes, summary, sentiment, pros, cons, name. "
            "sentiment must be one of: positive, negative, neutral. "
            "Do not include markdown, code fences, or any extra text.\n\n"
            f"Review text:\n{text}"
        )
        raw_response = model.invoke(json_prompt)
        parsed = parse_json_markdown(raw_response.content)
        if not isinstance(parsed, dict):
            raise ValueError("Model fallback did not return a JSON object.")
        return _coerce_review(parsed)


review_dict: Review = _invoke_with_fallback(review_text)

print(review_dict)
print(review_dict['sentiment'])
print(review_dict['summary'])
print(type(review_dict))

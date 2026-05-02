from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from json import JSONDecodeError
from jsonschema import validate, ValidationError
from langchain_core.exceptions import OutputParserException
from langchain_core.utils.json import parse_json_markdown

load_dotenv()

# model using gemma4:31b-cloud
model = ChatOllama(
    model="gemma4:31b-cloud",
    temperature=0.7
    # format='json'
    )


# schema
json_schema = {
    "title": "Review",
    "type": "object",
    "properties": {
    "key_themes": {
        "type": "array",
        "items": {
        "type": "string"
        },
        "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
        "type": "string",
        "description": "A brief summary of the review"
    },
    "sentiment": {
        "type": "string",
        "enum": ["positive", "negative", "neutral"],
        "description": "Return sentiment of the review either negative, positive or neutral"
    },
    "pros": {
        "type": ["array", "null"],
        "items": {
        "type": "string"
        },
        "description": "Write down all the pros inside a list"
    },
    "cons": {
        "type": ["array", "null"],
        "items": {
        "type": "string"
        },
        "description": "Write down all the cons inside a list"
    },
    "name": {
        "type": ["string", "null"],
        "description": "Write the name of the reviewer"
    }
    },
    "required": ["key_themes", "summary", "sentiment"]
}


structured_model = model.with_structured_output(json_schema)

review_text = """The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this."""


def invoke_review_with_fallback(text: str):
    def _validate_review_output(data: dict) -> dict:
        if not isinstance(data, dict):
            raise ValueError("Model output must be a JSON object.")
        if isinstance(data.get("sentiment"), str):
            data["sentiment"] = data["sentiment"].strip().lower()
        validate(instance=data, schema=json_schema)
        return data

    try:
        return _validate_review_output(structured_model.invoke(text))
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
                return _validate_review_output(parsed)
            except (JSONDecodeError, ValueError, ValidationError) as err:
                last_error = err

        raise ValueError(
            f"Could not parse model output as Review JSON after retry. Last parser error: {last_error}. "
            f"Last raw output: {last_output[:300]!r}"
        ) from last_error


result = invoke_review_with_fallback(review_text)
print(result)

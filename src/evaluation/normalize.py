import re
import string


def normalize_answer(text: str) -> str:
    """Lowercase an answer and remove punctuation, articles, and extra whitespace."""
    text = text.lower()

    # remove punctuation
    text = "".join(ch for ch in text if ch not in set(string.punctuation))

    # remove articles (whole words only)
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # normalize whitespace
    text = " ".join(text.split())

    return text
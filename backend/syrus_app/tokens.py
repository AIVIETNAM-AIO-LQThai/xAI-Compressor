"""Offline prompt-size estimate. No tokenizer for the fixed model is installed, so the estimate is deliberately pessimistic:
one token per 3 ASCII characters and per 1.5 other characters (Vietnamese diacritics cost more). Live usage reported by the
provider is the check on it."""
import json


def estimate(text: str) -> int:
    ascii_chars = len(text.encode("ascii", "ignore"))
    return ascii_chars // 3 + (len(text) - ascii_chars) * 2 // 3 + 1


def estimate_request(msgs: list[dict], schema: dict) -> int:
    """Everything the provider receives for one call: the messages and the response schema."""
    return estimate(json.dumps(msgs, ensure_ascii=False)) + estimate(json.dumps(schema))

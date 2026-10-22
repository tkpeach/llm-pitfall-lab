"""Strict extraction contract; no repair or recovery of model answers."""

import json
import re
from datetime import date

FIELDS = ("order_id", "product", "quantity", "unit_price_minor", "currency", "delivery_date")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON number: {value}")


def parse_answer(text: str) -> dict:
    answer = json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    if not isinstance(answer, dict) or set(answer) != set(FIELDS):
        raise ValueError("answer must contain exactly the six specified fields")
    for key in ("order_id", "product"):
        value = answer[key]
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{key} must be a nonempty string or null")
    for key, minimum in (("quantity", 1), ("unit_price_minor", 0)):
        value = answer[key]
        if value is not None and (type(value) is not int or value < minimum):
            raise ValueError(f"{key} must be an integer >= {minimum} or null")
    if answer["currency"] is not None and answer["currency"] not in ("CNY", "USD"):
        raise ValueError("unsupported currency")
    value = answer["delivery_date"]
    if value is not None:
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError("delivery_date must be YYYY-MM-DD or null")
        date.fromisoformat(value)
    return answer


def score_answer(text: str, gold: dict) -> dict:
    # Validate gold too: malformed labels cannot silently become model errors.
    gold = parse_answer(json.dumps(gold, ensure_ascii=False, allow_nan=False))
    try:
        predicted = parse_answer(text)
    except (ValueError, TypeError) as error:
        return {"valid": False, "exact": False, "reward": 0.0,
                "field_correct": dict.fromkeys(FIELDS, False), "error": str(error),
                "gold_nonnull_correct": 0,
                "gold_nonnull_count": sum(gold[k] is not None for k in FIELDS),
                "gold_null_correct": 0,
                "gold_null_count": sum(gold[k] is None for k in FIELDS)}
    correct = {key: predicted[key] == gold[key] for key in FIELDS}
    return {"valid": True, "exact": all(correct.values()),
            "reward": sum(correct.values()) / len(FIELDS), "field_correct": correct,
            "gold_nonnull_correct": sum(correct[k] for k in FIELDS if gold[k] is not None),
            "gold_nonnull_count": sum(gold[k] is not None for k in FIELDS),
            "gold_null_correct": sum(correct[k] for k in FIELDS if gold[k] is None),
            "gold_null_count": sum(gold[k] is None for k in FIELDS)}

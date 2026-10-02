from collections import Counter

from evaluation.normalize import normalize_answer


def _f1_single(prediction: str, gold: str) -> float:
    """Compute token-overlap F1 for one prediction and one normalized reference."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        # official SQuAD convention: exact equality (incl. both-empty) counts as F1=1
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def _em_single(prediction: str, gold: str) -> int:
    """Return 1 when normalized prediction and reference strings match exactly."""
    return int(normalize_answer(prediction) == normalize_answer(gold))


def compute_em_f1(prediction: str, gold_answers: list[str]) -> tuple[int, float]:
    """
    Args:
        prediction: Model's post-processed generated answer.
        gold_answers: Acceptable reference answers for the question.

    Returns:
        The maximum exact-match and token-F1 scores over the references.
    """
    if not gold_answers:
        # SQuAD v1.1 train questions always have >=1 gold answer; guard anyway
        return 0, 0.0

    em = max(_em_single(prediction, g) for g in gold_answers)
    f1 = max(_f1_single(prediction, g) for g in gold_answers)
    return em, f1


def postprocess_generation(raw_text: str) -> str:
    """
    Removes surrounding whitespace and quotes and an optional ``Answer:``
    prefix. Apply identically across retrieval-depth conditions before scoring.
    """
    text = raw_text.strip()
    if text.lower().startswith("answer:"):
        text = text[len("answer:"):].strip()
    text = text.strip('"').strip("'").strip()
    return text
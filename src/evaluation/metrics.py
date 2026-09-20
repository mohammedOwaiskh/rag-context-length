from collections import Counter

from evaluation.normalize import normalize_answer


def _f1_single(prediction: str, gold: str) -> float:
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
    return int(normalize_answer(prediction) == normalize_answer(gold))


def compute_em_f1(prediction: str, gold_answers: list[str]) -> tuple[int, float]:
    """
    prediction: model's (post-processed) generated answer string
    gold_answers: list of acceptable gold answer strings for this question

    Returns (exact_match, f1), each the max score over all gold answers.
    """
    if not gold_answers:
        # SQuAD v1.1 train questions always have >=1 gold answer; guard anyway
        return 0, 0.0

    em = max(_em_single(prediction, g) for g in gold_answers)
    f1 = max(_f1_single(prediction, g) for g in gold_answers)
    return em, f1


def postprocess_generation(raw_text: str) -> str:
    """
    Light cleanup applied identically across all k-conditions before scoring.
    Log raw vs. post-processed separately (see generate.py) so verbosity
    trends across k can still be inspected.
    """
    text = raw_text.strip()
    if text.lower().startswith("answer:"):
        text = text[len("answer:"):].strip()
    text = text.strip('"').strip("'").strip()
    return text
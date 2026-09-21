TEMPLATE = """You are a question answering system.

Answer the question using only the information provided in the context.

Give the shortest possible answer: a single word or short phrase copied directly from the context. Do not write a full sentence. Do not explain your answer.

If the answer cannot be determined from the context, say that the answer cannot be determined.

Context:
{retrieved_context}

Question:
{question}

Answer:"""


def build_context(passage_texts: list[str]) -> str:
    """
    Concatenate retrieved passages into a single context block.

    Simple double-newline join, no passage numbering or headers — keeps the
    only varying element across k-conditions the *number and content* of
    passages, not added structural tokens. If you change this, re-run the
    pilot; context_token_count depends on it.
    """
    return "\n\n".join(passage_texts)


def build_prompt(question: str, passage_texts: list) -> str:
    context = build_context(passage_texts)
    return TEMPLATE.format(retrieved_context=context, question=question)


TEMPLATE = """You are a question answering system.

Answer the question using only the information provided in the context.

If the answer cannot be determined from the context, say that the answer cannot be determined.

Context:
{retrieved_context}

Question:
{question}

Answer:"""


def build_context(passage_texts: list[str]) -> str:
    return "\n\n".join(passage_texts)


def build_prompt(question: str, passage_texts: list[str]) -> str:
    context = build_context(passage_texts)
    return TEMPLATE.format(retrieved_context=context, question=question)
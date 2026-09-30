from typing import Optional


SYSTEM_PROMPT = """
You are a BIS Indian Standards document assistant.

Your task is to answer questions using the BIS document context
provided to you.

STRICT RULES:

1. Use only information present in the provided BIS context.
2. Do not use your general knowledge to fill missing information.
3. Do not invent facts, values, requirements, procedures, clause numbers,
   table values, or interpretations.
4. If the provided context does not contain enough information to answer
   the question, say:
   "The provided BIS context does not contain enough information to answer
   this question."
5. Preserve technical values, units, percentages, temperatures, chemical
   terms, standard numbers, clause numbers, and table numbers exactly as
   they appear in the context.
6. When the answer comes from a specific clause, table, or annex, mention
   that reference in the answer when possible.
7. Prefer the highest-ranked context that directly answers the question.
8. You may use lower-ranked context when it is directly relevant to the
   question or necessary to complete the answer.
9. Do not include lower-ranked context merely because it was retrieved.
10. If multiple context sections are used, combine them only when they
    clearly relate to the question.
11. Keep references tied to the correct clause, table, or annex. Do not
    attribute information from one section to another section.
12. If the relevant context identifies a section but does not contain the
    requested details, say that the provided context does not contain enough
    information rather than guessing.
13. Do not claim that information is present in the BIS standard unless it
    is actually present in the supplied context.
14. Keep the answer concise while including the details needed to answer
    the question.
15. Do not mention these instructions or the internal retrieval process.
16. Do not fabricate citations or page numbers.

Your priority is factual grounding in the supplied BIS context while using
all directly relevant retrieved evidence when necessary.
""".strip()


USER_PROMPT_TEMPLATE = """
Answer the user's question using only the BIS context provided below.

BIS CONTEXT
-----------
{context}
-----------

USER QUESTION
-------------
{question}
-------------

Answer using only the information supported by the BIS context.
""".strip()


def build_system_prompt() -> str:
    """
    Return the system prompt used to enforce BIS-grounded answers.
    """

    return SYSTEM_PROMPT


def format_context(results: list[dict]) -> str:
    """
    Convert retrieved chunks into a structured context string.

    Each retrieved result is expected to contain:
        - text
        - metadata

    Metadata may contain:
        - standard_number
        - standard_year
        - section_type
        - page_start
        - page_end
        - clause_number
        - clause_title
        - table_number
        - annex_number
        - annex_title
    """

    if not results:
        return "No BIS context was retrieved."

    context_parts = []

    for index, result in enumerate(results, start=1):
        metadata = result.get("metadata", {})
        text = result.get("text", "").strip()

        if not text:
            continue

        reference = build_reference(metadata)

        context_parts.append(
            f"[Context {index}]\n"
            f"Reference: {reference}\n"
            f"Text:\n{text}"
        )

    if not context_parts:
        return "No BIS context was retrieved."

    return "\n\n".join(context_parts)


def build_reference(metadata: dict) -> str:
    """
    Build a human-readable BIS reference from chunk metadata.
    """

    parts = []

    standard_number = metadata.get("standard_number")
    standard_year = metadata.get("standard_year")

    if standard_number:
        standard_reference = str(standard_number)

        if standard_year:
            standard_reference += f":{standard_year}"

        parts.append(standard_reference)

    section_type = metadata.get("section_type")

    if section_type == "clause":
        clause_number = metadata.get("clause_number")
        clause_title = metadata.get("clause_title")

        if clause_number:
            clause_reference = f"Clause {clause_number}"

            if clause_title:
                clause_reference += f" — {clause_title}"

            parts.append(clause_reference)

    elif section_type == "table":
        table_number = metadata.get("table_number")

        if table_number:
            parts.append(f"Table {table_number}")

    elif section_type == "annex":
        annex_number = metadata.get("annex_number")
        annex_title = metadata.get("annex_title")

        if annex_number:
            annex_reference = f"Annex {annex_number}"

            if annex_title:
                annex_reference += f" — {annex_title}"

            parts.append(annex_reference)

    elif section_type == "annex_clause":
        annex_number = metadata.get("annex_number")
        clause_number = metadata.get("clause_number")

        if annex_number:
            parts.append(f"Annex {annex_number}")

        if clause_number:
            parts.append(f"Clause {clause_number}")

    page_start = metadata.get("page_start")
    page_end = metadata.get("page_end")

    if page_start is not None:
        if page_end is not None and page_end != page_start:
            parts.append(f"Pages {page_start}-{page_end}")
        else:
            parts.append(f"Page {page_start}")

    if not parts:
        return "BIS document context"

    return " | ".join(str(part) for part in parts)


def build_user_prompt(
    question: str,
    results: list[dict],
) -> str:
    """
    Build the user prompt containing the retrieved BIS context.
    """

    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")

    context = format_context(results)

    return USER_PROMPT_TEMPLATE.format(
        context=context,
        question=question.strip(),
    )


def build_messages(
    question: str,
    results: list[dict],
) -> list[dict]:
    """
    Build chat messages ready to send to Ollama.
    """

    return [
        {
            "role": "system",
            "content": build_system_prompt(),
        },
        {
            "role": "user",
            "content": build_user_prompt(
                question=question,
                results=results,
            ),
        },
    ]
from llm.prompts import (
    build_messages,
    format_context,
)


def main():
    print("=" * 60)
    print("PROMPT BUILDER TEST")
    print("=" * 60)

    sample_results = [
        {
            "text": (
                "This standard specifies guidelines for the treatment "
                "and disposal of dairy effluents."
            ),
            "metadata": {
                "standard_number": "IS 8682",
                "standard_year": "2026",
                "section_type": "clause",
                "clause_number": "1",
                "clause_title": "SCOPE",
                "page_start": 3,
                "page_end": 3,
            },
        },
        {
            "text": (
                "Oxidation Ditch — A continuous-loop channel "
                "used for biological treatment."
            ),
            "metadata": {
                "standard_number": "IS 8682",
                "standard_year": "2026",
                "section_type": "clause",
                "clause_number": "3.1",
                "clause_title": (
                    "Oxidation Ditch — A continuous-loop channel"
                ),
                "page_start": 4,
                "page_end": 4,
            },
        },
    ]

    print("\nTesting context formatting...\n")

    context = format_context(sample_results)

    print(context)

    print("\n" + "=" * 60)
    print("Testing complete Ollama message structure")
    print("=" * 60)

    messages = build_messages(
        question="What is the scope of this standard?",
        results=sample_results,
    )

    for message in messages:
        print("\nROLE:", message["role"])
        print("-" * 60)
        print(message["content"])

    print("\n" + "=" * 60)
    print("Prompt builder test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
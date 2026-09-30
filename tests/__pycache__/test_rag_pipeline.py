from retrieval.hybrid import HybridRetriever
from llm.ollama import OllamaClient
from llm.prompts import build_messages


def print_retrieved_results(results):
    print("\nRetrieved BIS context:")
    print("-" * 60)

    for result in results:
        metadata = result.get("metadata", {})
        section_type = metadata.get("section_type")

        if section_type == "clause":
            reference = f"Clause {metadata.get('clause_number')}"
        elif section_type == "table":
            reference = f"Table {metadata.get('table_number')}"
        elif section_type == "annex":
            reference = f"Annex {metadata.get('annex_number')}"
        elif section_type == "annex_clause":
            reference = (
                f"Annex {metadata.get('annex_number')} "
                f"Clause {metadata.get('clause_number')}"
            )
        else:
            reference = section_type or "Unknown"

        print(
            f"Rank {result.get('rank')}: {reference} "
            f"| hybrid={result.get('hybrid_score', 0):.4f}"
        )

    print("-" * 60)


def main():
    print("=" * 60)
    print("END-TO-END RAG PIPELINE TEST")
    print("=" * 60)

    print("\nInitializing hybrid retriever...")

    retriever = HybridRetriever(
        db_path="chroma_db",
        collection_name="bis_standards_test",
        embedding_model_name="all-MiniLM-L6-v2",
        semantic_candidates=15,
    )

    print("\nInitializing Ollama...")

    ollama = OllamaClient(
        base_url="http://localhost:11434",
        model="llama3.2:3b",
    )

    if not ollama.is_available():
        print("ERROR: Ollama server is not available.")
        return

    if not ollama.model_available():
        print(
            f"ERROR: Model '{ollama.model}' is not available."
        )
        return

    print("Ollama ready.")

    question = "What are the terms and definitions used for dairy effluent treatment?"

    print("\nUser question:")
    print(question)

    print("\nRetrieving relevant BIS content...")

    results = retriever.search(
        question,
        top_k=5,
    )

    if not results:
        print("ERROR: No retrieval results found.")
        return

    print_retrieved_results(results)

    print("\nBuilding grounded prompt...")

    messages = build_messages(
        question=question,
        results=results,
    )

    print("Prompt ready.")

    print("\nSending question and context to Llama 3.2:3b...")

    try:
        answer = ollama.chat(
            messages=messages,
            temperature=0.0,
        )
    except RuntimeError as exc:
        print("\nERROR: Ollama generation failed.")
        print(exc)
        return

    if not answer:
        print("\nERROR: Ollama returned an empty answer.")
        return

    print("\n" + "=" * 60)
    print("RAG ANSWER")
    print("=" * 60)

    print(answer)

    print("\n" + "=" * 60)
    print("RAG PIPELINE TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
from retrieval.hybrid import HybridRetriever


TEST_QUERIES = [
    "What is the scope of this standard?",
    "What are the terms and definitions used for dairy effluent treatment?",
    "What are the primary treatment methods for dairy wastewater?",
    "What methods are recommended for waste reduction?",
    "What is meant by reclamation of waste products?",
    "What is biochemical oxygen demand at 5 days and 20 degrees Celsius?",
    "What are the characteristics of effluents from medium sized dairies?",
    "What are the characteristics of effluents from large sized dairies?",
    "What are the characteristics of composite effluents from three dairies?",
    "Which standards are referred to in this standard?",
]


def display_result(result):
    metadata = result.get("metadata", {})

    section_type = metadata.get("section_type")

    if section_type == "clause":
        location = (
            f"Clause {metadata.get('clause_number')}"
        )

    elif section_type == "table":
        location = (
            f"Table {metadata.get('table_number')}"
        )

    elif section_type in {
        "annex",
        "annex_clause",
    }:
        location = (
            f"Annex {metadata.get('annex_number')}"
        )

    else:
        location = section_type or "Unknown"

    print(
        f"  Rank {result['rank']}: "
        f"{location} | "
        f"hybrid={result['hybrid_score']:.4f} | "
        f"semantic={result['semantic_score']:.4f} | "
        f"keyword={result['keyword_score']:.4f} | "
        f"structural={result['structural_score']:.4f} | "
        f"distance={result['distance']:.6f}"
    )


def main():
    print("=" * 70)
    print("HYBRID RETRIEVAL TEST")
    print("=" * 70)

    retriever = HybridRetriever(
        db_path="chroma_db",
        collection_name="bis_standards_test",
        embedding_model_name="all-MiniLM-L6-v2",
    )

    for index, query in enumerate(
        TEST_QUERIES,
        start=1,
    ):
        print("\n" + "-" * 70)
        print(f"Q{index:03d}: {query}")
        print("-" * 70)

        results = retriever.search(
            query,
            top_k=5,
            candidate_k=15,
        )

        for result in results:
            display_result(result)

    print("\n" + "=" * 70)
    print("HYBRID RETRIEVAL TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
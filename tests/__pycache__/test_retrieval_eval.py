from retrieval.semantic import SemanticRetriever
from retrieval.hybrid import HybridRetriever


EVALUATION_CASES = [
    {
        "id": "Q001",
        "query": "What is the scope of this standard?",
        "expected_section": "clause",
        "expected_clause": "1",
    },
    {
        "id": "Q002",
        "query": "What are the terms and definitions used for dairy effluent treatment?",
        "expected_section": "clause",
        "expected_clause": "3",
    },
    {
        "id": "Q003",
        "query": "What are the primary treatment methods for dairy wastewater?",
        "expected_section": "clause",
        "expected_clause": "7.1",
    },
    {
        "id": "Q004",
        "query": "What methods are recommended for waste reduction?",
        "expected_section": "clause",
        "expected_clause": "6.1",
    },
    {
        "id": "Q005",
        "query": "What is meant by reclamation of waste products?",
        "expected_section": "clause",
        "expected_clause": "6.2",
    },
    {
        "id": "Q006",
        "query": "What is biochemical oxygen demand at 5 days and 20 degrees Celsius?",
        "expected_section": "table",
        "acceptable_tables": ["2", "3", "4"],
    },
    {
        "id": "Q007",
        "query": "What are the characteristics of effluents from medium sized dairies?",
        "expected_section": "table",
        "expected_table": "2",
    },
    {
        "id": "Q008",
        "query": "What are the characteristics of effluents from large sized dairies?",
        "expected_section": "table",
        "expected_table": "3",
    },
    {
        "id": "Q009",
        "query": "What are the characteristics of composite effluents from three dairies?",
        "expected_section": "table",
        "expected_table": "4",
    },
    {
        "id": "Q010",
        "query": "Which standards are referred to in this standard?",
        "expected_section": "annex",
        "expected_annex": "A",
    },
]


def result_matches(result, case):
    metadata = result.get("metadata", {})

    expected_section = case.get("expected_section")
    if expected_section and metadata.get("section_type") != expected_section:
        return False

    if "expected_clause" in case:
        return metadata.get("clause_number") == case["expected_clause"]

    if "expected_table" in case:
        return metadata.get("table_number") == case["expected_table"]

    if "acceptable_tables" in case:
        return metadata.get("table_number") in case["acceptable_tables"]

    if "expected_annex" in case:
        return metadata.get("annex_number") == case["expected_annex"]

    return False


def evaluate_retriever(name, retriever):
    print("\n" + "=" * 60)
    print(f"{name.upper()} RETRIEVAL EVALUATION")
    print("=" * 60)

    total = len(EVALUATION_CASES)
    successful_hits = 0
    reciprocal_ranks = []

    query_results = []

    for case in EVALUATION_CASES:
        query = case["query"]
        results = retriever.search(query, top_k=5)

        matched_rank = -1

        for rank, result in enumerate(results, start=1):
            if result_matches(result, case):
                matched_rank = rank
                break

        if matched_rank != -1:
            successful_hits += 1
            reciprocal_ranks.append(1.0 / matched_rank)
        else:
            reciprocal_ranks.append(0.0)

        query_results.append(
            {
                "id": case["id"],
                "query": query,
                "matched_rank": matched_rank,
                "results": results,
            }
        )

        if matched_rank != -1:
            print(
                f"\n{case['id']} PASS "
                f"(matched at rank {matched_rank})"
            )
        else:
            print(f"\n{case['id']} FAIL")

        print(f"Query: {query}")

        print("Retrieved:")

        for result in results:
            metadata = result.get("metadata", {})

            section_type = metadata.get("section_type", "unknown")

            if section_type == "clause":
                label = f"Clause {metadata.get('clause_number')}"
            elif section_type == "table":
                label = f"Table {metadata.get('table_number')}"
            elif section_type == "annex":
                label = f"Annex {metadata.get('annex_number')}"
            elif section_type == "annex_clause":
                label = (
                    f"Annex {metadata.get('annex_number')} "
                    f"Clause {metadata.get('clause_number')}"
                )
            else:
                label = section_type

            distance = result.get("distance")

            if distance is not None:
                distance_text = f"{distance:.6f}"
            else:
                distance_text = "N/A"

            if "hybrid_score" in result:
                print(
                    f"  Rank {result['rank']}: {label} "
                    f"| distance={distance_text} "
                    f"| hybrid={result['hybrid_score']:.4f} "
                    f"| semantic={result.get('semantic_score', 0):.4f} "
                    f"| keyword={result.get('keyword_score', 0):.4f} "
                    f"| structural={result.get('structural_score', 0):.4f}"
                )
            else:
                print(
                    f"  Rank {result['rank']}: {label} "
                    f"| distance={distance_text}"
                )

    top5_hit_rate = successful_hits / total
    mrr = sum(reciprocal_ranks) / total

    print("\n" + "=" * 60)
    print(f"{name.upper()} EVALUATION SUMMARY")
    print("=" * 60)

    print(f"Total questions : {total}")
    print(f"Successful hits : {successful_hits}")
    print(f"Failed hits     : {total - successful_hits}")
    print(f"Top-5 Hit Rate : {top5_hit_rate * 100:.2f}%")
    print(f"MRR             : {mrr:.4f}")

    print("\n" + "=" * 60)
    print(f"{name.upper()} PER-QUERY SUMMARY")
    print("=" * 60)

    for result in query_results:
        status = "PASS" if result["matched_rank"] != -1 else "FAIL"

        print(
            f"{result['id']} | {status} | "
            f"matched_rank={result['matched_rank']}"
        )

    return {
        "name": name,
        "total": total,
        "successful_hits": successful_hits,
        "failed_hits": total - successful_hits,
        "top5_hit_rate": top5_hit_rate,
        "mrr": mrr,
        "query_results": query_results,
    }


def print_comparison(semantic_metrics, hybrid_metrics):
    print("\n")
    print("=" * 60)
    print("SEMANTIC vs HYBRID COMPARISON")
    print("=" * 60)

    print(
        f"{'Metric':<20}"
        f"{'Semantic':>15}"
        f"{'Hybrid':>15}"
    )

    print("-" * 60)

    print(
        f"{'Top-5 Hit Rate':<20}"
        f"{semantic_metrics['top5_hit_rate'] * 100:>14.2f}%"
        f"{hybrid_metrics['top5_hit_rate'] * 100:>14.2f}%"
    )

    print(
        f"{'MRR':<20}"
        f"{semantic_metrics['mrr']:>15.4f}"
        f"{hybrid_metrics['mrr']:>15.4f}"
    )

    print(
        f"{'Successful Hits':<20}"
        f"{semantic_metrics['successful_hits']:>15}"
        f"{hybrid_metrics['successful_hits']:>15}"
    )

    print(
        f"{'Failed Hits':<20}"
        f"{semantic_metrics['failed_hits']:>15}"
        f"{hybrid_metrics['failed_hits']:>15}"
    )

    print("=" * 60)


def main():
    semantic_retriever = SemanticRetriever(
        db_path="chroma_db",
        collection_name="bis_standards_test",
        embedding_model_name="all-MiniLM-L6-v2",
    )

    hybrid_retriever = HybridRetriever(
        db_path="chroma_db",
        collection_name="bis_standards_test",
        embedding_model_name="all-MiniLM-L6-v2",
        semantic_candidates=15,
    )

    semantic_metrics = evaluate_retriever(
        "Semantic",
        semantic_retriever,
    )

    hybrid_metrics = evaluate_retriever(
        "Hybrid",
        hybrid_retriever,
    )

    print_comparison(
        semantic_metrics,
        hybrid_metrics,
    )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
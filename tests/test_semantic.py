from retrieval.semantic import SemanticRetriever


print("=" * 60)
print("SEMANTIC RETRIEVAL TEST")
print("=" * 60)


# ---------------------------------------------------------
# STEP 1 — INITIALIZE RETRIEVER
# ---------------------------------------------------------

print("\nSTEP 1 — INITIALIZING RETRIEVER")

retriever = SemanticRetriever(
    db_path="chroma_db",
    collection_name="bis_standards_test",
)

print("Retriever initialized.")


# ---------------------------------------------------------
# STEP 2 — CHECK DATABASE
# ---------------------------------------------------------

print("\nSTEP 2 — CHECKING DATABASE")

count = retriever.store.count()

print(f"Stored chunks: {count}")

assert count == 58

print("Database validation passed.")


# ---------------------------------------------------------
# STEP 3 — TEST QUERY
# ---------------------------------------------------------

query = (
    "What are the guidelines for "
    "dairy wastewater treatment?"
)

print("\nSTEP 3 — SEMANTIC SEARCH")

print(f"Query: {query}")

results = retriever.search(
    query,
    top_k=5,
)


# ---------------------------------------------------------
# STEP 4 — VALIDATE RESULTS
# ---------------------------------------------------------

print("\nSTEP 4 — VALIDATING RESULTS")

print(
    f"Results returned: {len(results)}"
)

assert len(results) == 5

print("Result count validation passed.")


# ---------------------------------------------------------
# STEP 5 — DISPLAY RESULTS
# ---------------------------------------------------------

print("\nSTEP 5 — TOP RESULTS")

for result in results:
    metadata = result["metadata"]

    print("\n" + "-" * 60)

    print(
        f"Rank: {result['rank']}"
    )

    print(
        f"Chunk ID: {result['chunk_id']}"
    )

    print(
        f"Distance: {result['distance']:.6f}"
    )

    print(
        f"Section: "
        f"{metadata.get('section_type')}"
    )

    print(
        f"Clause: "
        f"{metadata.get('clause_number')}"
    )

    print(
        f"Table: "
        f"{metadata.get('table_number')}"
    )

    print(
        f"Pages: "
        f"{metadata.get('page_start')}"
        f"-"
        f"{metadata.get('page_end')}"
    )

    print(
        f"Text:\n{result['text'][:500]}"
    )


# ---------------------------------------------------------
# STEP 6 — TEST SECOND QUERY
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("SECOND QUERY TEST")
print("=" * 60)

query_2 = (
    "What is biochemical oxygen demand "
    "at 5 days and 20 degrees Celsius?"
)

print(f"\nQuery: {query_2}")

results_2 = retriever.search(
    query_2,
    top_k=5,
)

print(
    f"Results returned: {len(results_2)}"
)

assert len(results_2) == 5

for result in results_2:
    metadata = result["metadata"]

    print("\n" + "-" * 60)

    print(
        f"Rank: {result['rank']}"
    )

    print(
        f"Chunk ID: {result['chunk_id']}"
    )

    print(
        f"Distance: {result['distance']:.6f}"
    )

    print(
        f"Section: "
        f"{metadata.get('section_type')}"
    )

    print(
        f"Clause: "
        f"{metadata.get('clause_number')}"
    )

    print(
        f"Table: "
        f"{metadata.get('table_number')}"
    )

    print(
        f"Pages: "
        f"{metadata.get('page_start')}"
        f"-"
        f"{metadata.get('page_end')}"
    )

    print(
        f"Text:\n{result['text'][:500]}"
    )


print("\n" + "=" * 60)
print("SEMANTIC RETRIEVAL TEST PASSED")
print("=" * 60)
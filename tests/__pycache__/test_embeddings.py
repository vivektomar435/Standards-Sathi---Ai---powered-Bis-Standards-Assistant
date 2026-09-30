from retrieval.embeddings import EmbeddingModel


def main():
    print("=" * 60)
    print("EMBEDDING MODEL TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1 — LOAD MODEL
    # ---------------------------------------------------------

    print("\nSTEP 1 — LOADING MODEL")

    embedding_model = EmbeddingModel()

    # ---------------------------------------------------------
    # STEP 2 — CHECK DIMENSION
    # ---------------------------------------------------------

    print("\nSTEP 2 — EMBEDDING DIMENSION")

    dimension = embedding_model.dimension()

    print(f"Embedding dimension: {dimension}")

    # ---------------------------------------------------------
    # STEP 3 — DOCUMENT EMBEDDING
    # ---------------------------------------------------------

    print("\nSTEP 3 — DOCUMENT EMBEDDING")

    texts = [
        "This standard provides guidelines for treatment of effluents from dairy industry.",
        "Biochemical oxygen demand is an important parameter in wastewater treatment.",
        "The treatment system should be designed according to the characteristics of the effluent.",
    ]

    embeddings = embedding_model.embed_texts(texts)

    print(f"Input texts: {len(texts)}")
    print(f"Generated embeddings: {len(embeddings)}")

    # ---------------------------------------------------------
    # STEP 4 — VALIDATION
    # ---------------------------------------------------------

    print("\nSTEP 4 — VALIDATION")

    assert len(embeddings) == len(texts)

    for index, embedding in enumerate(embeddings):
        assert len(embedding) == dimension

        print(
            f"Text {index + 1}: "
            f"{len(embedding)} dimensions"
        )

    # ---------------------------------------------------------
    # STEP 5 — QUERY EMBEDDING
    # ---------------------------------------------------------

    print("\nSTEP 5 — QUERY EMBEDDING")

    query = "What are the guidelines for dairy wastewater treatment?"

    query_embedding = embedding_model.embed_query(query)

    print(f"Query: {query}")
    print(f"Query vector dimensions: {len(query_embedding)}")

    assert len(query_embedding) == dimension

    print("\n" + "=" * 60)
    print("EMBEDDING TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
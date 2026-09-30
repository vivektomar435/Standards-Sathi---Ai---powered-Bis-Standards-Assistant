from ingestion.pdf_loader import load_pdf
from ingestion.cleaner import clean_pages
from ingestion.chunker import chunk_document
from ingestion.metadata import (
    build_document_metadata,
    prepare_chunks_for_embedding,
)


PDF_PATH = "documents/standards/8682_2026.pdf"


def main():
    print("=" * 60)
    print("METADATA PIPELINE TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # STEP 1 — LOAD
    # ---------------------------------------------------------

    print("\nSTEP 1 — PDF LOADING")

    document = load_pdf(PDF_PATH)

    print(f"File: {document['file_name']}")
    print(f"Pages: {document['page_count']}")

    # ---------------------------------------------------------
    # STEP 2 — CLEAN
    # ---------------------------------------------------------

    print("\nSTEP 2 — CLEANING")

    cleaned_pages = clean_pages(document["pages"])

    print(f"Cleaned pages: {len(cleaned_pages)}")

    # ---------------------------------------------------------
    # STEP 3 — CHUNK
    # ---------------------------------------------------------

    print("\nSTEP 3 — CHUNKING")

    chunks = chunk_document(cleaned_pages)

    print(f"Chunks: {len(chunks)}")

    # ---------------------------------------------------------
    # STEP 4 — DOCUMENT METADATA
    # ---------------------------------------------------------

    print("\nSTEP 4 — DOCUMENT METADATA")

    document_metadata = build_document_metadata(
        document["file_name"]
    )

    for key, value in document_metadata.items():
        print(f"{key}: {value}")

    # ---------------------------------------------------------
    # STEP 5 — CHUNK METADATA
    # ---------------------------------------------------------

    print("\nSTEP 5 — PREPARING CHUNKS")

    prepared_chunks = prepare_chunks_for_embedding(
        chunks=chunks,
        file_name=document["file_name"],
    )

    print(f"Prepared chunks: {len(prepared_chunks)}")

    # ---------------------------------------------------------
    # STEP 6 — VALIDATION
    # ---------------------------------------------------------

    print("\nSTEP 6 — METADATA VALIDATION")

    required_keys = [
        "standard_number",
        "standard_year",
        "document_type",
        "department",
        "file_name",
        "chunk_id",
        "section_type",
        "page_start",
        "page_end",
        "clause_number",
        "clause_title",
        "table_number",
        "annex_number",
        "annex_title",
    ]

    for index, item in enumerate(prepared_chunks, start=1):

        assert item["text"], (
            f"Chunk {index} has empty text"
        )

        metadata = item["metadata"]

        for key in required_keys:
            assert key in metadata, (
                f"Chunk {index} missing metadata key: {key}"
            )

        assert metadata["chunk_id"], (
            f"Chunk {index} has no chunk_id"
        )

        assert metadata["page_start"] is not None, (
            f"Chunk {index} has no page_start"
        )

        assert metadata["page_end"] is not None, (
            f"Chunk {index} has no page_end"
        )

    print("All metadata checks passed.")

    # ---------------------------------------------------------
    # STEP 7 — SAMPLE
    # ---------------------------------------------------------

    print("\nSTEP 7 — SAMPLE METADATA")

    for item in prepared_chunks[:5]:

        print("\n" + "-" * 60)

        print("TEXT:")
        print(item["text"][:300])

        print("\nMETADATA:")

        for key, value in item["metadata"].items():
            print(f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("METADATA PIPELINE READY")
    print("=" * 60)


if __name__ == "__main__":
    main()
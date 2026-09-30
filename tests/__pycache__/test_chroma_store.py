from ingestion.pdf_loader import load_pdf
from ingestion.cleaner import clean_pages
from ingestion.chunker import chunk_document
from ingestion.metadata import prepare_chunks_for_embedding
from retrieval.embeddings import EmbeddingModel
from database.chroma_store import ChromaStore


PDF_PATH = "documents/standards/8682_2026.pdf"


print("=" * 60)
print("CHROMADB INGESTION TEST")
print("=" * 60)


# ---------------------------------------------------------
# STEP 1 — LOAD PDF
# ---------------------------------------------------------

print("\nSTEP 1 — PDF LOADING")

document = load_pdf(PDF_PATH)

print(f"File: {document['file_name']}")
print(f"Pages: {document['page_count']}")


# ---------------------------------------------------------
# STEP 2 — CLEAN
# ---------------------------------------------------------

print("\nSTEP 2 — CLEANING")

cleaned_pages = clean_pages(
    document["pages"]
)

print(
    f"Cleaned pages: {len(cleaned_pages)}"
)


# ---------------------------------------------------------
# STEP 3 — CHUNK
# ---------------------------------------------------------

print("\nSTEP 3 — CHUNKING")

chunks = chunk_document(
    cleaned_pages
)

print(f"Chunks: {len(chunks)}")


# ---------------------------------------------------------
# STEP 4 — METADATA
# ---------------------------------------------------------

print("\nSTEP 4 — METADATA")

prepared_chunks = prepare_chunks_for_embedding(
    chunks,
    document["file_name"],
)

print(
    f"Prepared chunks: {len(prepared_chunks)}"
)


# ---------------------------------------------------------
# STEP 5 — EMBEDDINGS
# ---------------------------------------------------------

print("\nSTEP 5 — EMBEDDINGS")

embedding_model = EmbeddingModel()

texts = [
    chunk["text"]
    for chunk in prepared_chunks
]

embeddings = embedding_model.embed_texts(
    texts
)

print(
    f"Generated embeddings: {len(embeddings)}"
)

print(
    f"Embedding dimension: "
    f"{len(embeddings[0])}"
)


# ---------------------------------------------------------
# STEP 6 — CHROMADB
# ---------------------------------------------------------

print("\nSTEP 6 — CHROMADB")

store = ChromaStore(
    db_path="chroma_db",
    collection_name="bis_standards_test",
)

before_count = store.count()

print(
    f"Records before ingestion: {before_count}"
)


# ---------------------------------------------------------
# STEP 7 — INSERT
# ---------------------------------------------------------

print("\nSTEP 7 — INSERTING CHUNKS")

processed = store.add_chunks(
    prepared_chunks,
    embeddings,
)

print(
    f"Processed chunks: {processed}"
)

after_count = store.count()

print(
    f"Records after ingestion: {after_count}"
)


# ---------------------------------------------------------
# STEP 8 — VALIDATION
# ---------------------------------------------------------

print("\nSTEP 8 — VALIDATION")

assert after_count == len(prepared_chunks)

print(
    "Record count validation passed."
)


# ---------------------------------------------------------
# STEP 9 — CHECK SAMPLE RECORD
# ---------------------------------------------------------

print("\nSTEP 9 — SAMPLE RECORD")

sample = store.get_by_ids(
    ["IS8682_2026_chunk_0001"]
)

print(
    f"Returned IDs: {sample['ids']}"
)

print(
    f"Documents returned: "
    f"{len(sample['documents'])}"
)

print(
    f"Metadata returned: "
    f"{len(sample['metadatas'])}"
)


assert len(sample["ids"]) == 1
assert len(sample["documents"]) == 1
assert len(sample["metadatas"]) == 1

print(
    "Sample record validation passed."
)


# ---------------------------------------------------------
# STEP 10 — DUPLICATE-SAFE TEST
# ---------------------------------------------------------

print("\nSTEP 10 — DUPLICATE-SAFE TEST")

store.add_chunks(
    prepared_chunks,
    embeddings,
)

final_count = store.count()

print(
    f"Records after second ingestion: "
    f"{final_count}"
)

assert final_count == len(prepared_chunks)

print(
    "Duplicate-safe ingestion passed."
)


print("\n" + "=" * 60)
print("CHROMADB TEST PASSED")
print("=" * 60)
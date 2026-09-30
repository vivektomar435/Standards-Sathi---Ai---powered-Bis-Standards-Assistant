from pathlib import Path

import chromadb


DEFAULT_DB_PATH = "chroma_db"
DEFAULT_COLLECTION_NAME = "bis_standards"


class ChromaStore:
    def __init__(
        self,
        db_path=DEFAULT_DB_PATH,
        collection_name=DEFAULT_COLLECTION_NAME,
    ):
        self.db_path = Path(db_path)
        self.collection_name = collection_name

        self.db_path.mkdir(parents=True, exist_ok=True)

        print(f"Initializing ChromaDB at: {self.db_path}")

        self.client = chromadb.PersistentClient(
            path=str(self.db_path)
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            },
        )

        print(
            f"Chroma collection ready: {self.collection_name}"
        )
        print(
            f"Existing records: {self.collection.count()}"
        )

    def _sanitize_metadata(self, metadata):
        """
        Chroma metadata values cannot be None.

        Remove None-valued fields while preserving all useful
        metadata fields.
        """
        sanitized = {}

        for key, value in metadata.items():
            if value is None:
                continue

            sanitized[key] = value

        return sanitized

    def _make_chroma_id(self, metadata, index):
        """
        Generate a stable ID for Chroma.

        Example:
        IS8682_2026_chunk_0001
        """

        standard_number = metadata.get(
            "standard_number",
            "UNKNOWN",
        )

        standard_year = metadata.get(
            "standard_year",
            "UNKNOWN",
        )

        standard_number = (
            str(standard_number)
            .replace(" ", "")
            .replace("/", "_")
        )

        standard_year = str(standard_year)

        return (
            f"{standard_number}_"
            f"{standard_year}_"
            f"chunk_{index:04d}"
        )

    def add_chunks(
        self,
        prepared_chunks,
        embeddings,
        batch_size=100,
    ):
        """
        Add prepared chunks and their embeddings to ChromaDB.

        prepared_chunks:
            List of:
            {
                "text": "...",
                "metadata": {...}
            }

        embeddings:
            List of embedding vectors.

        Returns:
            Number of records added/updated.
        """

        if len(prepared_chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        if not prepared_chunks:
            return 0

        total_processed = 0

        for batch_start in range(
            0,
            len(prepared_chunks),
            batch_size,
        ):
            batch_chunks = prepared_chunks[
                batch_start:batch_start + batch_size
            ]

            batch_embeddings = embeddings[
                batch_start:batch_start + batch_size
            ]

            ids = []
            documents = []
            metadatas = []

            for local_index, chunk in enumerate(
                batch_chunks,
                start=batch_start + 1,
            ):
                metadata = dict(
                    chunk.get("metadata", {})
                )

                chroma_metadata = (
                    self._sanitize_metadata(metadata)
                )

                chunk_id = self._make_chroma_id(
                    chroma_metadata,
                    local_index,
                )

                ids.append(chunk_id)

                documents.append(
                    chunk.get("text", "")
                )

                metadatas.append(
                    chroma_metadata
                )

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=batch_embeddings,
                metadatas=metadatas,
            )

            total_processed += len(ids)

            print(
                f"Processed {total_processed}/"
                f"{len(prepared_chunks)} chunks"
            )

        return total_processed

    def count(self):
        """Return the number of stored chunks."""
        return self.collection.count()

    def get_by_ids(self, ids):
        """Retrieve specific chunks by their Chroma IDs."""
        if not ids:
            return {
                "ids": [],
                "documents": [],
                "metadatas": [],
            }

        return self.collection.get(
            ids=ids,
            include=[
                "documents",
                "metadatas",
            ],
        )

    def get_all(self):
        """Retrieve all stored chunks."""
        return self.collection.get(
            include=[
                "documents",
                "metadatas",
            ],
        )

    def delete_by_ids(self, ids):
        """Delete specific chunks from the database."""
        if not ids:
            return

        self.collection.delete(ids=ids)

    def reset(self):
        """
        Delete and recreate the collection.

        Use carefully. This removes all stored vectors.
        """
        self.client.delete_collection(
            name=self.collection_name
        )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            configuration={
                "hnsw": {
                    "space": "cosine"
                }
            },
        )

        print(
            f"Collection reset: {self.collection_name}"
        )
from database.chroma_store import ChromaStore
from retrieval.embeddings import EmbeddingModel


DEFAULT_TOP_K = 5


class SemanticRetriever:
    def __init__(
        self,
        db_path="chroma_db",
        collection_name="bis_standards_test",
        embedding_model_name="all-MiniLM-L6-v2",
    ):
        print("Initializing semantic retriever...")

        self.embedding_model = EmbeddingModel(
            model_name=embedding_model_name
        )

        self.store = ChromaStore(
            db_path=db_path,
            collection_name=collection_name,
        )

        print("Semantic retriever ready.")

    def search(
        self,
        query,
        top_k=DEFAULT_TOP_K,
    ):
        """
        Perform semantic similarity search.

        Returns the most relevant BIS chunks for the query.
        """

        if not query or not query.strip():
            return []

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        query = query.strip()

        query_embedding = (
            self.embedding_model.embed_query(query)
        )

        results = self.store.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        return self._format_results(results)

    def _format_results(self, results):
        """
        Convert Chroma's nested response into a
        simple list of retrieval results.
        """

        formatted_results = []

        ids = results.get("ids", [[]])[0]
        documents = results.get(
            "documents",
            [[]],
        )[0]
        metadatas = results.get(
            "metadatas",
            [[]],
        )[0]
        distances = results.get(
            "distances",
            [[]],
        )[0]

        for index, chunk_id in enumerate(ids):
            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            document = (
                documents[index]
                if index < len(documents)
                else ""
            )

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            formatted_results.append({
                "rank": index + 1,
                "chunk_id": chunk_id,
                "text": document,
                "metadata": metadata,
                "distance": distance,
            })

        return formatted_results
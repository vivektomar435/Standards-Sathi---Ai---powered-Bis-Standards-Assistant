from sentence_transformers import SentenceTransformer


DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:

    def __init__(self, model_name=DEFAULT_MODEL_NAME):
        self.model_name = model_name

        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        print("Embedding model loaded.")

    def embed_texts(self, texts):

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

        return embeddings.tolist()

    def embed_query(self, query):

        if not query:
            return []

        embedding = self.model.encode(
            query,
            convert_to_numpy=True,
        )

        return embedding.tolist()

    def dimension(self):
        return self.model.get_embedding_dimension()
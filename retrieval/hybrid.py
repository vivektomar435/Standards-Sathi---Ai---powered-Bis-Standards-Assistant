import re

from database.chroma_store import ChromaStore
from retrieval.embeddings import EmbeddingModel


DEFAULT_TOP_K = 5
DEFAULT_SEMANTIC_CANDIDATES = 15

# Hybrid scoring weights
SEMANTIC_WEIGHT = 0.55
KEYWORD_WEIGHT = 0.15
STRUCTURAL_WEIGHT = 0.30


class HybridRetriever:
    def __init__(
        self,
        db_path="chroma_db",
        collection_name="bis_standards_test",
        embedding_model_name="all-MiniLM-L6-v2",
        semantic_candidates=DEFAULT_SEMANTIC_CANDIDATES,
    ):
        print("Initializing hybrid retriever...")

        self.embedding_model = EmbeddingModel(
            model_name=embedding_model_name
        )

        self.store = ChromaStore(
            db_path=db_path,
            collection_name=collection_name,
        )

        self.semantic_candidates = semantic_candidates

        print("Hybrid retriever ready.")

    # =========================================================
    # PUBLIC SEARCH
    # =========================================================

    def search(
        self,
        query,
        top_k=DEFAULT_TOP_K,
        candidate_k=None,
    ):
        if not query or not query.strip():
            return []

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        # Keep compatibility with the existing test file.
        if candidate_k is None:
            candidate_k = self.semantic_candidates

        if candidate_k < 1:
            raise ValueError("candidate_k must be at least 1.")

        query = query.strip()

        # -----------------------------------------------------
        # 1. Semantic candidates
        # -----------------------------------------------------

        semantic_results = self._semantic_candidates(
            query,
            candidate_k,
        )

        # -----------------------------------------------------
        # 2. Lexical candidates
        # -----------------------------------------------------

        lexical_results = self._lexical_candidates(
            query
        )

        # -----------------------------------------------------
        # 3. Structural candidates
        # -----------------------------------------------------

        structural_results = self._structural_candidates(
            query
        )

        # -----------------------------------------------------
        # 4. Merge candidate pools
        # -----------------------------------------------------

        candidates = self._merge_candidates(
            semantic_results,
            lexical_results,
            structural_results,
        )

        # -----------------------------------------------------
        # 5. Hybrid scoring
        # -----------------------------------------------------

        scored_results = []

        for candidate in candidates:

            metadata = candidate.get(
                "metadata",
                {},
            )

            semantic_score = candidate.get(
                "semantic_score",
                0.0,
            )

            keyword_score = self._keyword_score(
                query,
                candidate.get("text", ""),
                metadata,
            )

            structural_score = self._structural_score(
                query,
                metadata,
            )

            hybrid_score = (
                SEMANTIC_WEIGHT * semantic_score
                + KEYWORD_WEIGHT * keyword_score
                + STRUCTURAL_WEIGHT * structural_score
            )

            # -------------------------------------------------
            # Strong structural match protection
            #
            # Example:
            #
            # "terms and definitions"
            #       ↓
            # Clause 3 TERMINOLOGY
            #
            # This candidate might have semantic_score = 0
            # if it wasn't in the original semantic top-k.
            #
            # We therefore prevent a strong deterministic
            # structural match from being buried.
            # -------------------------------------------------

            if structural_score >= 1.0:
                hybrid_score = max(
                    hybrid_score,
                    0.75,
                )

            candidate["keyword_score"] = keyword_score
            candidate["structural_score"] = structural_score
            candidate["hybrid_score"] = hybrid_score

            scored_results.append(candidate)

        # -----------------------------------------------------
        # 6. Sort by hybrid score
        # -----------------------------------------------------

        scored_results.sort(
            key=lambda item: item["hybrid_score"],
            reverse=True,
        )

        # -----------------------------------------------------
        # 7. Return top-k
        # -----------------------------------------------------

        final_results = []

        for rank, result in enumerate(
            scored_results[:top_k],
            start=1,
        ):

            final_results.append(
                {
                    "rank": rank,
                    "chunk_id": result["chunk_id"],
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "distance": result.get(
                        "distance"
                    ),
                    "semantic_score": result.get(
                        "semantic_score",
                        0.0,
                    ),
                    "keyword_score": result.get(
                        "keyword_score",
                        0.0,
                    ),
                    "structural_score": result.get(
                        "structural_score",
                        0.0,
                    ),
                    "hybrid_score": result.get(
                        "hybrid_score",
                        0.0,
                    ),
                }
            )

        return final_results

    # =========================================================
    # SEMANTIC RETRIEVAL
    # =========================================================

    def _semantic_candidates(
        self,
        query,
        top_k,
    ):
        query_embedding = (
            self.embedding_model.embed_query(
                query
            )
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

        candidates = []

        ids = results.get(
            "ids",
            [[]],
        )[0]

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

            document = (
                documents[index]
                if index < len(documents)
                else ""
            )

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            distance = (
                distances[index]
                if index < len(distances)
                else None
            )

            semantic_score = (
                self._distance_to_score(
                    distance
                )
            )

            candidates.append(
                {
                    "chunk_id": chunk_id,
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                    "semantic_score": semantic_score,
                    "source": "semantic",
                }
            )

        return candidates

    # =========================================================
    # DISTANCE -> SIMILARITY SCORE
    # =========================================================

    def _distance_to_score(
        self,
        distance,
    ):
        """
        Chroma uses cosine distance.

        Lower distance = more similar.

        Convert distance to a simple similarity-like score.
        """

        if distance is None:
            return 0.0

        try:
            distance = float(distance)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        score = 1.0 - distance

        return max(
            0.0,
            min(1.0, score),
        )

    # =========================================================
    # LEXICAL RETRIEVAL
    # =========================================================

    def _lexical_candidates(
        self,
        query,
    ):
        """
        Scan all chunks and find chunks that share
        meaningful terms with the query.

        This is acceptable for the current small dataset.
        Later, for thousands of BIS documents, this can
        be replaced by BM25 or another indexed lexical
        retriever.
        """

        all_records = self.store.get_all()

        ids = all_records.get(
            "ids",
            [],
        )

        documents = all_records.get(
            "documents",
            [],
        )

        metadatas = all_records.get(
            "metadatas",
            [],
        )

        candidates = []

        for index, chunk_id in enumerate(ids):

            document = (
                documents[index]
                if index < len(documents)
                else ""
            )

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            keyword_score = (
                self._keyword_score(
                    query,
                    document,
                    metadata,
                )
            )

            if keyword_score > 0:

                candidates.append(
                    {
                        "chunk_id": chunk_id,
                        "text": document,
                        "metadata": metadata,
                        "distance": 1.0,
                        "semantic_score": 0.0,
                        "source": "lexical",
                    }
                )

        return candidates

    # =========================================================
    # STRUCTURAL RETRIEVAL
    # =========================================================

    def _structural_candidates(
        self,
        query,
    ):
        """
        Find chunks using known BIS document structure.

        Current deterministic mappings:

            terms / definitions / terminology
                -> Clause 3

            medium dairy
                -> Table 2

            large dairy
                -> Table 3

            composite / three dairies
                -> Table 4

            referred standards
                -> Annex A
        """

        all_records = self.store.get_all()

        ids = all_records.get(
            "ids",
            [],
        )

        documents = all_records.get(
            "documents",
            [],
        )

        metadatas = all_records.get(
            "metadatas",
            [],
        )

        matches = []

        query_lower = query.lower()

        query_terms = self._normalize_terms(
            query
        )

        for index, chunk_id in enumerate(ids):

            document = (
                documents[index]
                if index < len(documents)
                else ""
            )

            metadata = (
                metadatas[index]
                if index < len(metadatas)
                else {}
            )

            if self._is_structural_match(
                query_lower,
                query_terms,
                metadata,
            ):

                matches.append(
                    {
                        "chunk_id": chunk_id,
                        "text": document,
                        "metadata": metadata,
                        "distance": 1.0,
                        "semantic_score": 0.0,
                        "source": "structural",
                    }
                )

        return matches

    # =========================================================
    # STRUCTURAL MATCH DETECTION
    # =========================================================

    def _is_structural_match(
        self,
        query_lower,
        query_terms,
        metadata,
    ):
        section_type = str(
            metadata.get(
                "section_type"
            )
            or ""
        ).lower()

        clause_number = str(
            metadata.get(
                "clause_number"
            )
            or ""
        ).lower()

        clause_title = str(
            metadata.get(
                "clause_title"
            )
            or ""
        ).lower()

        table_number = str(
            metadata.get(
                "table_number"
            )
            or ""
        ).lower()

        annex_number = str(
            metadata.get(
                "annex_number"
            )
            or ""
        ).lower()

        annex_title = str(
            metadata.get(
                "annex_title"
            )
            or ""
        ).lower()

        # -----------------------------------------------------
        # TERMINOLOGY -> CLAUSE 3
        # -----------------------------------------------------

        terminology_words = {
            "term",
            "definition",
            "terminology",
        }

        if query_terms.intersection(
            terminology_words
        ):

            if section_type == "clause":

                if clause_number == "3":
                    return True

                if "terminology" in clause_title:
                    return True

        # -----------------------------------------------------
        # MEDIUM DAIRY -> TABLE 2
        # -----------------------------------------------------

        has_dairy = (
            "dairy" in query_terms
        )

        if (
            has_dairy
            and "medium" in query_terms
            and section_type == "table"
            and table_number == "2"
        ):
            return True

        # -----------------------------------------------------
        # LARGE DAIRY -> TABLE 3
        # -----------------------------------------------------

        if (
            has_dairy
            and "large" in query_terms
            and section_type == "table"
            and table_number == "3"
        ):
            return True

        # -----------------------------------------------------
        # COMPOSITE / THREE DAIRIES -> TABLE 4
        # -----------------------------------------------------

        if (
            section_type == "table"
            and table_number == "4"
        ):

            if "composite" in query_terms:
                return True

            if (
                "three" in query_terms
                and "dairy" in query_terms
            ):
                return True

        # -----------------------------------------------------
        # REFERRED STANDARDS -> ANNEX A
        # -----------------------------------------------------

        if (
            "referred" in query_terms
            and section_type == "annex"
        ):

            if annex_number == "a":
                return True

            if (
                "referred standards"
                in annex_title
            ):
                return True

        return False

    # =========================================================
    # MERGE CANDIDATES
    # =========================================================

    def _merge_candidates(
        self,
        semantic_candidates,
        lexical_candidates,
        structural_candidates,
    ):
        """
        Merge semantic, lexical, and structural candidates
        by chunk_id.

        A chunk can belong to multiple candidate sources.
        """

        merged = {}

        # -----------------------------------------------------
        # Semantic candidates
        # -----------------------------------------------------

        for candidate in semantic_candidates:

            chunk_id = candidate[
                "chunk_id"
            ]

            merged[chunk_id] = dict(
                candidate
            )

        # -----------------------------------------------------
        # Lexical candidates
        # -----------------------------------------------------

        for candidate in lexical_candidates:

            chunk_id = candidate[
                "chunk_id"
            ]

            if chunk_id not in merged:

                merged[chunk_id] = dict(
                    candidate
                )

            else:

                existing_source = (
                    merged[chunk_id].get(
                        "source",
                        "",
                    )
                )

                if "lexical" not in existing_source:

                    merged[chunk_id][
                        "source"
                    ] = (
                        existing_source
                        + "+lexical"
                    )

        # -----------------------------------------------------
        # Structural candidates
        # -----------------------------------------------------

        for candidate in structural_candidates:

            chunk_id = candidate[
                "chunk_id"
            ]

            if chunk_id not in merged:

                merged[chunk_id] = dict(
                    candidate
                )

            else:

                existing_source = (
                    merged[chunk_id].get(
                        "source",
                        "",
                    )
                )

                if (
                    "structural"
                    not in existing_source
                ):

                    merged[chunk_id][
                        "source"
                    ] = (
                        existing_source
                        + "+structural"
                    )

        return list(
            merged.values()
        )

    # =========================================================
    # KEYWORD SCORE
    # =========================================================

    def _keyword_score(
        self,
        query,
        text,
        metadata,
    ):
        """
        Calculate lexical overlap.

        Terms are normalized so:

            dairy / dairies
            term / terms
            definition / definitions
            effluent / effluents

        are treated as the same concept.
        """

        query_terms = (
            self._normalize_terms(
                query
            )
        )

        if not query_terms:
            return 0.0

        searchable_text = " ".join(
            [
                text or "",
                str(
                    metadata.get(
                        "clause_title",
                        "",
                    )
                ),
                str(
                    metadata.get(
                        "annex_title",
                        "",
                    )
                ),
            ]
        )

        document_terms = (
            self._normalize_terms(
                searchable_text
            )
        )

        if not document_terms:
            return 0.0

        matches = query_terms.intersection(
            document_terms
        )

        return (
            len(matches)
            / len(query_terms)
        )

    # =========================================================
    # STRUCTURAL SCORE
    # =========================================================

    def _structural_score(
        self,
        query,
        metadata,
    ):
        """
        Calculate structural relevance.

        1.0 = strong deterministic match
        0.5 = general structural hint
        0.0 = no structural match
        """

        query_lower = query.lower()

        query_terms = (
            self._normalize_terms(
                query
            )
        )

        section_type = str(
            metadata.get(
                "section_type"
            )
            or ""
        ).lower()

        clause_number = str(
            metadata.get(
                "clause_number"
            )
            or ""
        ).lower()

        clause_title = str(
            metadata.get(
                "clause_title"
            )
            or ""
        ).lower()

        table_number = str(
            metadata.get(
                "table_number"
            )
            or ""
        ).lower()

        annex_number = str(
            metadata.get(
                "annex_number"
            )
            or ""
        ).lower()

        annex_title = str(
            metadata.get(
                "annex_title"
            )
            or ""
        ).lower()

        # -----------------------------------------------------
        # TERMINOLOGY -> CLAUSE 3
        # -----------------------------------------------------

        terminology_words = {
            "term",
            "definition",
            "terminology",
        }

        if query_terms.intersection(
            terminology_words
        ):

            if section_type == "clause":

                if clause_number == "3":
                    return 1.0

                if "terminology" in clause_title:
                    return 1.0

        # -----------------------------------------------------
        # DAIRY TABLES
        # -----------------------------------------------------

        has_dairy = (
            "dairy" in query_terms
        )

        if has_dairy:

            # Medium dairy -> Table 2
            if (
                "medium" in query_terms
                and section_type == "table"
                and table_number == "2"
            ):
                return 1.0

            # Large dairy -> Table 3
            if (
                "large" in query_terms
                and section_type == "table"
                and table_number == "3"
            ):
                return 1.0

            # Composite dairy -> Table 4
            if (
                section_type == "table"
                and table_number == "4"
            ):

                if "composite" in query_terms:
                    return 1.0

                if (
                    "three" in query_terms
                    and "dairy" in query_terms
                ):
                    return 1.0

        # -----------------------------------------------------
        # REFERRED STANDARDS -> ANNEX A
        # -----------------------------------------------------

        if "referred" in query_terms:

            if section_type == "annex":

                if annex_number == "a":
                    return 1.0

                if (
                    "referred standards"
                    in annex_title
                ):
                    return 1.0

        # -----------------------------------------------------
        # GENERAL STRUCTURAL HINTS
        # -----------------------------------------------------

        if (
            "table" in query_lower
            and section_type == "table"
        ):
            return 0.5

        if (
            "annex" in query_lower
            and section_type == "annex"
        ):
            return 0.5

        if (
            "clause" in query_lower
            and section_type == "clause"
        ):
            return 0.5

        return 0.0

    # =========================================================
    # TERM NORMALIZATION
    # =========================================================

    def _normalize_terms(
        self,
        text,
    ):
        """
        Convert text into canonical searchable terms.

        Examples:

            dairies       -> dairy
            definitions   -> definition
            terms         -> term
            effluents     -> effluent
            standards     -> standard
        """

        if not text:
            return set()

        text = text.lower()

        raw_terms = re.findall(
            r"[a-zA-Z0-9]+(?:\.[0-9]+)*",
            text,
        )

        normalized = set()

        stop_words = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "what",
            "which",
            "who",
            "how",
            "for",
            "of",
            "to",
            "in",
            "on",
            "and",
            "or",
            "this",
            "that",
            "these",
            "those",
            "used",
            "use",
            "from",
            "with",
            "at",
            "by",
            "be",
            "as",
        }

        plural_map = {
            "dairies": "dairy",
            "terms": "term",
            "definitions": "definition",
            "effluents": "effluent",
            "standards": "standard",
            "methods": "method",
            "products": "product",
            "wastes": "waste",
            "treatments": "treatment",
            "characteristics": "characteristic",
            "guidelines": "guideline",
            "references": "reference",
        }

        for term in raw_terms:

            if term in stop_words:
                continue

            if term in plural_map:
                term = plural_map[term]

            normalized.add(term)

        return normalized
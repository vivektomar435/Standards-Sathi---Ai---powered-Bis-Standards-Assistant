"""
Scalable multi-standard ingestion manager for the BIS RAG chatbot.

Pipeline:

    PDF
      ↓
    PDF Loader
      ↓
    Structure-aware Chunker
      ↓
    Metadata
      ↓
    Document Registry
      ↓
    Embeddings
      ↓
    ChromaDB

Design goals:
- Reuse the project's existing ingestion/retrieval components.
- Be safe for repeated ingestion.
- Detect duplicate documents using SHA-256.
- Keep registry status synchronized with ingestion progress.
- Continue batch ingestion when one PDF fails.
- Provide dry-run, validation, health, and summary helpers.
- Preserve the current prototype collection by default.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging
import time

from ingestion.pdf_loader import load_pdf
from ingestion.chunker import StructureAwareChunker
from ingestion.metadata import (
    extract_standard_info,
    prepare_chunks_for_embedding,
)
from database.registry import DocumentRegistry
from database.chroma_store import ChromaStore
from retrieval.embeddings import EmbeddingModel


logger = logging.getLogger(__name__)


class IngestionManager:
    """
    Coordinates ingestion of one or more BIS PDF standards.

    Registry lifecycle:

        pending → processing → completed
                           ↘ failed

    The same PDF can safely be submitted repeatedly. If its SHA-256 hash
    already exists, it is skipped unless force=True.
    """

    def __init__(
        self,
        documents_path: str | Path = "documents/standards",
        chroma_path: str | Path = "chroma_db",
        collection_name: str = "bis_standards_test",
        registry_path: str | Path = "data/document_registry.db",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.documents_path = Path(documents_path)
        self.documents_path.mkdir(parents=True, exist_ok=True)

        self.registry = DocumentRegistry(
            db_path=str(registry_path)
        )

        self.chroma = ChromaStore(
            persist_directory=str(chroma_path),
            collection_name=collection_name,
        )

        self.embedding_model = EmbeddingModel(
            model_name=embedding_model
        )

        self.chunker = StructureAwareChunker()
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        self.chroma_path = Path(chroma_path)
        self.registry_path = Path(registry_path)

        logger.info(
            "IngestionManager initialized | documents=%s | "
            "collection=%s | embedding=%s",
            self.documents_path,
            self.collection_name,
            self.embedding_model_name,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_pdf_path(pdf_path: str | Path) -> Path:
        """Validate and normalize a single PDF path."""
        path = Path(pdf_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF file does not exist: {path}")

        if not path.is_file():
            raise ValueError(f"Path is not a file: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a PDF file, got: {path}")

        return path

    @staticmethod
    def _validate_loaded_document(
        document: Dict[str, Any],
        pdf_path: Path,
    ) -> tuple[int, list]:
        """Validate the object returned by load_pdf()."""
        if not isinstance(document, dict):
            raise TypeError(
                f"PDF loader returned {type(document).__name__}; "
                "expected dict."
            )

        pages = document.get("pages", [])
        page_count = document.get("page_count", len(pages))

        if not isinstance(pages, list):
            raise TypeError(
                f"PDF loader returned invalid pages for {pdf_path.name}."
            )

        if not pages:
            raise ValueError(
                f"No text pages were extracted from {pdf_path.name}."
            )

        try:
            page_count = int(page_count)
        except (TypeError, ValueError):
            page_count = len(pages)

        if page_count <= 0:
            page_count = len(pages)

        return page_count, pages

    @staticmethod
    def _validate_chunks(
        chunks: Any,
        pdf_path: Path,
    ) -> list[dict[str, Any]]:
        """Validate chunks before embeddings are generated."""
        if not isinstance(chunks, list):
            raise TypeError(
                f"Chunker returned {type(chunks).__name__}; expected list."
            )

        if not chunks:
            raise ValueError(
                f"No chunks were created from {pdf_path.name}."
            )

        valid_chunks: list[dict[str, Any]] = []

        for index, chunk in enumerate(chunks, start=1):
            if not isinstance(chunk, dict):
                raise TypeError(
                    f"Chunk {index} from {pdf_path.name} is not a dict."
                )

            text = str(chunk.get("text", "")).strip()
            if not text:
                logger.warning(
                    "Ignoring empty chunk %s from %s",
                    index,
                    pdf_path.name,
                )
                continue

            chunk["text"] = text
            valid_chunks.append(chunk)

        if not valid_chunks:
            raise ValueError(
                f"All chunks were empty for {pdf_path.name}."
            )

        return valid_chunks

    # ------------------------------------------------------------------
    # Single PDF ingestion
    # ------------------------------------------------------------------

    def ingest_file(
        self,
        pdf_path: str | Path,
        force: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Ingest one BIS PDF.

        Parameters
        ----------
        pdf_path:
            PDF to ingest.

        force:
            Re-process an already registered identical document.

        dry_run:
            Validate the file, metadata and extraction without writing
            embeddings/Chroma or changing the registry.
        """
        started = time.perf_counter()
        pdf_path = self._validate_pdf_path(pdf_path)

        logger.info("Starting ingestion: %s", pdf_path)

        # 1. Identify standard
        standard_info = extract_standard_info(pdf_path.name)

        standard_number = standard_info.get("standard_number")
        standard_year = standard_info.get("standard_year")

        if not standard_number or not standard_year:
            raise ValueError(
                "Could not determine standard number/year from filename: "
                f"{pdf_path.name}"
            )

        logger.info(
            "Detected standard: IS %s:%s",
            standard_number,
            standard_year,
        )

        # 2. Hash before doing expensive embedding work
        document_hash = self.registry.calculate_file_hash(pdf_path)

        # 3. Duplicate detection
        existing = self.registry.get_document(
            standard_number=standard_number,
            standard_year=standard_year,
            document_hash=document_hash,
        )

        if existing and not force:
            logger.info(
                "Skipping already-ingested document: %s",
                pdf_path.name,
            )

            return {
                "status": "skipped",
                "reason": "already_ingested",
                "standard_number": standard_number,
                "standard_year": standard_year,
                "file_name": pdf_path.name,
                "document_id": existing.get("id"),
                "document_hash": document_hash,
                "elapsed_seconds": round(
                    time.perf_counter() - started, 3
                ),
            }

        # 4. Load PDF
        logger.info("Loading PDF: %s", pdf_path)

        document = load_pdf(pdf_path)
        page_count, pages = self._validate_loaded_document(
            document,
            pdf_path,
        )

        logger.info(
            "Loaded %s pages from %s",
            page_count,
            pdf_path.name,
        )

        # 5. Chunk
        logger.info("Creating structure-aware chunks...")

        chunks = self.chunker.chunk_pages(pages)
        chunks = self._validate_chunks(chunks, pdf_path)

        logger.info(
            "Created %s valid chunks",
            len(chunks),
        )

        # 6. Metadata
        chunks = prepare_chunks_for_embedding(
            chunks=chunks,
            standard_info=standard_info,
        )

        # 7. Dry run ends here
        if dry_run:
            return {
                "status": "dry_run",
                "standard_number": standard_number,
                "standard_year": standard_year,
                "file_name": pdf_path.name,
                "document_hash": document_hash,
                "page_count": page_count,
                "chunk_count": len(chunks),
                "embedding_model": self.embedding_model_name,
                "elapsed_seconds": round(
                    time.perf_counter() - started, 3
                ),
            }

        # 8. Register or reuse existing record.
        #
        # Reusing the existing ID during force ingestion avoids creating
        # unnecessary duplicate registry rows for the same file hash.
        if existing:
            document_id = existing.get("id")

            if document_id is None:
                raise ValueError(
                    "Existing registry record has no document ID."
                )

            self.registry.update_status(
                document_id,
                "processing",
            )
        else:
            document_id = self.registry.register_document(
                standard_number=standard_number,
                standard_year=standard_year,
                title=standard_info.get(
                    "title",
                    f"IS {standard_number}:{standard_year}",
                ),
                filename=pdf_path.name,
                file_path=str(pdf_path),
                document_hash=document_hash,
                embedding_model=self.embedding_model_name,
                status="pending",
                version=str(standard_year),
            )

            self.registry.update_status(
                document_id,
                "processing",
            )

        try:
            # 9. Embeddings
            texts = [chunk["text"] for chunk in chunks]

            logger.info(
                "Generating embeddings for %s chunks...",
                len(texts),
            )

            embeddings = self.embedding_model.embed_texts(texts)

            if len(embeddings) != len(chunks):
                raise ValueError(
                    "Embedding count does not match chunk count: "
                    f"{len(embeddings)} embeddings for "
                    f"{len(chunks)} chunks."
                )

            logger.info(
                "Generated embeddings | dimension=%s",
                self.embedding_model.dimension,
            )

            # 10. Chroma upsert
            logger.info(
                "Storing %s chunks in ChromaDB...",
                len(chunks),
            )

            self.chroma.add_chunks(
                chunks=chunks,
                embeddings=embeddings,
            )

            logger.info(
                "Stored %s chunks in ChromaDB",
                len(chunks),
            )

            # 11. Registry success
            self.registry.update_ingestion_result(
                document_id=document_id,
                page_count=page_count,
                chunk_count=len(chunks),
                embedding_model=self.embedding_model_name,
                status="completed",
            )

            elapsed = round(
                time.perf_counter() - started,
                3,
            )

            logger.info(
                "Ingestion completed | IS %s:%s | %ss",
                standard_number,
                standard_year,
                elapsed,
            )

            return {
                "status": "completed",
                "document_id": document_id,
                "standard_number": standard_number,
                "standard_year": standard_year,
                "file_name": pdf_path.name,
                "document_hash": document_hash,
                "page_count": page_count,
                "chunk_count": len(chunks),
                "embedding_model": self.embedding_model_name,
                "elapsed_seconds": elapsed,
            }

        except Exception as exc:
            logger.exception(
                "Ingestion failed for %s",
                pdf_path,
            )

            # Registry failure must not hide the original exception.
            try:
                self.registry.update_status(
                    document_id,
                    "failed",
                )
            except Exception:
                logger.exception(
                    "Could not mark document %s as failed",
                    document_id,
                )

            return {
                "status": "failed",
                "document_id": document_id,
                "standard_number": standard_number,
                "standard_year": standard_year,
                "file_name": pdf_path.name,
                "document_hash": document_hash,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "elapsed_seconds": round(
                    time.perf_counter() - started,
                    3,
                ),
            }

    # ------------------------------------------------------------------
    # Directory ingestion
    # ------------------------------------------------------------------

    def ingest_directory(
        self,
        directory: Optional[str | Path] = None,
        force: bool = False,
        recursive: bool = False,
        continue_on_error: bool = True,
        max_files: Optional[int] = None,
        dry_run: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Ingest PDFs from a directory.

        By default, one failed PDF does not stop the remaining batch.
        """
        directory = (
            Path(directory)
            if directory is not None
            else self.documents_path
        )

        if not directory.exists():
            raise FileNotFoundError(
                f"Directory does not exist: {directory}"
            )

        if not directory.is_dir():
            raise ValueError(
                f"Expected a directory, got: {directory}"
            )

        if recursive:
            pdf_files = sorted(
                p for p in directory.rglob("*.pdf")
                if p.is_file()
            )
        else:
            pdf_files = sorted(
                p for p in directory.glob("*.pdf")
                if p.is_file()
            )

        if max_files is not None:
            if max_files <= 0:
                raise ValueError("max_files must be greater than zero.")
            pdf_files = pdf_files[:max_files]

        if not pdf_files:
            logger.warning(
                "No PDF files found in %s",
                directory,
            )
            return []

        logger.info(
            "Found %s PDF files in %s",
            len(pdf_files),
            directory,
        )

        results: List[Dict[str, Any]] = []

        for index, pdf_path in enumerate(pdf_files, start=1):
            logger.info(
                "Processing PDF %s/%s: %s",
                index,
                len(pdf_files),
                pdf_path.name,
            )

            try:
                result = self.ingest_file(
                    pdf_path,
                    force=force,
                    dry_run=dry_run,
                )
            except Exception as exc:
                logger.exception(
                    "Unexpected batch error for %s",
                    pdf_path,
                )

                result = {
                    "status": "failed",
                    "file_name": pdf_path.name,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }

                if not continue_on_error:
                    results.append(result)
                    raise

            results.append(result)

        summary = self.summarize_results(results)

        logger.info(
            "Batch ingestion finished | total=%s | completed=%s | "
            "skipped=%s | failed=%s",
            summary["total"],
            summary["completed"],
            summary["skipped"],
            summary["failed"],
        )

        return results

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------

    @staticmethod
    def summarize_results(
        results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Return aggregate statistics for an ingestion batch."""
        summary: Dict[str, Any] = {
            "total": len(results),
            "completed": 0,
            "skipped": 0,
            "dry_run": 0,
            "failed": 0,
            "pages": 0,
            "chunks": 0,
            "elapsed_seconds": 0.0,
        }

        for result in results:
            status = result.get("status")

            if status == "completed":
                summary["completed"] += 1
            elif status == "skipped":
                summary["skipped"] += 1
            elif status == "dry_run":
                summary["dry_run"] += 1
            elif status == "failed":
                summary["failed"] += 1

            summary["pages"] += int(result.get("page_count") or 0)
            summary["chunks"] += int(result.get("chunk_count") or 0)
            summary["elapsed_seconds"] += float(
                result.get("elapsed_seconds") or 0
            )

        summary["elapsed_seconds"] = round(
            summary["elapsed_seconds"],
            3,
        )

        return summary

    # ------------------------------------------------------------------
    # System information / health
    # ------------------------------------------------------------------

    def info(self) -> Dict[str, Any]:
        """Return current ingestion-system information."""
        registry_info = self.registry.info()

        return {
            "documents_path": str(self.documents_path),
            "registry_path": str(self.registry_path),
            "chroma_path": str(self.chroma_path),
            "collection_name": self.collection_name,
            "embedding_model": self.embedding_model_name,
            "embedding_dimension": self.embedding_model.dimension,
            "registry": registry_info,
            "chroma_count": self.chroma.count(),
        }

    def health_check(self) -> Dict[str, Any]:
        """
        Verify that the major ingestion dependencies can be initialized.

        This is intentionally lightweight: it does not ingest a document.
        """
        checks: Dict[str, Any] = {}

        try:
            checks["documents_path"] = {
                "ok": self.documents_path.exists()
            }
        except Exception as exc:
            checks["documents_path"] = {
                "ok": False,
                "error": str(exc),
            }

        try:
            checks["registry"] = {
                "ok": self.registry.info() is not None
            }
        except Exception as exc:
            checks["registry"] = {
                "ok": False,
                "error": str(exc),
            }

        try:
            checks["chroma"] = {
                "ok": self.chroma.count() >= 0,
                "count": self.chroma.count(),
            }
        except Exception as exc:
            checks["chroma"] = {
                "ok": False,
                "error": str(exc),
            }

        try:
            checks["embedding_model"] = {
                "ok": self.embedding_model.dimension > 0,
                "dimension": self.embedding_model.dimension,
            }
        except Exception as exc:
            checks["embedding_model"] = {
                "ok": False,
                "error": str(exc),
            }

        overall_ok = all(
            bool(check.get("ok"))
            for check in checks.values()
        )

        return {
            "ok": overall_ok,
            "checks": checks,
        }


# ----------------------------------------------------------------------
# Command-line usage
# ----------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    manager = IngestionManager()

    print("\n=== BIS Ingestion System ===")
    print(manager.info())

    print("\n=== Health Check ===")
    print(manager.health_check())

    print("\n=== Scanning for BIS PDFs ===\n")

    results = manager.ingest_directory(
        continue_on_error=True,
    )

    print("\n=== Ingestion Results ===")

    for result in results:
        print(result)

    print("\n=== Summary ===")
    print(manager.summarize_results(results))

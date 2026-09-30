from pathlib import Path
import re


DEFAULT_DEPARTMENT = "Environment and Ecology"
DEFAULT_DOCUMENT_TYPE = "Indian Standard"


def extract_standard_info(file_name):
    """
    Extract standard number and year from filenames such as:

        8682_2026.pdf
        IS_8682_2026.pdf
        IS 8682 2026.pdf
    """

    stem = Path(file_name).stem

    match = re.search(
        r"(?:IS[\s_-]*)?(\d+)[\s_-]+(\d{4})",
        stem,
        flags=re.IGNORECASE,
    )

    if not match:
        return {
            "standard_number": None,
            "standard_year": None,
        }

    return {
        "standard_number": f"IS {match.group(1)}",
        "standard_year": match.group(2),
    }


def build_document_metadata(
    file_name,
    department=DEFAULT_DEPARTMENT,
    document_type=DEFAULT_DOCUMENT_TYPE,
):
    """
    Build metadata shared by every chunk belonging to one PDF.
    """

    standard_info = extract_standard_info(file_name)

    return {
        "standard_number": standard_info["standard_number"],
        "standard_year": standard_info["standard_year"],
        "document_type": document_type,
        "department": department,
        "file_name": file_name,
    }


def build_chunk_metadata(chunk, document_metadata):
    """
    Combine document-level metadata with chunk-level metadata.
    """

    metadata = dict(document_metadata)

    metadata.update(
        {
            "chunk_id": chunk.get("chunk_id"),
            "section_type": chunk.get("section_type"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "clause_number": chunk.get("clause_number"),
            "clause_title": chunk.get("clause_title"),
            "table_number": chunk.get("table_number"),
            "annex_number": chunk.get("annex_number"),
            "annex_title": chunk.get("annex_title"),
        }
    )

    return metadata


def prepare_chunks_for_embedding(
    chunks,
    file_name,
    department=DEFAULT_DEPARTMENT,
    document_type=DEFAULT_DOCUMENT_TYPE,
):
    """
    Add complete metadata to every chunk.

    Returns:
        List of dictionaries containing:

        {
            "text": "...",
            "metadata": {...}
        }
    """

    document_metadata = build_document_metadata(
        file_name=file_name,
        department=department,
        document_type=document_type,
    )

    prepared_chunks = []

    for index, chunk in enumerate(chunks, start=1):
        metadata = build_chunk_metadata(
            chunk=chunk,
            document_metadata=document_metadata,
        )

        # Give every chunk a stable ID if one is not already present.
        if not metadata["chunk_id"]:
            standard_number = (
                document_metadata["standard_number"] or "UNKNOWN"
            )

            standard_number = standard_number.replace(" ", "")

            metadata["chunk_id"] = (
                f"{standard_number}_"
                f"{document_metadata['standard_year'] or 'UNKNOWN'}_"
                f"chunk_{index:04d}"
            )

        prepared_chunks.append(
            {
                "text": chunk.get("text", ""),
                "metadata": metadata,
            }
        )

    return prepared_chunks
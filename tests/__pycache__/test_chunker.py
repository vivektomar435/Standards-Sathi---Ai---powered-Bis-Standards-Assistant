from ingestion.pdf_loader import load_pdf
from ingestion.cleaner import clean_pages
from ingestion.chunker import chunk_pages


# ============================================================
# TEST PDF
# ============================================================

PDF_PATH = "documents/standards/8682_2026.pdf"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_chunk(chunk):
    """
    Print one complete chunk in a readable format.
    """

    print("\n" + "=" * 80)

    print(
        "Chunk ID:",
        chunk["chunk_id"]
    )

    print(
        "Pages:",
        chunk["page_start"],
        "-",
        chunk["page_end"]
    )

    print(
        "Section Type:",
        chunk["section_type"]
    )

    print(
        "Clause:",
        chunk["clause_number"]
    )

    print(
        "Clause Title:",
        chunk["clause_title"]
    )

    print(
        "Annex:",
        chunk["annex_number"]
    )

    print(
        "Annex Title:",
        chunk["annex_title"]
    )

    print(
        "Characters:",
        len(chunk["text"])
    )

    print("=" * 80)

    print(chunk["text"])


def print_summary(chunks):
    """
    Print structural statistics for the chunked PDF.
    """

    print("\n")
    print("=" * 80)
    print("CHUNKER SUMMARY")
    print("=" * 80)

    print(
        "Total chunks:",
        len(chunks)
    )

    # --------------------------------------------------------
    # Count section types
    # --------------------------------------------------------

    section_counts = {}

    for chunk in chunks:

        section_type = (
            chunk["section_type"]
        )

        section_counts[section_type] = (
            section_counts.get(
                section_type,
                0
            ) + 1
        )

    print("\nSECTION TYPES")
    print("-------------")

    for section_type, count in section_counts.items():

        print(
            f"{section_type}: {count}"
        )

    # --------------------------------------------------------
    # Unique clauses
    # --------------------------------------------------------

    clauses = []

    for chunk in chunks:

        clause = chunk["clause_number"]

        if clause and clause not in clauses:
            clauses.append(clause)

    print("\nDETECTED CLAUSES")
    print("----------------")

    if clauses:

        print(
            ", ".join(clauses)
        )

    else:

        print("None")

    # --------------------------------------------------------
    # Detected annexes
    # --------------------------------------------------------

    annexes = []

    for chunk in chunks:

        annex = chunk["annex_number"]

        if annex and annex not in annexes:
            annexes.append(annex)

    print("\nDETECTED ANNEXES")
    print("----------------")

    if annexes:

        print(
            ", ".join(annexes)
        )

    else:

        print("None")


def check_suspicious_numeric_clauses(chunks):
    """
    Look for obvious cases where a numeric/table value
    accidentally became a clause.

    This is a validation helper, not a replacement for
    manual inspection.
    """

    suspicious = []

    for chunk in chunks:

        clause = chunk["clause_number"]
        title = chunk["clause_title"]

        if not clause or not title:
            continue

        title_lower = title.lower()

        suspicious_words = [
            "%",
            "°c",
            "µm",
            "mg/l",
            "litres",
            "litre",
            "ml",
            "kg",
            "mm",
            "cm"
        ]

        if any(
            word in title_lower
            for word in suspicious_words
        ):

            suspicious.append(chunk)

    print("\n")
    print("=" * 80)
    print("SUSPICIOUS CLAUSE CHECK")
    print("=" * 80)

    if not suspicious:

        print(
            "No obvious measurement/unit false clauses detected."
        )

        return

    print(
        f"Found {len(suspicious)} suspicious chunk(s):"
    )

    for chunk in suspicious:

        print("\n" + "-" * 80)

        print(
            "Chunk:",
            chunk["chunk_id"]
        )

        print(
            "Clause:",
            chunk["clause_number"]
        )

        print(
            "Title:",
            chunk["clause_title"]
        )

        print(
            "Pages:",
            chunk["page_start"],
            "-",
            chunk["page_end"]
        )


def check_annex_boundaries(chunks):
    """
    Verify that annexes are recognized separately from
    preceding numerical clauses.
    """

    print("\n")
    print("=" * 80)
    print("ANNEX BOUNDARY CHECK")
    print("=" * 80)

    annex_chunks = [
        chunk
        for chunk in chunks
        if chunk["section_type"]
        in ("annex", "annex_clause")
    ]

    if not annex_chunks:

        print(
            "WARNING: No annex chunks detected."
        )

        return

    print(
        f"Detected {len(annex_chunks)} annex-related chunks."
    )

    for chunk in annex_chunks:

        print("\n" + "-" * 80)

        print(
            "Chunk:",
            chunk["chunk_id"]
        )

        print(
            "Section Type:",
            chunk["section_type"]
        )

        print(
            "Annex:",
            chunk["annex_number"]
        )

        print(
            "Clause:",
            chunk["clause_number"]
        )

        print(
            "Pages:",
            chunk["page_start"],
            "-",
            chunk["page_end"]
        )

        # Print only first 300 characters so the validation
        # output stays readable.
        preview = chunk["text"][:300]

        print(
            "Preview:",
            preview.replace("\n", " | ")
        )


def check_page_ranges(chunks):
    """
    Check that every chunk has a valid page range.
    """

    print("\n")
    print("=" * 80)
    print("PAGE RANGE CHECK")
    print("=" * 80)

    problems = []

    for chunk in chunks:

        start = chunk["page_start"]
        end = chunk["page_end"]

        if start > end:

            problems.append(chunk)

    if not problems:

        print(
            "All chunk page ranges are valid."
        )

    else:

        print(
            f"Found {len(problems)} invalid page ranges."
        )

        for chunk in problems:

            print(
                chunk["chunk_id"],
                chunk["page_start"],
                chunk["page_end"]
            )


# ============================================================
# MAIN TEST
# ============================================================

def main():

    # ========================================================
    # 1. LOAD COMPLETE PDF
    # ========================================================

    print("\n")
    print("=" * 80)
    print("STEP 1 — PDF LOADING")
    print("=" * 80)

    pdf_data = load_pdf(
        PDF_PATH
    )

    print(
        "File:",
        pdf_data["file_name"]
    )

    print(
        "PDF pages:",
        pdf_data["page_count"]
    )

    # ========================================================
    # 2. CLEAN COMPLETE PDF
    # ========================================================

    print("\n")
    print("=" * 80)
    print("STEP 2 — CLEANING")
    print("=" * 80)

    cleaned_pages = clean_pages(
        pdf_data["pages"]
    )

    print(
        "Cleaned pages:",
        len(cleaned_pages)
    )

    # ========================================================
    # 3. CHUNK COMPLETE PDF
    # ========================================================

    print("\n")
    print("=" * 80)
    print("STEP 3 — CHUNKING")
    print("=" * 80)

    chunks = chunk_pages(
        cleaned_pages
    )

    print(
        "Total chunks:",
        len(chunks)
    )

    # ========================================================
    # 4. SUMMARY
    # ========================================================

    print_summary(
        chunks
    )

    # ========================================================
    # 5. VALIDATION CHECKS
    # ========================================================

    check_suspicious_numeric_clauses(
        chunks
    )

    check_annex_boundaries(
        chunks
    )

    check_page_ranges(
        chunks
    )

    # ========================================================
    # 6. PRINT ALL CHUNKS
    # ========================================================

    print("\n")
    print("=" * 80)
    print("ALL CHUNKS")
    print("=" * 80)

    for chunk in chunks:

        print_chunk(
            chunk
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
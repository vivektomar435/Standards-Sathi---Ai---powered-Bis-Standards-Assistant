import re


# -------------------------------------------------------------------
# Known BIS boilerplate/footer patterns
# -------------------------------------------------------------------

BIS_FOOTER_PATTERNS = [
    r"Free Standard provided by BIS via BSB Edge Private Limited.*",
    r"prayagraj\(manish45360@gmail\.com\).*",
]


# -------------------------------------------------------------------
# Repeated BIS page header patterns
# -------------------------------------------------------------------

BIS_HEADER_PATTERNS = [
    # Example: IS 8682 : 2026
    r"^IS\s+\d+(?:\s*:\s*\d{4})?$",

    # Handles possible extraction spacing:
    # IS 8682:2026
    # IS 8682 :2026
    r"^IS\s+\d+\s*:\s*\d{4}$",
]


def remove_bis_boilerplate(text):
    """
    Remove known BIS footer/boilerplate text.

    Only known boilerplate patterns are removed.
    Technical content is otherwise preserved.
    """

    for pattern in BIS_FOOTER_PATTERNS:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE
        )

    return text


def is_bis_header(line):
    """
    Detect repeated BIS standard header lines such as:

        IS 8682 : 2026
        IS 8682:2026
        IS 8682 :2026

    These headers are normally repeated on every page and are
    not useful as semantic document content.
    """

    line = line.strip()

    if not line:
        return False

    for pattern in BIS_HEADER_PATTERNS:
        if re.fullmatch(
            pattern,
            line,
            flags=re.IGNORECASE
        ):
            return True

    return False


def is_standalone_page_number(line):
    """
    Detect a line containing only a page number.

    This function is intentionally conservative.

    It does NOT remove every numeric line because BIS standards
    contain many legitimate numeric values, measurements, limits,
    clause numbers, table values, etc.

    Page numbers are removed only when they appear as standalone
    lines.
    """

    line = line.strip()

    if not line:
        return False

    return bool(
        re.fullmatch(r"\d{1,3}", line)
    )


def clean_page_lines(lines):
    """
    Clean individual page lines.

    Removes:
    - repeated BIS standard headers
    - standalone page numbers at the beginning/end of a page
    - known BIS boilerplate

    Preserves technical numeric content in the body.
    """

    if not lines:
        return []

    cleaned = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Remove repeated BIS standard header.
        if is_bis_header(line):
            continue

        cleaned.append(line)

    # ---------------------------------------------------------------
    # Remove standalone page numbers only at page boundaries.
    #
    # This is important because a number such as "100" in the middle
    # of a technical paragraph/table may be meaningful.
    # ---------------------------------------------------------------

    while cleaned and is_standalone_page_number(cleaned[0]):
        cleaned.pop(0)

    while cleaned and is_standalone_page_number(cleaned[-1]):
        cleaned.pop()

    return cleaned


def clean_text(text):
    """
    Clean extracted PDF text while preserving technical content.

    Processing:
    1. Normalize line endings.
    2. Remove known BIS boilerplate.
    3. Remove repeated BIS page headers.
    4. Remove standalone page numbers at page boundaries.
    5. Normalize whitespace.
    6. Preserve meaningful line structure.
    """

    if not text:
        return ""

    # ---------------------------------------------------------------
    # Normalize line endings
    # ---------------------------------------------------------------

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # ---------------------------------------------------------------
    # Remove known BIS boilerplate
    # ---------------------------------------------------------------

    text = remove_bis_boilerplate(text)

    # ---------------------------------------------------------------
    # Split into lines
    # ---------------------------------------------------------------

    lines = text.split("\n")

    # ---------------------------------------------------------------
    # Remove repeated headers and boundary page numbers
    # ---------------------------------------------------------------

    lines = clean_page_lines(lines)

    # ---------------------------------------------------------------
    # Normalize horizontal whitespace.
    #
    # Do NOT aggressively join lines because line boundaries can
    # contain useful information for:
    # - clauses
    # - tables
    # - headings
    # - annexes
    # - lists
    # ---------------------------------------------------------------

    cleaned_lines = []

    for line in lines:
        line = re.sub(r"[ \t]+", " ", line)
        line = line.strip()

        if line:
            cleaned_lines.append(line)

    # ---------------------------------------------------------------
    # Normalize excessive blank lines
    # ---------------------------------------------------------------

    text = "\n".join(cleaned_lines)

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()


def clean_pages(pages):
    """
    Clean all extracted PDF pages.

    Input:
        [
            {
                "page_number": 1,
                "text": "..."
            },
            ...
        ]

    Output:
        [
            {
                "page_number": 1,
                "text": "..."
            },
            ...
        ]

    Page numbers are preserved so the chunker can maintain
    page-level metadata.
    """

    cleaned_pages = []

    for page in pages:
        page_number = page["page_number"]
        raw_text = page.get("text", "")

        cleaned_text = clean_text(raw_text)

        cleaned_pages.append({
            "page_number": page_number,
            "text": cleaned_text
        })

    return cleaned_pages
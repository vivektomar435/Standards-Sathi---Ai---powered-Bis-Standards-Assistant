import re


# ============================================================
# REGEX PATTERNS
# ============================================================

CLAUSE_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)*)\s+(.+)$"
)

ANNEX_PATTERN = re.compile(
    r"^ANNEX\s+([A-Z])(?:\s*[-—:]\s*(.*))?$",
    re.IGNORECASE
)

ANNEX_CLAUSE_PATTERN = re.compile(
    r"^([A-Z])-(\d+(?:\.\d+)*)\s+(.+)$",
    re.IGNORECASE
)

TABLE_PATTERN = re.compile(
    r"^Table\s+(\d+)"
    r"(?:\s*\((Continued|Concluded)\))?"
    r"(?:\s*(?:[-—:]\s*)?(.*?))?$",
    re.IGNORECASE
)

# Handles lines such as:
# Table 4.
#
# These are heading fragments and should NOT become their own chunks.
TABLE_DOT_PATTERN = re.compile(
    r"^Table\s+(\d+)\s*\.\s*$",
    re.IGNORECASE
)


# ============================================================
# REQUIRED METADATA SCHEMA
# ============================================================

REQUIRED_METADATA_KEYS = [
    "chunk_id",
    "type",
    "section_type",
    "text",
    "page_number",
    "start_page",
    "end_page",
    "page_start",
    "page_end",
    "clause_number",
    "title",
    "clause_title",
    "table_number",
    "annex",
    "annex_number",
    "annex_title",
]


# ============================================================
# BASIC CLEANING HELPERS
# ============================================================

def is_standalone_page_number(line):
    """
    Detect a line containing only a page number.
    """
    line = line.strip()

    if not line:
        return False

    return bool(re.fullmatch(r"\d{1,3}", line))


def is_bis_header(line):
    """
    Detect common BIS standard header lines.
    """
    line = line.strip()

    if not line:
        return False

    patterns = [
        r"^IS\s+\d+(?:\s*:\s*\d{4})?$",
        r"^IS\s+\d+\s*:\s*\d{4}$",
    ]

    for pattern in patterns:
        if re.fullmatch(pattern, line, flags=re.IGNORECASE):
            return True

    return False


def is_bis_boilerplate(line):
    """
    Detect known BIS PDF distribution/footer boilerplate.
    """
    if not line:
        return False

    patterns = [
        r"Free Standard provided by BIS via BSB Edge Private Limited.*",
        r"prayagraj\(manish45360@gmail\.com\).*",
    ]

    for pattern in patterns:
        if re.search(pattern, line, flags=re.IGNORECASE):
            return True

    return False


def clean_page_text(text):
    """
    Apply light structural cleaning to page text.

    Important:
    We do NOT try to correct OCR/extraction errors here.
    Standards text should remain as faithful as possible
    to the extracted PDF content.
    """
    if not text:
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = text.split("\n")

    cleaned = []

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if is_bis_boilerplate(line):
            continue

        if is_bis_header(line):
            continue

        cleaned.append(line)

    # Remove page number from beginning/end if present.
    while cleaned and is_standalone_page_number(cleaned[0]):
        cleaned.pop(0)

    while cleaned and is_standalone_page_number(cleaned[-1]):
        cleaned.pop()

    normalized = []

    for line in cleaned:
        line = re.sub(r"[ \t]+", " ", line)
        line = line.strip()

        if line:
            normalized.append(line)

    return "\n".join(normalized).strip()


# ============================================================
# CLAUSE DETECTION
# ============================================================

def detect_clause(line):
    """
    Detect a normal numbered clause.

    Examples accepted:
        1 Introduction
        3.1 General
        5.3.1.1 Treatment process
        7.3.6 Final disposal

    Examples rejected:
        20 °C(mg/l)
        90 %; BOD reduction may reach
        0 123 456
        2 170 1 590 230

    The important rule is that the clause title must BEGIN
    with an alphabetic character.

    This prevents table values such as:

        20 °C(mg/l)

    and:

        90 %; BOD reduction may reach

    from being interpreted as clauses.
    """
    line = line.strip()

    if not line:
        return None

    match = CLAUSE_PATTERN.match(line)

    if not match:
        return None

    clause_number = match.group(1)
    title = match.group(2).strip()

    # --------------------------------------------------------
    # Reject invalid clause numbers
    # --------------------------------------------------------

    if clause_number.startswith("0"):
        return None

    if not title:
        return None

    # --------------------------------------------------------
    # IMPORTANT FIX
    #
    # A real clause title in this document begins with a
    # letter. Table values often begin with:
    #
    #   °
    #   %
    #   (
    #   numbers
    #   units
    #
    # Therefore the FIRST character of the title must itself
    # be alphabetic.
    # --------------------------------------------------------

    if not re.match(r"^[A-Za-z]", title):
        return None

    # --------------------------------------------------------
    # Require at least one alphabetic character.
    # --------------------------------------------------------

    if not re.search(r"[A-Za-z]", title):
        return None

    # --------------------------------------------------------
    # Reject titles that are effectively numeric/symbol data.
    # --------------------------------------------------------

    if re.fullmatch(r"[\d\s.,%°()/+\-]+", title):
        return None

    # --------------------------------------------------------
    # Preserve the previous conservative check:
    # the first alphabetic character should be uppercase.
    #
    # This helps prevent fragments from being detected as
    # structural clauses.
    # --------------------------------------------------------

    first_alpha = re.search(r"[A-Za-z]", title)

    if not first_alpha:
        return None

    if not first_alpha.group(0).isupper():
        return None

    return {
        "number": clause_number,
        "title": title,
    }


# ============================================================
# ANNEX DETECTION
# ============================================================

def detect_annex(line):
    """
    Detect an annex heading.

    Examples:
        ANNEX A
        ANNEX A - LIST OF REFERRED STANDARDS
        ANNEX B — INDIAN STANDARDS FOR DAIRY PRODUCTS
    """
    line = line.strip()

    if not line:
        return None

    match = ANNEX_PATTERN.match(line)

    if not match:
        return None

    annex_number = match.group(1).upper()
    title = (match.group(2) or "").strip()

    return {
        "number": annex_number,
        "title": title,
    }


# ============================================================
# ANNEX CLAUSE DETECTION
# ============================================================

def detect_annex_clause(line):
    """
    Detect annex clauses.

    Example:
        B-1 Indian Standards for ...
    """
    line = line.strip()

    if not line:
        return None

    match = ANNEX_CLAUSE_PATTERN.match(line)

    if not match:
        return None

    annex_number = match.group(1).upper()
    clause_number = match.group(2)
    title = match.group(3).strip()

    return {
        "annex": annex_number,
        "number": clause_number,
        "title": title,
    }


# ============================================================
# TABLE DETECTION
# ============================================================

def detect_table(line):
    """
    Detect table headings.

    Examples:
        Table 1
        Table 2 Characteristics...
        Table 5 (Continued)
        Table 5 (Concluded)

    A standalone:
        Table 4.

    is deliberately ignored because it is usually an
    extraction fragment and the real Table 4 heading/content
    follows separately.
    """
    line = line.strip()

    if not line:
        return None

    # --------------------------------------------------------
    # Ignore standalone "Table N."
    # --------------------------------------------------------

    dot_only_match = TABLE_DOT_PATTERN.fullmatch(line)

    if dot_only_match:
        return None

    match = TABLE_PATTERN.match(line)

    if not match:
        return None

    table_number = match.group(1)
    continuation = match.group(2)
    title = (match.group(3) or "").strip()

    return {
        "number": table_number,
        "title": title,
        "continuation": continuation,
    }


# ============================================================
# ANNEX TITLE HELPERS
# ============================================================

def is_parenthetical_line(line):
    """
    Detect a standalone parenthetical line.
    """
    line = line.strip()

    if not line:
        return True

    return line.startswith("(") and line.endswith(")")


def find_annex_title(lines, start_index):
    """
    Look ahead a few lines after an ANNEX heading to find
    its actual title.

    Example:

        ANNEX A

        LIST OF REFERRED STANDARDS

    returns:

        LIST OF REFERRED STANDARDS
    """
    max_lookahead = min(len(lines), start_index + 5)

    for index in range(start_index + 1, max_lookahead):
        candidate = lines[index].strip()

        if not candidate:
            continue

        if is_parenthetical_line(candidate):
            continue

        if detect_annex(candidate):
            break

        if detect_annex_clause(candidate):
            continue

        if detect_clause(candidate):
            break

        if detect_table(candidate):
            break

        return candidate

    return ""


# ============================================================
# CHUNK CREATION
# ============================================================

def make_chunk(
    chunk_id,
    section_type,
    text,
    page_start,
    page_end,
    clause_number=None,
    title=None,
    table_number=None,
    annex_number=None,
    annex_title=None,
):
    """
    Create a chunk with a stable metadata schema.

    Multiple aliases are intentionally preserved because
    different parts of the project/tests may refer to the
    same information using different field names.
    """

    if section_type == "front_matter":
        chunk_type = "front_matter"

    elif section_type == "clause":
        chunk_type = "clause"

    elif section_type == "table":
        chunk_type = "table"

    elif section_type == "annex":
        chunk_type = "annex"

    elif section_type == "annex_clause":
        chunk_type = "annex_clause"

    else:
        chunk_type = section_type

    return {
        "chunk_id": chunk_id,

        "type": chunk_type,
        "section_type": section_type,

        "text": text.strip(),

        "page_number": page_start,

        "start_page": page_start,
        "end_page": page_end,

        "page_start": page_start,
        "page_end": page_end,

        "clause_number": clause_number,

        "title": title,
        "clause_title": title,

        "table_number": table_number,

        "annex": annex_number,
        "annex_number": annex_number,
        "annex_title": annex_title,
    }


# ============================================================
# METADATA NORMALIZATION
# ============================================================

def normalize_chunk_metadata(chunks):
    """
    Guarantee that every chunk has the same metadata schema.

    Also reassign sequential chunk IDs after merging.
    """

    normalized = []

    for index, chunk in enumerate(chunks, start=1):

        for key in REQUIRED_METADATA_KEYS:
            if key not in chunk:
                chunk[key] = None

        # Keep aliases synchronized.
        if chunk.get("page_start") is None:
            chunk["page_start"] = chunk.get("start_page")

        if chunk.get("page_end") is None:
            chunk["page_end"] = chunk.get("end_page")

        if chunk.get("start_page") is None:
            chunk["start_page"] = chunk.get("page_start")

        if chunk.get("end_page") is None:
            chunk["end_page"] = chunk.get("page_end")

        if chunk.get("clause_title") is None:
            chunk["clause_title"] = chunk.get("title")

        if chunk.get("title") is None:
            chunk["title"] = chunk.get("clause_title")

        if chunk.get("annex_number") is None:
            chunk["annex_number"] = chunk.get("annex")

        if chunk.get("annex") is None:
            chunk["annex"] = chunk.get("annex_number")

        if chunk.get("chunk_id") is None:
            chunk["chunk_id"] = index

        chunk["chunk_id"] = index

        normalized.append(chunk)

    return normalized


# ============================================================
# APPEND CURRENT CHUNK
# ============================================================

def append_current_chunk(
    chunks,
    chunk_id,
    current_type,
    current_lines,
    current_start_page,
    current_end_page,
    current_clause_number=None,
    current_title=None,
    current_table_number=None,
    current_annex_number=None,
    current_annex_title=None,
):
    """
    Convert the current in-memory section into a chunk.
    """

    if not current_lines:
        return chunk_id

    text = "\n".join(current_lines).strip()

    if not text:
        return chunk_id

    chunk = make_chunk(
        chunk_id=chunk_id,
        section_type=current_type,
        text=text,
        page_start=current_start_page,
        page_end=current_end_page,
        clause_number=current_clause_number,
        title=current_title,
        table_number=current_table_number,
        annex_number=current_annex_number,
        annex_title=current_annex_title,
    )

    chunks.append(chunk)

    return chunk_id + 1


# ============================================================
# MERGE ADJACENT CHUNKS
# ============================================================

def merge_adjacent_same_type_chunks(chunks):
    """
    Merge chunks that clearly belong to the same structural
    section.

    Important cases:

    - consecutive front matter
    - Table 5 + Table 5 (Continued)
    - Table 5 (Continued) + Table 5 (Concluded)
    - consecutive chunks belonging to the same annex
    """

    if not chunks:
        return []

    merged = []

    for chunk in chunks:

        if not merged:
            merged.append(chunk)
            continue

        previous = merged[-1]

        # ----------------------------------------------------
        # Front matter
        # ----------------------------------------------------

        if (
            previous["section_type"] == "front_matter"
            and chunk["section_type"] == "front_matter"
        ):
            previous["text"] += "\n" + chunk["text"]
            previous["end_page"] = chunk["end_page"]
            previous["page_end"] = chunk["page_end"]
            continue

        # ----------------------------------------------------
        # Same table
        # ----------------------------------------------------

        if (
            previous["section_type"] == "table"
            and chunk["section_type"] == "table"
            and previous.get("table_number")
            == chunk.get("table_number")
        ):
            previous["text"] += "\n" + chunk["text"]
            previous["end_page"] = chunk["end_page"]
            previous["page_end"] = chunk["page_end"]
            continue

        # ----------------------------------------------------
        # Same annex
        #
        # Do NOT merge annex clauses into the annex heading.
        # ----------------------------------------------------

        if (
            previous["section_type"] == "annex"
            and chunk["section_type"] == "annex"
            and previous.get("annex_number")
            == chunk.get("annex_number")
        ):
            previous["text"] += "\n" + chunk["text"]
            previous["end_page"] = chunk["end_page"]
            previous["page_end"] = chunk["page_end"]
            continue

        merged.append(chunk)

    return merged


# ============================================================
# MAIN CHUNKING FUNCTION
# ============================================================

def chunk_pages(pages):
    """
    Convert cleaned PDF pages into structural chunks.

    Supported structures:

        front_matter
        clause
        table
        annex
        annex_clause
    """

    chunks = []

    chunk_id = 1

    current_type = None
    current_lines = []

    current_start_page = None
    current_end_page = None

    current_clause_number = None
    current_title = None

    current_table_number = None

    current_annex_number = None
    current_annex_title = None

    def flush_current():
        nonlocal chunk_id
        nonlocal current_type
        nonlocal current_lines
        nonlocal current_start_page
        nonlocal current_end_page
        nonlocal current_clause_number
        nonlocal current_title
        nonlocal current_table_number
        nonlocal current_annex_number
        nonlocal current_annex_title

        chunk_id = append_current_chunk(
            chunks=chunks,
            chunk_id=chunk_id,
            current_type=current_type,
            current_lines=current_lines,
            current_start_page=current_start_page,
            current_end_page=current_end_page,
            current_clause_number=current_clause_number,
            current_title=current_title,
            current_table_number=current_table_number,
            current_annex_number=current_annex_number,
            current_annex_title=current_annex_title,
        )

        current_type = None
        current_lines = []

        current_start_page = None
        current_end_page = None

        current_clause_number = None
        current_title = None

        current_table_number = None

        current_annex_number = None
        current_annex_title = None

    for page in pages:

        page_number = page["page_number"]
        raw_text = page.get("text", "")

        page_text = clean_page_text(raw_text)

        if not page_text:
            continue

        lines = page_text.split("\n")

        for line_index, line in enumerate(lines):

            line = line.strip()

            if not line:
                continue

            # ------------------------------------------------
            # Detect structural elements
            # ------------------------------------------------

            annex = detect_annex(line)

            annex_clause = detect_annex_clause(line)

            table = detect_table(line)

            clause = detect_clause(line)

            # ------------------------------------------------
            # ANNEX
            # ------------------------------------------------

            if annex:

                flush_current()

                current_type = "annex"
                current_start_page = page_number
                current_end_page = page_number

                current_annex_number = annex["number"]

                current_annex_title = annex["title"]

                # If the heading itself does not contain the
                # title, look ahead for it.
                if not current_annex_title:
                    current_annex_title = find_annex_title(
                        lines,
                        line_index
                    )

                current_lines = [line]

                continue

            # ------------------------------------------------
            # ANNEX CLAUSE
            # ------------------------------------------------

            if annex_clause:

                flush_current()

                current_type = "annex_clause"

                current_start_page = page_number
                current_end_page = page_number

                current_annex_number = annex_clause["annex"]

                current_clause_number = annex_clause["number"]

                current_title = annex_clause["title"]

                current_annex_title = None

                current_lines = [line]

                continue

            # ------------------------------------------------
            # TABLE
            # ------------------------------------------------

            if table:

                flush_current()

                current_type = "table"

                current_start_page = page_number
                current_end_page = page_number

                current_table_number = table["number"]

                current_title = table["title"]

                current_annex_number = None
                current_annex_title = None

                current_lines = [line]

                continue

            # ------------------------------------------------
            # CLAUSE
            # ------------------------------------------------

            if clause:

                flush_current()

                current_type = "clause"

                current_start_page = page_number
                current_end_page = page_number

                current_clause_number = clause["number"]

                current_title = clause["title"]

                current_table_number = None

                current_annex_number = None
                current_annex_title = None

                current_lines = [line]

                continue

            # ------------------------------------------------
            # NORMAL CONTENT
            # ------------------------------------------------

            if current_type is None:

                current_type = "front_matter"

                current_start_page = page_number
                current_end_page = page_number

                current_clause_number = None
                current_title = None

                current_table_number = None

                current_annex_number = None
                current_annex_title = None

                current_lines = [line]

            else:

                current_lines.append(line)

            current_end_page = page_number

    # --------------------------------------------------------
    # Flush final chunk
    # --------------------------------------------------------

    flush_current()

    # --------------------------------------------------------
    # Merge structurally related chunks
    # --------------------------------------------------------

    chunks = merge_adjacent_same_type_chunks(chunks)

    # --------------------------------------------------------
    # Guarantee metadata consistency
    # --------------------------------------------------------

    chunks = normalize_chunk_metadata(chunks)

    return chunks


# ============================================================
# DOCUMENT-LEVEL ALIASES
# ============================================================

def chunk_document(pages):
    """
    Alias for chunk_pages().
    """
    return chunk_pages(pages)


def chunk_pages_to_chunks(pages):
    """
    Backward-compatible alias.
    """
    return chunk_pages(pages)
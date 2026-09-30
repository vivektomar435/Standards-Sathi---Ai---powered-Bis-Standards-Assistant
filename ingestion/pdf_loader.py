from pathlib import Path
from pypdf import PdfReader


def load_pdf(pdf_path):
    """
    Load a PDF and extract text page by page.

    Args:
        pdf_path (str | Path): Path to the PDF file.

    Returns:
        dict: PDF information and extracted pages.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, got: {pdf_path.suffix}"
        )

    reader = PdfReader(str(pdf_path))

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):
        text = page.extract_text() or ""

        pages.append({
            "page_number": page_number,
            "text": text
        })

    return {
        "file_name": pdf_path.name,
        "file_path": str(pdf_path),
        "page_count": len(reader.pages),
        "pages": pages
    }
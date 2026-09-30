from ingestion.pdf_loader import load_pdf
from ingestion.cleaner import clean_pages


PDF_PATH = "documents/standards/8682_2026.pdf"


def main():

    # Load the complete PDF
    pdf_data = load_pdf(PDF_PATH)

    # Clean ALL pages
    cleaned_pages = clean_pages(pdf_data["pages"])

    print("\nCLEANER TEST")
    print("------------")

    print("File:", pdf_data["file_name"])
    print("Original pages:", pdf_data["page_count"])
    print("Cleaned pages:", len(cleaned_pages))

    # Display ALL cleaned pages
    for page in cleaned_pages:

        print("\n" + "=" * 60)
        print(f"PAGE {page['page_number']}")
        print("=" * 60)

        print(page["text"])


if __name__ == "__main__":
    main()
from ingestion.pdf_loader import load_pdf


PDF_PATH = "documents/standards/8682_2026.pdf"


def main():
    data = load_pdf(PDF_PATH)

    print("\nPDF INFORMATION")
    print("----------------")
    print("File:", data["file_name"])
    print("Pages:", data["page_count"])

    print("\nALL PAGES")
    print("---------")

    for page in data["pages"]:

        print("\n" + "=" * 60)
        print(f"PAGE {page['page_number']}")
        print("=" * 60)

        print(page["text"])


if __name__ == "__main__":
    main()
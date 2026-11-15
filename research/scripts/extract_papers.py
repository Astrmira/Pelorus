from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "papers" / "pdf"
TEXT_DIR = ROOT / "papers" / "notes"


def main() -> None:
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        reader = PdfReader(pdf_path)
        text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
        output_path = TEXT_DIR / f"{pdf_path.stem}.txt"
        output_path.write_text(text, encoding="utf-8")
        print(f"已提取：{pdf_path.name} -> {output_path.name}")


if __name__ == "__main__":
    main()

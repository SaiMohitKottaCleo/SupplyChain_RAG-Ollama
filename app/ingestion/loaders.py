from pathlib import Path
from typing import List, Dict, Any
import requests
from bs4 import BeautifulSoup
import pypdf
from docx import Document
import pandas as pd
from loguru import logger


class DocumentLoader:
    """
    Loads documents from PDFs, DOCX, TXT files, and URLs into a uniform format.

    Every load method returns a list of dicts:
        [{"text": str, "metadata": {"source": str, "source_type": str, ...}}, ...]

    One dict per logical unit (page for PDFs, section for DOCX, whole file for TXT).
    Multiple dicts per document allow per-page/per-section metadata tracking —
    so citations can say "page 4 of X12_850_Guide.pdf" not just "X12_850_Guide.pdf".
    """

    @staticmethod
    def load_pdf(path: Path) -> List[Dict[str, Any]]:
        """
        Extract text from a PDF, one dict per page.

        pypdf reads the page content streams and extracts text from
        Tj/TJ operators. Works on born-digital PDFs. Scanned PDFs
        (images) will return empty text — they need OCR, which we skip.
        """
        docs = []
        try:
            reader = pypdf.PdfReader(str(path))
            total_pages = len(reader.pages)
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                # Skip pages with trivially short content (cover pages, blank pages)
                if text and len(text.strip()) > 50:
                    docs.append({
                        "text": text.strip(),
                        "metadata": {
                            "source":       path.name,
                            "source_type":  "pdf",
                            "page":         i + 1,
                            "total_pages":  total_pages,
                        }
                    })
            logger.info(f"PDF loaded: {path.name} — {len(docs)}/{total_pages} pages with content")
        except Exception as e:
            logger.error(f"Failed to load PDF {path}: {e}")
        return docs

    @staticmethod
    def load_docx(path: Path) -> List[Dict[str, Any]]:
        """
        Extract text from a DOCX file, grouped by heading sections.

        A DOCX is a ZIP of XML files. python-docx parses word/document.xml
        and exposes paragraphs with their style names. We accumulate paragraphs
        until we hit a new Heading — that boundary becomes a section cut.
        This preserves semantic structure better than splitting at N characters.
        """
        docs = []
        try:
            doc = Document(str(path))
            current_section: List[str] = []
            current_heading = "Introduction"

            for para in doc.paragraphs:
                if para.style.name.startswith("Heading"):
                    # Flush the current section before starting the new one
                    if current_section:
                        docs.append({
                            "text": "\n".join(current_section),
                            "metadata": {
                                "source":       path.name,
                                "source_type":  "docx",
                                "section":      current_heading,
                            }
                        })
                        current_section = []
                    current_heading = para.text
                elif para.text.strip():
                    current_section.append(para.text.strip())

            # Flush the final section
            if current_section:
                docs.append({
                    "text": "\n".join(current_section),
                    "metadata": {
                        "source":       path.name,
                        "source_type":  "docx",
                        "section":      current_heading,
                    }
                })
            logger.info(f"DOCX loaded: {path.name} — {len(docs)} sections")
        except Exception as e:
            logger.error(f"Failed to load DOCX {path}: {e}")
        return docs

    @staticmethod
    def load_txt(path: Path) -> List[Dict[str, Any]]:
        """
        Load a plain text or Markdown file as a single document unit.
        The chunker downstream will split it into retrieval-sized pieces.
        """
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            if not text.strip():
                logger.warning(f"Empty file: {path.name}")
                return []
            logger.info(f"TXT loaded: {path.name} — {len(text):,} characters")
            return [{
                "text": text,
                "metadata": {
                    "source":       path.name,
                    "source_type":  "txt",
                }
            }]
        except Exception as e:
            logger.error(f"Failed to load TXT {path}: {e}")
            return []

    @staticmethod
    def load_url(url: str) -> List[Dict[str, Any]]:
        """
        Scrape a public URL and extract the main text content.

        Steps:
        1. HTTP GET with a browser-like User-Agent (some sites block bots)
        2. Parse HTML with BeautifulSoup
        3. Remove non-content elements: scripts, styles, nav, footer, header
        4. Extract remaining text, collapse whitespace
        """
        try:
            response = requests.get(url, timeout=15, headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/120.0.0.0 Safari/537.36"
            })
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Surgically remove non-content DOM nodes
            for tag in soup(["script", "style", "nav", "footer",
                              "header", "aside", "form", "iframe"]):
                tag.decompose()

            # Get text, join with newlines, strip blank lines
            raw_text = soup.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            clean_text = "\n".join(lines)

            if len(clean_text) < 100:
                logger.warning(f"Very little content scraped from: {url}")
                return []

            logger.info(f"URL loaded: {url} — {len(clean_text):,} characters")
            return [{
                "text": clean_text,
                "metadata": {
                    "source":       url,
                    "source_type":  "url",
                }
            }]
        except Exception as e:
            logger.error(f"Failed to load URL {url}: {e}")
            return []

    @staticmethod
    def load_excel(path: Path) -> List[Dict[str, Any]]:
        """
        Load an Excel (.xlsx / .xls) or CSV file.

        Strategy — one dict per sheet (Excel) or one dict total (CSV):
          - Each row is rendered as "Column: value  |  Column: value  | ..."
          - All rows for a sheet are joined into a single text block
          - Empty rows and fully-null rows are skipped

        Why row-by-row text instead of raw CSV?
        The chunker splits on semantic boundaries like paragraph breaks.
        "Column: value | Column: value" lines give the sentence-transformer
        enough natural-language signal to embed meaningfully — a raw CSV
        header line like "part_id,qty,uom" does not.
        """
        docs = []
        try:
            if path.suffix.lower() == ".csv":
                sheets = {"sheet": pd.read_csv(str(path), dtype=str)}
            else:
                sheets = pd.read_excel(str(path), sheet_name=None, dtype=str)

            for sheet_name, df in sheets.items():
                df = df.dropna(how="all").fillna("")
                if df.empty:
                    continue

                lines = []
                for _, row in df.iterrows():
                    parts = [
                        f"{col}: {val}"
                        for col, val in row.items()
                        if str(val).strip()
                    ]
                    if parts:
                        lines.append("  |  ".join(parts))

                if not lines:
                    continue

                text = f"Sheet: {sheet_name}\n\n" + "\n".join(lines)
                docs.append({
                    "text": text,
                    "metadata": {
                        "source":       path.name,
                        "source_type":  path.suffix.lstrip("."),
                        "sheet":        sheet_name,
                        "rows":         len(lines),
                    }
                })

            logger.info(
                f"Excel/CSV loaded: {path.name} — "
                f"{len(docs)} sheet(s), "
                f"{sum(d['metadata']['rows'] for d in docs)} rows"
            )
        except Exception as e:
            logger.error(f"Failed to load Excel/CSV {path}: {e}")
        return docs

    @staticmethod
    def load_text(text: str, source_name: str = "inline_text") -> List[Dict[str, Any]]:
        """
        Wrap a raw string as a document dict — used when the caller already
        has text in memory (e.g., Streamlit paste box, API input).
        """
        if not text.strip():
            return []
        logger.info(f"Text loaded: '{source_name}' — {len(text):,} characters")
        return [{
            "text": text,
            "metadata": {
                "source":      source_name,
                "source_type": "text",
            }
        }]

    @classmethod
    def load(cls, source) -> List[Dict[str, Any]]:
        """
        Auto-detect source type and dispatch to the right loader.
        Accepts: URL string, Path object, file path string, or raw text string.

        Raw text detection: a string containing newlines cannot be a file path.
        """
        if isinstance(source, str):
            if source.startswith("http"):
                return cls.load_url(source)
            # Multi-line strings are raw text, not paths
            if "\n" in source:
                return cls.load_text(source)

        path = Path(source)
        if not path.exists():
            logger.warning(f"File not found: {path}")
            return []

        ext = path.suffix.lower()
        if ext == ".pdf":
            return cls.load_pdf(path)
        elif ext == ".docx":
            return cls.load_docx(path)
        elif ext in [".txt", ".md"]:
            return cls.load_txt(path)
        elif ext in [".xlsx", ".xls", ".csv"]:
            return cls.load_excel(path)
        else:
            logger.warning(f"Unsupported file type: {ext} — skipping {path.name}")
            return []

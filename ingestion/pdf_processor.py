import logging
import json
from typing import Any, Optional
from PyPDF2 import PdfReader

try:
    from pdf2image import convert_from_bytes
    import pytesseract
    _OCR_AVAILABLE = True
except Exception:
    _OCR_AVAILABLE = False


class PDFProcessor:
    """
    PDF processing with a smart fallback:
    1) Try PyPDF2 text extraction.
    2) If almost empty → OCR each page (if deps are available).
    """

    @staticmethod
    def allowed_file(filename: str, allowed_extensions: set) -> bool:
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

    @staticmethod
    def _extract_text_pypdf2(file_stream) -> str:
        try:
            pdf_reader = PdfReader(file_stream)
            text_content = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text() or ""
                    text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    logging.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    text_content += f"\n--- Page {page_num + 1} ---\n[Text extraction failed]\n"
            return text_content.strip()
        except Exception as e:
            logging.error(f"Failed to extract text from PDF: {e}")
            raise

    @staticmethod
    def _extract_text_ocr(pdf_bytes: bytes, langs: str = "heb+eng") -> str:
        """
        OCR fallback using pdf2image + pytesseract.
        Requires system Tesseract installed + language data (heb/eng).
        """
        if not _OCR_AVAILABLE:
            return ""

        try:
            pages = convert_from_bytes(pdf_bytes, dpi=300)  # needs poppler on some setups
        except Exception as e:
            logging.warning(f"OCR: failed to render PDF pages: {e}")
            return ""

        text_content = []
        for idx, img in enumerate(pages, start=1):
            try:
                page_text = pytesseract.image_to_string(img, lang=langs) or ""
                text_content.append(f"\n--- Page {idx} ---\n{page_text}\n")
            except Exception as e:
                logging.warning(f"OCR: failed on page {idx}: {e}")
                text_content.append(f"\n--- Page {idx} ---\n[OCR failed]\n")

        return "".join(text_content).strip()

    @staticmethod
    def extract_text_from_pdf(file_stream) -> str:
        """
        First try PyPDF2; if result is basically empty → OCR fallback (if available).
        """
        # Read all bytes so we can re-open for OCR if needed
        try:
            pdf_bytes = file_stream.read()
        except Exception as e:
            logging.error(f"Failed to read PDF bytes: {e}")
            raise

        # 1) primary: PyPDF2
        from io import BytesIO
        text = PDFProcessor._extract_text_pypdf2(BytesIO(pdf_bytes))

        # Heuristic: if too short, try OCR
        if len(text.replace("-", "").replace("\n", "").strip()) < 50:
            ocr_text = PDFProcessor._extract_text_ocr(pdf_bytes)
            if ocr_text:
                return ocr_text

        return text


def json_payload_len(path: str, prefer: Optional[str] = None) -> int:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data: Any = json.load(f)
        if isinstance(data, list):
            return len(data)
        if isinstance(data, dict):
            if prefer and isinstance(data.get(prefer), list):
                return len(data[prefer])
            if isinstance(data.get("majors"), list):
                return len(data["majors"])
            if isinstance(data.get("items"), list):
                return len(data["items"])
            for v in data.values():
                if isinstance(v, list):
                    return len(v)
    except Exception:
        pass
    return 0

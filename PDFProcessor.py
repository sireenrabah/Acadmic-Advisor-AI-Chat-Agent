import logging
from PyPDF2 import PdfReader

class PDFProcessor:
    """
    PDF processing class to handle PDF file operations
    Provides methods for reading and extracting content from PDF files
    """

    @staticmethod
    def allowed_file(filename: str, allowed_extensions: set) -> bool:
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in allowed_extensions

    @staticmethod
    def extract_text_from_pdf(file_stream) -> str:
        try:
            pdf_reader = PdfReader(file_stream)
            text_content = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                except Exception as e:
                    logging.warning(f"Failed to extract text from page {page_num + 1}: {e}")
                    text_content += f"\n--- Page {page_num + 1} ---\n[Text extraction failed]\n"
            return text_content.strip()
        except Exception as e:
            logging.error(f"Failed to extract text from PDF: {e}")
            raise

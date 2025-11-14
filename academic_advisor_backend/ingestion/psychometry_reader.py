"""
PsychometryReader: Extract psychometric test scores from PDF/image files.

Handles Israeli psychometric test results with:
- Overall score (200-800)
- Quantitative reasoning score
- Verbal reasoning score  
- English score
- Test date
- Test taker name

Uses OCR (Tesseract) + LLM fallback for robust extraction.
"""

from __future__ import annotations
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# OCR
try:
    import pytesseract
    from PIL import Image
    import fitz  # PyMuPDF for PDF
except ImportError:
    pytesseract = None
    Image = None
    fitz = None

# LLM fallback
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    import os
except ImportError:
    ChatGoogleGenerativeAI = None


class PsychometryReader:
    """Extract psychometric test scores from PDF/image files."""
    
    def __init__(
        self,
        use_llm_fallback: bool = True,
        debug: bool = False,
    ):
        """
        Args:
            use_llm_fallback: If True, use LLM when OCR extraction fails
            debug: If True, save debug output to psychometry_text_debug.txt
        """
        self.use_llm_fallback = use_llm_fallback
        self.debug = debug
        self.llm = None
        
        if use_llm_fallback and ChatGoogleGenerativeAI:
            api_key = os.getenv("GOOGLE_API_KEY", "").strip()
            if api_key:
                try:
                    self.llm = ChatGoogleGenerativeAI(
                        model="gemini-2.0-flash",
                        google_api_key=api_key,
                        temperature=0.0
                    )
                except Exception as e:
                    print(f"[psychometry:warning] Could not initialize LLM: {e}")
    
    def run(
        self,
        pdf_path: str,
        out_json: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Main entry point: extract psychometric scores from PDF/image.
        
        Args:
            pdf_path: Path to psychometric test PDF or image
            out_json: Optional path to save extracted JSON
            
        Returns:
            (scores_dict, summary_text)
            
        Example output:
        {
            "overall_score": 725,
            "quantitative": 148,
            "verbal": 142,
            "english": 135,
            "test_date": "2024-01-15",
            "test_taker": "John Doe"
        }
        """
        # Extract text from PDF/image
        text = self._extract_text(pdf_path)
        
        if self.debug:
            debug_path = Path(pdf_path).parent / "psychometry_text_debug.txt"
            debug_path.write_text(text, encoding="utf-8")
            print(f"[psychometry:debug] Saved raw text to {debug_path}")
        
        # Try heuristic extraction first
        scores = self._extract_scores_heuristic(text)
        
        # If heuristic failed, try LLM fallback
        if not scores.get("overall_score") and self.llm and self.use_llm_fallback:
            print("[psychometry] Heuristic extraction failed, trying LLM fallback...")
            scores = self._extract_scores_llm(text)
        
        # Save to JSON if requested
        if out_json:
            out_path = Path(out_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[psychometry] Saved to {out_json}")
        
        # Generate summary
        summary = self._generate_summary(scores)
        
        return scores, summary
    
    def _extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF or image file using OCR."""
        path = Path(pdf_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {pdf_path}")
        
        # Handle image files
        if path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            if not pytesseract or not Image:
                raise RuntimeError("pytesseract and Pillow required for image processing")
            
            img = Image.open(path)
            text = pytesseract.image_to_string(img, lang="heb+eng")
            return text
        
        # Handle PDF files
        if not fitz:
            raise RuntimeError("PyMuPDF (fitz) required for PDF processing")
        
        doc = fitz.open(pdf_path)
        all_text = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Try text extraction first
            page_text = page.get_text()
            
            # If no text, use OCR
            if not page_text.strip():
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img, lang="heb+eng")
            
            all_text.append(page_text)
        
        doc.close()
        return "\n\n".join(all_text)
    
    def _extract_scores_heuristic(self, text: str) -> Dict[str, Any]:
        """Extract scores using regex patterns."""
        scores = {}
        
        # Overall score (200-800 range)
        # Look for patterns like "ציון כללי: 725" or "Total Score: 725"
        overall_patterns = [
            r"(?:ציון כללי|כללי|Total Score|Overall)[\s:]*(\d{3})",
            r"(\d{3})\s*(?:ציון כללי|כללי)",
            r"(?:^|\s)([2-8]\d{2})(?:\s|$)",  # Any 3-digit number 200-899
        ]
        
        for pattern in overall_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                score = int(match.group(1))
                if 200 <= score <= 800:
                    scores["overall_score"] = score
                    break
        
        # Quantitative reasoning (100-200 range)
        quant_patterns = [
            r"(?:חשיבה כמותית|כמותי|Quantitative|Math)[\s:]*(\d{3})",
            r"(\d{3})\s*(?:חשיבה כמותית|כמותי)",
        ]
        
        for pattern in quant_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                if 50 <= score <= 200:
                    scores["quantitative"] = score
                    break
        
        # Verbal reasoning (100-200 range)
        verbal_patterns = [
            r"(?:חשיבה מילולית|מילולי|Verbal)[\s:]*(\d{3})",
            r"(\d{3})\s*(?:חשיבה מילולית|מילולי)",
        ]
        
        for pattern in verbal_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                if 50 <= score <= 200:
                    scores["verbal"] = score
                    break
        
        # English score (100-200 range)
        english_patterns = [
            r"(?:אנגלית|English)[\s:]*(\d{3})",
            r"(\d{3})\s*(?:אנגלית|English)",
        ]
        
        for pattern in english_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                if 50 <= score <= 200:
                    scores["english"] = score
                    break
        
        # Test date (various formats)
        date_patterns = [
            r"(\d{1,2}[/-]\d{1,2}[/-]\d{4})",  # DD/MM/YYYY or DD-MM-YYYY
            r"(\d{4}[/-]\d{1,2}[/-]\d{1,2})",  # YYYY/MM/DD or YYYY-MM-DD
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                scores["test_date"] = match.group(1)
                break
        
        # Test taker name (look for Hebrew/English name patterns)
        name_patterns = [
            r"(?:שם|Name)[\s:]*([א-תa-zA-Z\s]{2,50})",
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                if len(name) > 2:
                    scores["test_taker"] = name
                    break
        
        return scores
    
    def _extract_scores_llm(self, text: str) -> Dict[str, Any]:
        """Use LLM to extract scores from text."""
        if not self.llm:
            return {}
        
        prompt = f"""Extract psychometric test scores from this Hebrew/English document.

Document text:
{text[:3000]}

Extract the following information in JSON format:
{{
    "overall_score": <integer 200-800, or null if not found>,
    "quantitative": <integer 50-200 for quantitative reasoning, or null>,
    "verbal": <integer 50-200 for verbal reasoning, or null>,
    "english": <integer 50-200 for English, or null>,
    "test_date": "<date string in YYYY-MM-DD format, or null>",
    "test_taker": "<name of test taker, or null>"
}}

Notes:
- Overall score is typically 200-800 range (ציון כללי)
- Section scores are typically 50-200 range
- Return null for any field not found in the document
- Return ONLY valid JSON, no other text

JSON:"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            scores = json.loads(content)
            
            # Validate score ranges
            if scores.get("overall_score"):
                if not (200 <= scores["overall_score"] <= 800):
                    scores["overall_score"] = None
            
            for section in ["quantitative", "verbal", "english"]:
                if scores.get(section):
                    if not (50 <= scores[section] <= 200):
                        scores[section] = None
            
            return scores
            
        except Exception as e:
            print(f"[psychometry:llm_error] {e}")
            return {}
    
    def _generate_summary(self, scores: Dict[str, Any]) -> str:
        """Generate human-readable summary of scores."""
        parts = []
        
        if scores.get("test_taker"):
            parts.append(f"Test Taker: {scores['test_taker']}")
        
        if scores.get("test_date"):
            parts.append(f"Test Date: {scores['test_date']}")
        
        if scores.get("overall_score"):
            parts.append(f"Overall Score: {scores['overall_score']}/800")
        
        if scores.get("quantitative"):
            parts.append(f"Quantitative Reasoning: {scores['quantitative']}/200")
        
        if scores.get("verbal"):
            parts.append(f"Verbal Reasoning: {scores['verbal']}/200")
        
        if scores.get("english"):
            parts.append(f"English: {scores['english']}/200")
        
        if not parts:
            return "No psychometric scores found in document"
        
        return "\n".join(parts)


def main():
    """Test the psychometry reader."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python psychometry_reader.py <pdf_path> [out_json]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    out_json = sys.argv[2] if len(sys.argv) > 2 else None
    
    reader = PsychometryReader(use_llm_fallback=True, debug=True)
    scores, summary = reader.run(pdf_path, out_json)
    
    print("\n" + "="*60)
    print("PSYCHOMETRIC TEST RESULTS")
    print("="*60)
    print(summary)
    print("="*60)
    
    if out_json:
        print(f"\nFull data saved to: {out_json}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from pathlib import Path

try:
    import pymupdf
except Exception:  # pragma: no cover
    pymupdf = None

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None


class OCRProcessor:
    """Extracts text from uploaded documents and decides whether OCR is needed."""

    def __init__(self, lang: str = "eng", min_confidence: float = 70.0):
        self.lang = lang
        self.min_confidence = min_confidence

    def needs_ocr(self, file_path: str) -> bool:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md", ".csv"}:
            return False
        if suffix in {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            return True
        return False

    def extract_text(self, file_path: str) -> tuple[str, bool, float]:
        pages = self.extract_pages(file_path)
        text = "\n\n".join(page_text for page_text, _, _ in pages)
        ocr_confidences = [confidence for _, used_ocr, confidence in pages if used_ocr and confidence is not None]
        used_ocr = bool(ocr_confidences)
        confidence = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else 100.0
        return text, used_ocr, confidence

    def extract_pages(self, file_path: str) -> list[tuple[str, bool, float | None]]:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix in {".txt", ".md", ".csv"}:
            try:
                text = path.read_text(encoding="utf-8")
            except Exception:
                text = path.read_text(errors="ignore", encoding="utf-8")
            sections = [section.strip() for section in text.split("\n\n") if section.strip()]
            return [(section, False, None) for section in sections or [text.strip()]]

        if suffix == ".pdf":
            if pymupdf is None:
                return []
            doc = pymupdf.open(file_path)
            extracted: list[tuple[str, bool, float | None]] = []
            for page in doc:
                text = page.get_text() or ""
                if text.strip():
                    extracted.append((text, False, None))
                    continue
                if pytesseract is not None and Image is not None:
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
                    image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_text, confidence = self._extract_ocr(image)
                    extracted.append((ocr_text, True, confidence))
                else:
                    extracted.append(("", True, 0.0))
            return extracted

        if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            if pytesseract is not None and Image is not None:
                image = Image.open(path)
                text, confidence = self._extract_ocr(image)
                return [(text, True, confidence)]
            return [("", True, 0.0)]

        text = path.read_text(errors="ignore", encoding="utf-8")
        return [(text, False, None)]

    def _extract_ocr(self, image) -> tuple[str, float]:
        text = pytesseract.image_to_string(image, lang=self.lang)
        data = pytesseract.image_to_data(
            image,
            lang=self.lang,
            output_type=pytesseract.Output.DICT,
        )
        scores: list[float] = []
        for value in data.get("conf", []):
            try:
                score = float(value)
            except (TypeError, ValueError):
                continue
            if score >= 0:
                scores.append(score)
        confidence = sum(scores) / len(scores) if scores else 0.0
        return text, confidence

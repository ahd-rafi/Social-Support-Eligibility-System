"""Tests for document parsers."""

import pytest
from pathlib import Path


def test_parser_imports():
    """Test that all parser modules can be imported."""
    from src.parsers import (
        OCRProcessor,
        PDFParser,
        ExcelParser,
        ParserRouter,
        DocumentType,
    )
    
    assert OCRProcessor is not None
    assert PDFParser is not None
    assert ExcelParser is not None
    assert ParserRouter is not None
    assert DocumentType is not None


def test_document_type_enum():
    """Test DocumentType enum."""
    from src.parsers import DocumentType
    
    assert DocumentType.EMIRATES_ID == "emirates_id"
    assert DocumentType.BANK_STATEMENT == "bank_statement"
    assert DocumentType.RESUME == "resume"
    assert DocumentType.CREDIT_REPORT == "credit_report"
    assert DocumentType.ASSETS_LIABILITIES == "assets_liabilities"


def test_parser_router_detection():
    """Test parser router document type detection."""
    from src.parsers import ParserRouter, DocumentType
    
    router = ParserRouter()
    
    # Test filename-based detection
    assert router.detect_document_type(Path("emirates_id.png")) == DocumentType.EMIRATES_ID
    assert router.detect_document_type(Path("bank_statement.pdf")) == DocumentType.BANK_STATEMENT
    assert router.detect_document_type(Path("resume.pdf")) == DocumentType.RESUME
    assert router.detect_document_type(Path("credit_report.pdf")) == DocumentType.CREDIT_REPORT
    assert router.detect_document_type(Path("assets_liabilities.xlsx")) == DocumentType.ASSETS_LIABILITIES


def test_parser_router_hint():
    """Test parser router with hints."""
    from src.parsers import ParserRouter, DocumentType
    
    router = ParserRouter()
    
    # Test hint-based detection (overrides filename)
    assert router.detect_document_type(Path("document.pdf"), hint="bank statement") == DocumentType.BANK_STATEMENT
    assert router.detect_document_type(Path("file.pdf"), hint="resume") == DocumentType.RESUME


def test_ocr_processor_init():
    """Test OCR processor initialization."""
    from src.parsers import OCRProcessor
    
    # Should not raise even if Tesseract not installed
    try:
        processor = OCRProcessor()
        # Check if available
        available = processor.is_available()
        print(f"Tesseract available: {available}")
    except ImportError:
        pytest.skip("pytesseract not installed")


def test_pdf_parser_init():
    """Test PDF parser initialization."""
    from src.parsers import PDFParser
    
    try:
        parser = PDFParser()
        assert parser is not None
    except ImportError:
        pytest.skip("PyPDF2 not installed")


def test_excel_parser_init():
    """Test Excel parser initialization."""
    from src.parsers import ExcelParser
    
    try:
        parser = ExcelParser()
        assert parser is not None
    except ImportError:
        pytest.skip("pandas/openpyxl not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

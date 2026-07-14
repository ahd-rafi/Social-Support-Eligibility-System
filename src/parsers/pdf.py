"""PDF text extraction using PyPDF2."""

import logging
from pathlib import Path
from typing import List, Dict, Any

try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

logger = logging.getLogger(__name__)


class PDFParser:
    """Extract text from PDF files."""

    def __init__(self):
        """Initialize PDF parser."""
        if PdfReader is None:
            raise ImportError(
                "PyPDF2 is required for PDF parsing. "
                "Install with: pip install PyPDF2"
            )
        logger.info("PDF parser initialized")

    def extract_text(self, pdf_path: Path) -> str:
        """Extract all text from a PDF file.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text as string
        """
        try:
            logger.debug(f"Processing PDF: {pdf_path}")
            
            reader = PdfReader(pdf_path)
            
            text_parts = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                    logger.debug(f"  Page {i+1}: {len(page_text)} characters")
            
            full_text = "\n\n".join(text_parts)
            
            logger.debug(f"Extracted {len(full_text)} total characters from {len(reader.pages)} pages")
            
            return full_text
            
        except Exception as e:
            logger.error(f"PDF parsing failed for {pdf_path}: {e}")
            raise

    def extract_pages(self, pdf_path: Path) -> List[str]:
        """Extract text from each page separately.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of text strings, one per page
        """
        try:
            reader = PdfReader(pdf_path)
            
            pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
                else:
                    pages.append("")
                logger.debug(f"  Page {i+1}: {len(pages[-1])} characters")
            
            return pages
            
        except Exception as e:
            logger.error(f"PDF page extraction failed for {pdf_path}: {e}")
            raise

    def get_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract PDF metadata.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dict with metadata fields
        """
        try:
            reader = PdfReader(pdf_path)
            
            metadata = {
                "num_pages": len(reader.pages),
                "pdf_version": reader.pdf_header if hasattr(reader, 'pdf_header') else None,
            }
            
            # Add document info if available
            if reader.metadata:
                for key, value in reader.metadata.items():
                    # Remove leading slash from key names
                    clean_key = key.lstrip('/')
                    metadata[clean_key] = value
            
            return metadata
            
        except Exception as e:
            logger.error(f"PDF metadata extraction failed for {pdf_path}: {e}")
            return {}

    def extract_with_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text and metadata from PDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dict with 'text', 'pages', and 'metadata' keys
        """
        return {
            "text": self.extract_text(pdf_path),
            "pages": self.extract_pages(pdf_path),
            "metadata": self.get_metadata(pdf_path),
        }


# Specialized parsers for specific document types

class BankStatementParser(PDFParser):
    """Parse bank statement PDFs."""

    def parse(self, pdf_path: Path) -> Dict[str, Any]:
        """Parse a bank statement PDF into structured data.
        
        Args:
            pdf_path: Path to bank statement PDF
            
        Returns:
            Dict with parsed fields
        """
        text = self.extract_text(pdf_path)
        
        # This is a simplified parser - production would need more robust parsing
        # In the actual system, this will be enhanced by LLM (Phi-4-mini) for structuring
        
        data = {
            "raw_text": text,
            "account_holder": None,
            "bank_name": None,
            "account_number": None,
            "statement_period": None,
            "transactions": [],
            "summary": {
                "opening_balance": None,
                "closing_balance": None,
                "total_credits": None,
                "total_debits": None,
            }
        }
        
        logger.info(f"Parsed bank statement: {pdf_path.name}")
        logger.debug(f"Extracted {len(text)} characters")
        
        return data


class ResumeParser(PDFParser):
    """Parse resume PDFs."""

    def parse(self, pdf_path: Path) -> Dict[str, Any]:
        """Parse a resume PDF into structured data.
        
        Args:
            pdf_path: Path to resume PDF
            
        Returns:
            Dict with parsed fields
        """
        text = self.extract_text(pdf_path)
        
        # Simplified parser - will be enhanced by LLM
        data = {
            "raw_text": text,
            "name": None,
            "contact": {
                "email": None,
                "phone": None,
            },
            "experience": [],
            "education": [],
            "skills": [],
        }
        
        logger.info(f"Parsed resume: {pdf_path.name}")
        logger.debug(f"Extracted {len(text)} characters")
        
        return data


class CreditReportParser(PDFParser):
    """Parse credit report PDFs."""

    def parse(self, pdf_path: Path) -> Dict[str, Any]:
        """Parse a credit report PDF into structured data.
        
        Args:
            pdf_path: Path to credit report PDF
            
        Returns:
            Dict with parsed fields
        """
        text = self.extract_text(pdf_path)
        
        # Simplified parser - will be enhanced by LLM
        data = {
            "raw_text": text,
            "credit_score": None,
            "report_date": None,
            "accounts": [],
            "outstanding_debts": None,
            "payment_history": {
                "on_time_payments": None,
                "missed_payments": None,
            },
            "address": None,
            "employer": None,
        }
        
        logger.info(f"Parsed credit report: {pdf_path.name}")
        logger.debug(f"Extracted {len(text)} characters")
        
        return data


# Global instances
_pdf_parser = None
_bank_statement_parser = None
_resume_parser = None
_credit_report_parser = None


def get_pdf_parser() -> PDFParser:
    """Get or create global PDF parser instance."""
    global _pdf_parser
    if _pdf_parser is None:
        _pdf_parser = PDFParser()
    return _pdf_parser


def get_bank_statement_parser() -> BankStatementParser:
    """Get or create global bank statement parser."""
    global _bank_statement_parser
    if _bank_statement_parser is None:
        _bank_statement_parser = BankStatementParser()
    return _bank_statement_parser


def get_resume_parser() -> ResumeParser:
    """Get or create global resume parser."""
    global _resume_parser
    if _resume_parser is None:
        _resume_parser = ResumeParser()
    return _resume_parser


def get_credit_report_parser() -> CreditReportParser:
    """Get or create global credit report parser."""
    global _credit_report_parser
    if _credit_report_parser is None:
        _credit_report_parser = CreditReportParser()
    return _credit_report_parser


if __name__ == "__main__":
    # Test PDF parser
    logging.basicConfig(level=logging.INFO)
    
    parser = PDFParser()
    print("✓ PDF parser initialized")
    print(f"  Available parsers: PDFParser, BankStatementParser, ResumeParser, CreditReportParser")

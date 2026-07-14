"""Parser router - routes files to appropriate parsers based on type."""

import logging
import mimetypes
from pathlib import Path
from typing import Dict, Any, Optional
from enum import Enum

from src.parsers.ocr import get_ocr_processor
from src.parsers.pdf import (
    get_pdf_parser,
    get_bank_statement_parser,
    get_resume_parser,
    get_credit_report_parser,
)
from src.parsers.excel import get_excel_parser, get_assets_liabilities_parser

logger = logging.getLogger(__name__)

# Force reload OCR with fixes
OCR_RELOAD_MARKER = "v2.0"


class DocumentType(str, Enum):
    """Supported document types."""
    EMIRATES_ID = "emirates_id"
    BANK_STATEMENT = "bank_statement"
    RESUME = "resume"
    CREDIT_REPORT = "credit_report"
    ASSETS_LIABILITIES = "assets_liabilities"
    UNKNOWN = "unknown"


class ParserRouter:
    """Route documents to appropriate parsers."""

    def __init__(self):
        """Initialize parser router."""
        # Force reload OCR to get latest fixes
        self.ocr = get_ocr_processor(force_reload=True)
        self.pdf_parser = get_pdf_parser()
        self.excel_parser = get_excel_parser()
        
        # Specialized parsers
        self.bank_statement_parser = get_bank_statement_parser()
        self.resume_parser = get_resume_parser()
        self.credit_report_parser = get_credit_report_parser()
        self.assets_liabilities_parser = get_assets_liabilities_parser()
        
        logger.info("Parser router initialized")

    def detect_document_type(
        self,
        file_path: Path,
        hint: Optional[str] = None,
    ) -> DocumentType:
        """Detect document type from filename and optional hint.
        
        Args:
            file_path: Path to document
            hint: Optional hint about document type from user/context
            
        Returns:
            Detected DocumentType
        """
        filename = file_path.name.lower()
        
        # Check hint first
        if hint:
            hint_lower = hint.lower()
            if "emirates" in hint_lower or "id" in hint_lower:
                return DocumentType.EMIRATES_ID
            elif "bank" in hint_lower or "statement" in hint_lower:
                return DocumentType.BANK_STATEMENT
            elif "resume" in hint_lower or "cv" in hint_lower:
                return DocumentType.RESUME
            elif "credit" in hint_lower or "report" in hint_lower:
                return DocumentType.CREDIT_REPORT
            elif "asset" in hint_lower or "liabilit" in hint_lower:
                return DocumentType.ASSETS_LIABILITIES
        
        # Check filename patterns
        if "emirates" in filename or "eid" in filename or "_id" in filename:
            return DocumentType.EMIRATES_ID
        elif "bank" in filename or "statement" in filename:
            return DocumentType.BANK_STATEMENT
        elif "resume" in filename or "cv" in filename:
            return DocumentType.RESUME
        elif "credit" in filename or "report" in filename:
            return DocumentType.CREDIT_REPORT
        elif "asset" in filename or "liabilit" in filename:
            return DocumentType.ASSETS_LIABILITIES
        
        return DocumentType.UNKNOWN

    def get_file_type(self, file_path: Path) -> str:
        """Get MIME type or file extension.
        
        Args:
            file_path: Path to file
            
        Returns:
            MIME type or extension
        """
        # Try MIME type first
        mime_type, _ = mimetypes.guess_type(str(file_path))
        if mime_type:
            return mime_type
        
        # Fall back to extension
        return file_path.suffix.lower()

    def parse(
        self,
        file_path: Path,
        document_type: Optional[DocumentType] = None,
        hint: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Parse a document using the appropriate parser.
        
        Args:
            file_path: Path to document
            document_type: Optional explicit document type
            hint: Optional hint about document type
            
        Returns:
            Dict with parsed data and metadata
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Detect document type if not provided
        if document_type is None:
            document_type = self.detect_document_type(file_path, hint)
        
        logger.info(f"Parsing {document_type.value}: {file_path.name}")
        
        # Get file type
        file_type = self.get_file_type(file_path)
        
        result = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_type": file_type,
            "document_type": document_type.value,
            "data": None,
            "raw_text": None,
        }
        
        try:
            # Route to appropriate parser based on file type and document type
            if file_type in ["image/png", "image/jpeg", "image/jpg", ".png", ".jpg", ".jpeg"]:
                # Image files - use OCR
                if document_type == DocumentType.EMIRATES_ID:
                    result["data"] = self.ocr.extract_from_emirates_id(file_path)
                else:
                    result["raw_text"] = self.ocr.extract_text(file_path)
                    result["data"] = {"text": result["raw_text"]}
            
            elif file_type in ["application/pdf", ".pdf"]:
                # PDF files - use specialized parsers
                if document_type == DocumentType.BANK_STATEMENT:
                    result["data"] = self.bank_statement_parser.parse(file_path)
                elif document_type == DocumentType.RESUME:
                    result["data"] = self.resume_parser.parse(file_path)
                elif document_type == DocumentType.CREDIT_REPORT:
                    result["data"] = self.credit_report_parser.parse(file_path)
                else:
                    # Generic PDF parsing
                    result["raw_text"] = self.pdf_parser.extract_text(file_path)
                    result["data"] = {"text": result["raw_text"]}
                
                # Extract raw_text if available
                if result["data"] and "raw_text" in result["data"]:
                    result["raw_text"] = result["data"]["raw_text"]
            
            elif file_type in [
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.ms-excel",
                ".xlsx",
                ".xls",
            ]:
                # Excel files
                if document_type == DocumentType.ASSETS_LIABILITIES:
                    result["data"] = self.assets_liabilities_parser.parse(file_path)
                else:
                    # Generic Excel parsing
                    sheets = self.excel_parser.read_all_sheets(file_path)
                    result["data"] = {
                        sheet_name: self.excel_parser.dataframe_to_dict(df)
                        for sheet_name, df in sheets.items()
                    }
            
            elif file_type in [
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".docx",
            ]:
                # Word documents - would need python-docx for proper parsing
                # For now, treat as unknown
                logger.warning(f"DOCX parsing not fully implemented: {file_path}")
                result["data"] = {"error": "DOCX parsing not implemented"}
            
            else:
                logger.warning(f"Unknown file type: {file_type}")
                result["data"] = {"error": f"Unsupported file type: {file_type}"}
            
            logger.info(f"✓ Parsed {document_type.value}: {file_path.name}")
            
        except Exception as e:
            logger.error(f"Parsing failed for {file_path}: {e}", exc_info=True)
            result["data"] = {"error": str(e)}
        
        return result

    def parse_all_documents(
        self,
        documents: Dict[str, Path],
    ) -> Dict[str, Dict[str, Any]]:
        """Parse multiple documents.
        
        Args:
            documents: Dict mapping document types to file paths
            
        Returns:
            Dict mapping document types to parsed results
        """
        results = {}
        
        for doc_type_str, file_path in documents.items():
            try:
                # Convert string to DocumentType enum
                doc_type = DocumentType(doc_type_str) if doc_type_str in DocumentType.__members__.values() else DocumentType.UNKNOWN
                
                results[doc_type_str] = self.parse(file_path, document_type=doc_type)
                
            except Exception as e:
                logger.error(f"Failed to parse {doc_type_str}: {e}")
                results[doc_type_str] = {
                    "file_path": str(file_path),
                    "document_type": doc_type_str,
                    "data": {"error": str(e)},
                }
        
        return results


# Global instance
_parser_router = None


def get_parser_router() -> ParserRouter:
    """Get or create global parser router instance."""
    global _parser_router
    if _parser_router is None:
        _parser_router = ParserRouter()
    return _parser_router


if __name__ == "__main__":
    # Test parser router
    logging.basicConfig(level=logging.INFO)
    
    router = ParserRouter()
    print("✓ Parser router initialized")
    print(f"  Supported document types: {[dt.value for dt in DocumentType]}")
    
    # Test document type detection
    test_files = [
        "emirates_id.png",
        "bank_statement.pdf",
        "resume.pdf",
        "credit_report.pdf",
        "assets_liabilities.xlsx",
    ]
    
    print("\nDocument type detection:")
    for filename in test_files:
        file_path = Path(filename)
        doc_type = router.detect_document_type(file_path)
        print(f"  {filename} → {doc_type.value}")

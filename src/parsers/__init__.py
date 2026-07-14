"""Document parsers for multimodal data processing."""

from src.parsers.ocr import OCRProcessor, get_ocr_processor
from src.parsers.pdf import (
    PDFParser,
    BankStatementParser,
    ResumeParser,
    CreditReportParser,
    get_pdf_parser,
    get_bank_statement_parser,
    get_resume_parser,
    get_credit_report_parser,
)
from src.parsers.excel import (
    ExcelParser,
    AssetsLiabilitiesParser,
    get_excel_parser,
    get_assets_liabilities_parser,
)
from src.parsers.router import (
    ParserRouter,
    DocumentType,
    get_parser_router,
)

__all__ = [
    "OCRProcessor",
    "get_ocr_processor",
    "PDFParser",
    "BankStatementParser",
    "ResumeParser",
    "CreditReportParser",
    "get_pdf_parser",
    "get_bank_statement_parser",
    "get_resume_parser",
    "get_credit_report_parser",
    "ExcelParser",
    "AssetsLiabilitiesParser",
    "get_excel_parser",
    "get_assets_liabilities_parser",
    "ParserRouter",
    "DocumentType",
    "get_parser_router",
]

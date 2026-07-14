"""Excel file parsing using pandas and openpyxl."""

import logging
from pathlib import Path
from typing import Dict, Any, List

try:
    import pandas as pd
    import openpyxl
except ImportError:
    pd = None
    openpyxl = None

logger = logging.getLogger(__name__)


class ExcelParser:
    """Parse Excel files for structured data extraction."""

    def __init__(self):
        """Initialize Excel parser."""
        if pd is None or openpyxl is None:
            raise ImportError(
                "pandas and openpyxl are required for Excel parsing. "
                "Install with: pip install pandas openpyxl"
            )
        logger.info("Excel parser initialized")

    def read_sheet(self, excel_path: Path, sheet_name: str = 0) -> pd.DataFrame:
        """Read a single sheet from Excel file.
        
        Args:
            excel_path: Path to Excel file
            sheet_name: Sheet name or index (default: 0 for first sheet)
            
        Returns:
            DataFrame with sheet data
        """
        try:
            logger.debug(f"Reading sheet '{sheet_name}' from {excel_path}")
            df = pd.read_excel(excel_path, sheet_name=sheet_name, engine='openpyxl')
            logger.debug(f"  Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to read sheet '{sheet_name}' from {excel_path}: {e}")
            raise

    def read_all_sheets(self, excel_path: Path) -> Dict[str, pd.DataFrame]:
        """Read all sheets from Excel file.
        
        Args:
            excel_path: Path to Excel file
            
        Returns:
            Dict mapping sheet names to DataFrames
        """
        try:
            logger.debug(f"Reading all sheets from {excel_path}")
            sheets = pd.read_excel(excel_path, sheet_name=None, engine='openpyxl')
            logger.debug(f"  Found {len(sheets)} sheets: {list(sheets.keys())}")
            return sheets
            
        except Exception as e:
            logger.error(f"Failed to read sheets from {excel_path}: {e}")
            raise

    def dataframe_to_dict(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Convert DataFrame to list of dictionaries.
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            List of dicts, one per row
        """
        # Replace NaN with None for JSON serialization
        df = df.where(pd.notnull(df), None)
        return df.to_dict('records')

    def get_summary_stats(self, df: pd.DataFrame, numeric_only: bool = True) -> Dict[str, Any]:
        """Get summary statistics for DataFrame.
        
        Args:
            df: Pandas DataFrame
            numeric_only: Only compute for numeric columns
            
        Returns:
            Dict with summary statistics
        """
        stats = {}
        
        if numeric_only:
            numeric_cols = df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                stats[col] = {
                    "count": int(df[col].count()),
                    "sum": float(df[col].sum()) if not pd.isna(df[col].sum()) else None,
                    "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                    "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                    "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                }
        
        return stats


class AssetsLiabilitiesParser(ExcelParser):
    """Parse Assets/Liabilities Excel files."""

    def parse(self, excel_path: Path) -> Dict[str, Any]:
        """Parse an assets/liabilities spreadsheet.
        
        Expected format:
        - Sheet 1 "Assets": Type, Description, Value columns
        - Sheet 2 "Liabilities": Type, Description, Amount, Monthly Payment columns
        
        Args:
            excel_path: Path to Excel file
            
        Returns:
            Dict with parsed assets and liabilities
        """
        try:
            logger.info(f"Parsing assets/liabilities: {excel_path}")
            
            sheets = self.read_all_sheets(excel_path)
            
            data = {
                "assets": [],
                "liabilities": [],
                "summary": {
                    "total_assets": 0,
                    "total_liabilities": 0,
                    "net_worth": 0,
                }
            }
            
            # Parse assets sheet
            assets_sheet = None
            for name in sheets.keys():
                if 'asset' in name.lower():
                    assets_sheet = sheets[name]
                    break
            
            if assets_sheet is not None and not assets_sheet.empty:
                # Handle different possible column names
                value_col = None
                for col in assets_sheet.columns:
                    if 'value' in col.lower() or 'amount' in col.lower():
                        value_col = col
                        break
                
                if value_col:
                    assets_list = self.dataframe_to_dict(assets_sheet)
                    data["assets"] = assets_list
                    
                    # Calculate total
                    total_assets = assets_sheet[value_col].sum()
                    data["summary"]["total_assets"] = float(total_assets) if not pd.isna(total_assets) else 0
            
            # Parse liabilities sheet
            liabilities_sheet = None
            for name in sheets.keys():
                if 'liabilit' in name.lower() or 'debt' in name.lower():
                    liabilities_sheet = sheets[name]
                    break
            
            if liabilities_sheet is not None and not liabilities_sheet.empty:
                # Handle different possible column names
                amount_col = None
                for col in liabilities_sheet.columns:
                    if 'amount' in col.lower() or 'balance' in col.lower() or 'value' in col.lower():
                        amount_col = col
                        break
                
                if amount_col:
                    liabilities_list = self.dataframe_to_dict(liabilities_sheet)
                    data["liabilities"] = liabilities_list
                    
                    # Calculate total
                    total_liabilities = liabilities_sheet[amount_col].sum()
                    data["summary"]["total_liabilities"] = float(total_liabilities) if not pd.isna(total_liabilities) else 0
            
            # Calculate net worth
            data["summary"]["net_worth"] = (
                data["summary"]["total_assets"] - data["summary"]["total_liabilities"]
            )
            
            logger.info(f"  Assets: {len(data['assets'])} items, AED {data['summary']['total_assets']:,.0f}")
            logger.info(f"  Liabilities: {len(data['liabilities'])} items, AED {data['summary']['total_liabilities']:,.0f}")
            logger.info(f"  Net Worth: AED {data['summary']['net_worth']:,.0f}")
            
            return data
            
        except Exception as e:
            logger.error(f"Failed to parse assets/liabilities from {excel_path}: {e}")
            raise


# Global instances
_excel_parser = None
_assets_liabilities_parser = None


def get_excel_parser() -> ExcelParser:
    """Get or create global Excel parser instance."""
    global _excel_parser
    if _excel_parser is None:
        _excel_parser = ExcelParser()
    return _excel_parser


def get_assets_liabilities_parser() -> AssetsLiabilitiesParser:
    """Get or create global assets/liabilities parser."""
    global _assets_liabilities_parser
    if _assets_liabilities_parser is None:
        _assets_liabilities_parser = AssetsLiabilitiesParser()
    return _assets_liabilities_parser


if __name__ == "__main__":
    # Test Excel parser
    logging.basicConfig(level=logging.INFO)
    
    parser = ExcelParser()
    print("✓ Excel parser initialized")
    print(f"  Available parsers: ExcelParser, AssetsLiabilitiesParser")

"""OCR processing using Tesseract."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import os

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

logger = logging.getLogger(__name__)

# Version marker to force reload
OCR_VERSION = "2.0_FIXED"  # Updated with preprocessing


class OCRProcessor:
    """Extract text from images using Tesseract OCR."""

    def __init__(self, tesseract_cmd: Optional[str] = None):
        """Initialize OCR processor.
        
        Args:
            tesseract_cmd: Path to tesseract executable (optional, uses system PATH by default)
        """
        if pytesseract is None:
            raise ImportError(
                "pytesseract and Pillow are required for OCR. "
                "Install with: pip install pytesseract Pillow"
            )
        
        # Set tesseract command if provided
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        elif os.environ.get("TESSERACT_CMD"):
            pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]
        
        self.tesseract_cmd = pytesseract.pytesseract.tesseract_cmd
        logger.info(f"OCR processor initialized with Tesseract: {self.tesseract_cmd}")

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for better OCR accuracy.
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to grayscale
        image = image.convert('L')
        
        # Increase contrast and brightness
        from PIL import ImageEnhance
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)  # Increase contrast
        
        # Enhance brightness
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)  # Slightly brighten
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.5)
        
        return image

    def extract_text(
        self,
        image_path: Path,
        lang: str = "eng",
        config: str = "--psm 4",  # Changed from 6 to 4 (single column)
        preprocess: bool = True,  # Enable preprocessing by default
    ) -> str:
        """Extract text from an image file.
        
        Args:
            image_path: Path to image file
            lang: Language code (default: eng for English)
            config: Tesseract configuration string (default: PSM 6 = assume uniform block of text)
            
        Returns:
            Extracted text as string
        """
        try:
            logger.debug(f"Processing image: {image_path}")
            
            # Open image
            image = Image.open(image_path)
            
            # Preprocess image for better OCR
            if preprocess:
                image = self.preprocess_image(image)
                logger.debug("Image preprocessed (contrast enhanced)")
            
            # Perform OCR
            text = pytesseract.image_to_string(image, lang=lang, config=config)
            
            # Clean up
            text = text.strip()
            
            logger.debug(f"Extracted {len(text)} characters from {image_path.name}")
            
            return text
            
        except Exception as e:
            logger.error(f"OCR failed for {image_path}: {e}")
            raise

    def extract_with_confidence(
        self,
        image_path: Path,
        lang: str = "eng",
        config: str = "--psm 6",
    ) -> Dict[str, Any]:
        """Extract text with confidence scores.
        
        Args:
            image_path: Path to image file
            lang: Language code
            config: Tesseract configuration
            
        Returns:
            Dict with 'text', 'confidence', and 'data' keys
        """
        try:
            image = Image.open(image_path)
            
            # Get detailed data
            data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
            
            # Extract text
            text = pytesseract.image_to_string(image, lang=lang, config=config).strip()
            
            # Calculate average confidence (filter out -1 which means no detection)
            confidences = [int(conf) for conf in data['conf'] if int(conf) != -1]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            return {
                "text": text,
                "confidence": avg_confidence,
                "data": data,
            }
            
        except Exception as e:
            logger.error(f"OCR with confidence failed for {image_path}: {e}")
            raise

    def extract_from_emirates_id(self, image_path: Path) -> Dict[str, str]:
        """Extract structured data from Emirates ID image.
        
        This is a specialized parser that expects specific field layouts.
        
        Args:
            image_path: Path to Emirates ID image
            
        Returns:
            Dict with extracted fields
        """
        # Use PSM 4 (single column) for better label:value pair extraction with preprocessing
        text = self.extract_text(image_path, config="--psm 4", preprocess=True)
        
        # Clean common OCR errors for Emirates ID
        text = text.replace("lame", "Name")
        text = text.replace("lationality", "Nationality")
        text = text.replace("|D Number", "ID Number")
        text = text.replace("AREAL", "A REAL")
        
        logger.info(f"OCR extracted {len(text)} chars from Emirates ID")
        logger.debug(f"Cleaned OCR text: {text[:200]}")  # Log first 200 chars for debugging
        
        # Parse text into structured fields
        # This is a simple implementation - production would need more robust parsing
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        data = {
            "raw_text": text,
            "name": None,
            "id_number": None,
            "dob": None,
            "nationality": None,
            "gender": None,
            "emirate": None,
        }
        
        # Simple field extraction (looks for known patterns)
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Look for name (usually after "Name" or "English")
            if ("name" in line_lower and "english" in line_lower) or line_lower.startswith("name"):
                # Name is on the same line after colon or next line
                if ":" in line:
                    parts = line.split(":", 1)
                    if len(parts) > 1 and parts[1].strip():
                        data["name"] = parts[1].strip()
                elif i + 1 < len(lines):
                    next_line = lines[i + 1]
                    # Skip if next line looks like another field
                    if not any(keyword in next_line.lower() for keyword in ["id", "date", "birth", "nationality", "gender", "emirate", "synthetic"]):
                        data["name"] = next_line.strip()
            
            # Look for ID number pattern (784-YYYY-NNNNNNN-C)
            if "784-" in line:
                # Extract ID number
                parts = line.split()
                for part in parts:
                    if part.startswith("784-") and len(part) >= 15:
                        data["id_number"] = part
                        break
            
            # Look for ID Number field
            if "id number" in line_lower and ":" in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    potential_id = parts[1].strip()
                    if "784-" in potential_id:
                        data["id_number"] = potential_id.split()[0] if " " in potential_id else potential_id
            
            # Look for date patterns (YYYY-MM-DD) for DOB
            if ("birth" in line_lower or "dob" in line_lower) and ":" in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    date_str = parts[1].strip()
                    if date_str.count("-") == 2 and len(date_str) >= 10:
                        try:
                            from datetime import datetime
                            datetime.strptime(date_str[:10], "%Y-%m-%d")
                            data["dob"] = date_str[:10]
                        except:
                            pass
            
            # Look for nationality
            if "nationality" in line_lower and ":" in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    nat = parts[1].strip()
                    # Skip if it looks like another field
                    if not any(keyword in nat.lower() for keyword in ["gender", "male", "female", "date"]):
                        data["nationality"] = nat
            
            # Look for gender
            if "gender" in line_lower and ":" in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    gender_str = parts[1].strip().lower()
                    if "male" in gender_str:
                        data["gender"] = "M" if "female" not in gender_str else "F"
            elif ("male" in line_lower or "female" in line_lower) and data["gender"] is None:
                data["gender"] = "M" if line_lower.strip() == "male" else "F" if line_lower.strip() == "female" else data["gender"]
            
            # Look for emirate
            if "emirate" in line_lower and ":" in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    emirate_str = parts[1].strip()
                    # Skip if it looks like warning text
                    if "synthetic" not in emirate_str.lower() and "not a real" not in emirate_str.lower():
                        data["emirate"] = emirate_str
        
        logger.info(f"Extracted Emirates ID data: {list(data.keys())}")
        
        return data

    def is_available(self) -> bool:
        """Check if Tesseract is available on the system."""
        try:
            version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract version: {version}")
            return True
        except Exception as e:
            logger.warning(f"Tesseract not available: {e}")
            return False


# Global instance
_ocr_processor = None


def get_ocr_processor(force_reload: bool = False) -> OCRProcessor:
    """Get or create global OCR processor instance."""
    global _ocr_processor
    if _ocr_processor is None or force_reload:
        _ocr_processor = OCRProcessor()
        logger.info(f"OCR processor loaded - version {OCR_VERSION}")
    return _ocr_processor


if __name__ == "__main__":
    # Test OCR processor
    logging.basicConfig(level=logging.INFO)
    
    processor = OCRProcessor()
    
    if processor.is_available():
        print("✓ Tesseract OCR is available")
        print(f"  Command: {processor.tesseract_cmd}")
    else:
        print("✗ Tesseract OCR is not available")
        print("  Install from: https://github.com/UB-Mannheim/tesseract/wiki")

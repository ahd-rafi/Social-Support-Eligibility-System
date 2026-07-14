"""Generate synthetic applicant data for testing."""

import json
import random
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Configuration
NUM_PROFILES = 15  # 12-15 as per scope
OUTPUT_DIR = Path(__file__).parent / "output"

# UAE-specific data
EMIRATES = ["Abu Dhabi", "Dubai", "Sharjah", "Ajman", "Ras Al Khaimah", "Fujairah", "Umm Al Quwain"]
NATIONALITIES = ["UAE", "Egypt", "India", "Pakistan", "Philippines", "Bangladesh", "Sri Lanka", "Jordan"]
FIRST_NAMES_MALE = ["Ahmed", "Mohammed", "Ali", "Hassan", "Omar", "Khalid", "Saeed", "Abdullah", "Rashid", "Salem"]
FIRST_NAMES_FEMALE = ["Fatima", "Aisha", "Maryam", "Noura", "Layla", "Sara", "Hessa", "Shaikha", "Latifa", "Amna"]
LAST_NAMES = ["Al-Mansoori", "Al-Mazrouei", "Al-Shamsi", "Al-Dhaheri", "Al-Kaabi", "Khan", "Patel", "Ahmed", "Hassan", "Ali"]

# Employment sectors
SECTORS = {
    "hospitality": ["Grand Hyatt Abu Dhabi", "Marriott Hotel", "Hilton Hotel", "Sheraton Hotel", "Restaurant Group LLC"],
    "retail": ["Abu Dhabi Mall", "Carrefour", "Lulu Hypermarket", "Retail Services Co."],
    "healthcare": ["Abu Dhabi Health Services", "Sheikh Khalifa Hospital", "Cleveland Clinic", "Mediclinic"],
    "construction": ["Arabtec Construction", "ALEC Engineering", "Trojan General Contracting"],
    "admin": ["Abu Dhabi Services Co.", "Government Services Office", "Emirates Support Services"],
}

# Skills by sector
SKILLS = {
    "hospitality": ["customer service", "food service", "hotel operations", "housekeeping", "front desk"],
    "retail": ["sales", "inventory management", "customer service", "cash handling", "merchandising"],
    "healthcare": ["patient care", "medical terminology", "health & safety", "clinical support"],
    "construction": ["carpentry", "electrical work", "plumbing", "HVAC", "construction safety"],
    "admin": ["data entry", "Microsoft Office", "filing", "scheduling", "customer support"],
}


class SyntheticDataGenerator:
    """Generate synthetic applicant profiles with all required documents."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track generated data for consistency checks
        self.profiles = []

    def generate_profile(self, profile_id: int) -> Dict[str, Any]:
        """Generate a complete applicant profile."""
        
        # Basic demographics
        gender = random.choice(["M", "F"])
        first_name = random.choice(FIRST_NAMES_MALE if gender == "M" else FIRST_NAMES_FEMALE)
        last_name = random.choice(LAST_NAMES)
        name = f"{first_name} {last_name}"
        
        age = random.randint(25, 55)
        dob = datetime.now() - timedelta(days=age * 365)
        
        nationality = random.choice(NATIONALITIES)
        emirate = random.choice(EMIRATES)
        
        # Emirates ID
        id_number = f"784-{dob.year}-{random.randint(1000000, 9999999)}-{random.randint(1, 9)}"
        
        # Employment
        sector = random.choice(list(SECTORS.keys()))
        employer = random.choice(SECTORS[sector])
        employment_months = random.randint(6, 120)  # 6 months to 10 years
        employment_status = "employed" if employment_months > 0 else "unemployed"
        
        # For some profiles, make them recently unemployed
        if random.random() < 0.15:  # 15% unemployed
            employment_status = "unemployed"
            employment_months = 0
            previous_employer = employer
            employer = None
        
        # Income
        if employment_status == "employed":
            monthly_income = random.randint(3000, 25000)
        else:
            monthly_income = 0
        
        # Family
        household_size = random.randint(2, 7)
        num_dependents = household_size - 1  # Excluding applicant
        
        # Assets and liabilities
        has_car = random.random() < 0.6
        car_value = random.randint(15000, 80000) if has_car else 0
        savings = random.randint(0, 50000)
        
        has_property = random.random() < 0.15  # 15% own property
        property_value = random.randint(500000, 2000000) if has_property else 0
        
        # Liabilities
        car_loan = random.randint(10000, car_value - 5000) if has_car and random.random() < 0.7 else 0
        personal_loan = random.randint(5000, 30000) if random.random() < 0.4 else 0
        credit_card_debt = random.randint(2000, 15000) if random.random() < 0.5 else 0
        mortgage = random.randint(200000, 800000) if has_property and random.random() < 0.9 else 0
        
        total_assets = car_value + savings + property_value
        total_liabilities = car_loan + personal_loan + credit_card_debt + mortgage
        net_worth = total_assets - total_liabilities
        
        # Credit score
        if credit_card_debt > 10000 or personal_loan > 20000:
            credit_score = random.randint(300, 650)  # Lower for high debt
        else:
            credit_score = random.randint(550, 900)
        
        missed_payments = 0 if credit_score > 700 else random.randint(0, 5)
        
        # Eligibility calculation (simplified)
        income_per_family = monthly_income / household_size if monthly_income > 0 else 0
        debt_to_income = (total_liabilities / (monthly_income * 12)) if monthly_income > 0 else 0
        
        # Determine eligibility category
        if income_per_family < 2000 and net_worth < 50000:
            eligibility = "approved_financial"
        elif income_per_family < 4000 and employment_status == "employed":
            eligibility = "approved_enablement"
        else:
            eligibility = "soft_decline"
        
        profile = {
            "profile_id": f"applicant_{profile_id:03d}",
            "name": name,
            "gender": gender,
            "age": age,
            "dob": dob.strftime("%Y-%m-%d"),
            "nationality": nationality,
            "emirate": emirate,
            "emirates_id": id_number,
            "email": f"{first_name.lower()}.{last_name.lower()}@email.ae",
            "phone": f"+971-{random.randint(50, 56)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}",
            "sector": sector,
            "employer": employer,
            "employment_status": employment_status,
            "employment_months": employment_months,
            "monthly_income": monthly_income,
            "household_size": household_size,
            "num_dependents": num_dependents,
            "car_value": car_value,
            "savings": savings,
            "property_value": property_value,
            "total_assets": total_assets,
            "car_loan": car_loan,
            "personal_loan": personal_loan,
            "credit_card_debt": credit_card_debt,
            "mortgage": mortgage,
            "total_liabilities": total_liabilities,
            "net_worth": net_worth,
            "credit_score": credit_score,
            "missed_payments": missed_payments,
            "income_per_family": income_per_family,
            "debt_to_income": debt_to_income,
            "eligibility": eligibility,
            "skills": random.sample(SKILLS[sector], k=min(random.randint(3, 5), len(SKILLS[sector]))),
        }
        
        return profile

    def add_inconsistencies(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Add deliberate inconsistencies to some profiles for validation testing."""
        
        inconsistencies = []
        
        # 20% chance of address mismatch
        if random.random() < 0.20:
            profile["credit_report_emirate"] = random.choice([e for e in EMIRATES if e != profile["emirate"]])
            inconsistencies.append("address_mismatch")
        else:
            profile["credit_report_emirate"] = profile["emirate"]
        
        # 15% chance of income mismatch
        if random.random() < 0.15:
            variance = random.uniform(0.3, 0.6)  # 30-60% difference
            profile["bank_statement_income"] = int(profile["monthly_income"] * (1 + variance))
            inconsistencies.append("income_mismatch")
        else:
            profile["bank_statement_income"] = profile["monthly_income"]
        
        # 10% chance of employer mismatch
        if random.random() < 0.10 and profile["employer"]:
            # Slight name variation
            profile["bank_statement_employer"] = profile["employer"].replace("Abu Dhabi", "AD").replace("LLC", "Co.")
            inconsistencies.append("employer_mismatch")
        else:
            profile["bank_statement_employer"] = profile["employer"]
        
        # 10% chance of family size mismatch
        if random.random() < 0.10:
            profile["credit_report_dependents"] = profile["num_dependents"] + random.choice([-1, 1])
            inconsistencies.append("family_size_mismatch")
        else:
            profile["credit_report_dependents"] = profile["num_dependents"]
        
        profile["has_inconsistencies"] = len(inconsistencies) > 0
        profile["inconsistency_types"] = inconsistencies
        
        return profile

    def generate_emirates_id_image(self, profile: Dict[str, Any], output_path: Path):
        """Generate a synthetic Emirates ID image using Pillow."""
        
        # Create white background
        width, height = 800, 500
        img = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a font, fallback to default
        try:
            font_large = ImageFont.truetype("arial.ttf", 28)
            font_medium = ImageFont.truetype("arial.ttf", 20)
            font_small = ImageFont.truetype("arial.ttf", 16)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw borders
        draw.rectangle([10, 10, width-10, height-10], outline='black', width=3)
        
        # Title
        draw.text((width//2, 40), "EMIRATES ID (SYNTHETIC)", fill='black', font=font_large, anchor='mm')
        
        # Fields
        y_offset = 100
        fields = [
            ("Name (English):", profile["name"]),
            ("ID Number:", profile["emirates_id"]),
            ("Date of Birth:", profile["dob"]),
            ("Nationality:", profile["nationality"]),
            ("Gender:", "Male" if profile["gender"] == "M" else "Female"),
            ("Emirate:", profile["emirate"]),
        ]
        
        for label, value in fields:
            draw.text((50, y_offset), label, fill='black', font=font_medium)  # Changed to black and medium font
            draw.text((250, y_offset), str(value), fill='black', font=font_large)  # Larger font for values
            y_offset += 50  # More spacing
        
        # Warning text
        draw.text((width//2, height-30), "SYNTHETIC DATA - NOT A REAL ID", fill='red', font=font_small, anchor='mm')
        
        # Add slight rotation for realism (optional)
        if random.random() < 0.3:
            img = img.rotate(random.uniform(-1.5, 1.5), expand=False, fillcolor='white')
        
        img.save(output_path)
        logger.debug(f"Generated Emirates ID: {output_path.name}")

    def generate_all_profiles(self):
        """Generate all synthetic profiles and documents."""
        
        logger.info(f"Generating {NUM_PROFILES} synthetic applicant profiles...")
        
        for i in range(1, NUM_PROFILES + 1):
            profile = self.generate_profile(i)
            profile = self.add_inconsistencies(profile)
            
            # Create profile directory
            profile_dir = self.output_dir / profile["profile_id"]
            profile_dir.mkdir(exist_ok=True)
            
            # Generate Emirates ID image
            self.generate_emirates_id_image(
                profile,
                profile_dir / "emirates_id.png"
            )
            
            # Save profile metadata
            with open(profile_dir / "profile.json", "w") as f:
                json.dump(profile, f, indent=2)
            
            self.profiles.append(profile)
            
            logger.info(f"✓ Generated profile {i}/{NUM_PROFILES}: {profile['name']} ({profile['eligibility']})")
        
        # Save summary
        self.save_summary()
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Generated {NUM_PROFILES} complete profiles")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"{'='*60}\n")

    def save_summary(self):
        """Save a summary CSV of all profiles."""
        
        df = pd.DataFrame(self.profiles)
        summary_path = self.output_dir / "profiles_summary.csv"
        df.to_csv(summary_path, index=False)
        
        logger.info(f"Summary saved to: {summary_path}")
        
        # Print statistics
        logger.info("\n" + "="*60)
        logger.info("Profile Statistics:")
        logger.info("="*60)
        logger.info(f"Total profiles: {len(self.profiles)}")
        logger.info(f"Eligibility breakdown:")
        for category in ["approved_financial", "approved_enablement", "soft_decline"]:
            count = sum(1 for p in self.profiles if p["eligibility"] == category)
            pct = (count / len(self.profiles)) * 100
            logger.info(f"  - {category}: {count} ({pct:.1f}%)")
        
        logger.info(f"\nProfiles with inconsistencies:")
        inconsistent = sum(1 for p in self.profiles if p["has_inconsistencies"])
        logger.info(f"  Total: {inconsistent} ({(inconsistent/len(self.profiles))*100:.1f}%)")
        
        # Count by inconsistency type
        inconsistency_counts = {}
        for p in self.profiles:
            for inc_type in p.get("inconsistency_types", []):
                inconsistency_counts[inc_type] = inconsistency_counts.get(inc_type, 0) + 1
        
        if inconsistency_counts:
            logger.info(f"  By type:")
            for inc_type, count in inconsistency_counts.items():
                logger.info(f"    - {inc_type}: {count}")


def main():
    """Main entry point."""
    generator = SyntheticDataGenerator(OUTPUT_DIR)
    generator.generate_all_profiles()
    
    logger.info("\n✓ Synthetic data generation complete!")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review generated profiles in: {OUTPUT_DIR}")
    logger.info(f"  2. Run: python -m data.seed to populate databases")


if __name__ == "__main__":
    main()

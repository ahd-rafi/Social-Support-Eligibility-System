"""
Generate missing documents (PDF, Excel) from profile.json files
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
import random
from typing import Dict, Any, List

# PDF generation
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT

# Excel generation
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, PatternFill


class DocumentGenerator:
    """Generate bank statements, resumes, credit reports, and assets Excel"""
    
    def __init__(self, profile_dir: Path):
        self.profile_dir = profile_dir
        self.profile = self._load_profile()
    
    def _load_profile(self) -> Dict[str, Any]:
        """Load profile.json"""
        profile_path = self.profile_dir / "profile.json"
        with open(profile_path, 'r') as f:
            return json.load(f)
    
    def generate_all(self):
        """Generate all missing documents"""
        print(f"Generating documents for {self.profile['name']}...")
        self.generate_bank_statement()
        self.generate_resume()
        self.generate_credit_report()
        self.generate_assets_excel()
        print("✓ All documents generated")
    
    def generate_bank_statement(self):
        """Generate bank_statement.pdf"""
        output_path = self.profile_dir / "bank_statement.pdf"
        
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle('Title', parent=styles['Heading1'], 
                                     fontSize=16, alignment=TA_CENTER, spaceAfter=20)
        story.append(Paragraph("EMIRATES NATIONAL BANK", title_style))
        story.append(Paragraph("Bank Account Statement", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))
        
        # Account info
        account_num = f"AE{random.randint(1000000000, 9999999999)}"
        story.append(Paragraph(f"<b>Account Holder:</b> {self.profile['name']}", styles['Normal']))
        story.append(Paragraph(f"<b>Account Number:</b> {account_num}", styles['Normal']))
        story.append(Paragraph(f"<b>Period:</b> {datetime.now().strftime('%B %Y')}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Generate transactions
        transactions = self._generate_transactions()
        
        # Transaction table
        data = [["Date", "Description", "Debit", "Credit", "Balance"]]
        balance = random.randint(2000, 10000)
        
        for tx in transactions:
            balance += tx['amount']
            debit = f"-{abs(tx['amount']):.2f}" if tx['amount'] < 0 else ""
            credit = f"{tx['amount']:.2f}" if tx['amount'] > 0 else ""
            data.append([
                tx['date'].strftime('%Y-%m-%d'),
                tx['description'],
                debit,
                credit,
                f"{balance:.2f}"
            ])
        
        table = Table(data, colWidths=[1*inch, 3*inch, 1*inch, 1*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        story.append(table)
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(f"<b>Average Monthly Income:</b> AED {self.profile['monthly_income']:,.2f}", styles['Normal']))
        
        doc.build(story)
        print(f"  ✓ Bank statement: {output_path.name}")

    def _generate_transactions(self) -> List[Dict[str, Any]]:
        """Generate realistic bank transactions"""
        transactions = []
        today = datetime.now()
        
        # Salary deposit
        for i in range(3):
            salary_date = today - timedelta(days=30*i + 5)
            transactions.append({
                'date': salary_date,
                'description': f"Salary - {self.profile['employer']}",
                'amount': self.profile['monthly_income']
            })
        
        # Random expenses
        expenses = [
            ("Rent Payment", -3000),
            ("Utilities - DEWA", -450),
            ("Grocery Store", -800),
            ("Fuel Station", -250),
            ("Restaurant", -150),
            ("School Fees", -1200),
            ("Internet Bill", -300),
        ]
        
        for i in range(20):
            tx_date = today - timedelta(days=random.randint(1, 90))
            expense = random.choice(expenses)
            transactions.append({
                'date': tx_date,
                'description': expense[0],
                'amount': expense[1] + random.randint(-50, 50)
            })
        
        transactions.sort(key=lambda x: x['date'])
        return transactions
    
    def generate_resume(self):
        """Generate resume.pdf"""
        output_path = self.profile_dir / "resume.pdf"
        
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Header
        title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=20, 
                                     alignment=TA_CENTER, spaceAfter=10)
        story.append(Paragraph(self.profile['name'], title_style))
        story.append(Paragraph(f"{self.profile['email']} | {self.profile['phone']}", 
                              ParagraphStyle('Contact', parent=styles['Normal'], alignment=TA_CENTER)))
        story.append(Paragraph(f"{self.profile['emirate']}, UAE", 
                              ParagraphStyle('Location', parent=styles['Normal'], alignment=TA_CENTER)))
        story.append(Spacer(1, 0.3*inch))
        
        # Professional Summary
        story.append(Paragraph("<b>PROFESSIONAL SUMMARY</b>", styles['Heading2']))
        summary = f"Experienced {self.profile['sector']} professional with {self.profile['employment_months']//12} years of experience. "
        summary += "Strong track record in " + ", ".join(self.profile['skills'][:3]) + "."
        story.append(Paragraph(summary, styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Work Experience
        story.append(Paragraph("<b>WORK EXPERIENCE</b>", styles['Heading2']))
        work_years = self.profile['employment_months'] // 12
        start_year = datetime.now().year - work_years
        story.append(Paragraph(f"<b>{self.profile['employer']}</b>", styles['Normal']))
        story.append(Paragraph(f"<i>{self.profile['sector'].title()} Specialist | {start_year} - Present</i>", styles['Normal']))
        story.append(Paragraph(f"• Monthly Salary: AED {self.profile['monthly_income']:,.0f}", styles['Normal']))
        story.append(Paragraph(f"• Key Skills: {', '.join(self.profile['skills'])}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Education
        story.append(Paragraph("<b>EDUCATION</b>", styles['Heading2']))
        story.append(Paragraph("<b>Bachelor's Degree</b>", styles['Normal']))
        story.append(Paragraph(f"University, {self.profile['nationality']}", styles['Normal']))
        story.append(Spacer(1, 0.2*inch))
        
        # Skills
        story.append(Paragraph("<b>SKILLS</b>", styles['Heading2']))
        for skill in self.profile['skills']:
            story.append(Paragraph(f"• {skill.title()}", styles['Normal']))
        
        doc.build(story)
        print(f"  ✓ Resume: {output_path.name}")
    
    def generate_credit_report(self):
        """Generate credit_report.pdf"""
        output_path = self.profile_dir / "credit_report.pdf"
        
        doc = SimpleDocTemplate(str(output_path), pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=18, alignment=TA_CENTER)
        story.append(Paragraph("AL ETIHAD CREDIT BUREAU", title_style))
        story.append(Paragraph("Credit Report", styles['Heading2']))
        story.append(Spacer(1, 0.3*inch))
        
        # Personal Info
        story.append(Paragraph(f"<b>Name:</b> {self.profile['name']}", styles['Normal']))
        story.append(Paragraph(f"<b>Emirates ID:</b> {self.profile['emirates_id']}", styles['Normal']))
        story.append(Paragraph(f"<b>Date of Birth:</b> {self.profile['dob']}", styles['Normal']))
        story.append(Paragraph(f"<b>Emirate:</b> {self.profile['credit_report_emirate']}", styles['Normal']))
        story.append(Paragraph(f"<b>Report Date:</b> {datetime.now().strftime('%Y-%m-%d')}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Credit Score
        story.append(Paragraph("<b>CREDIT SCORE</b>", styles['Heading2']))
        story.append(Paragraph(f"<font size=24 color='blue'>{self.profile['credit_score']}</font>", styles['Normal']))
        story.append(Paragraph(f"Rating: {'Poor' if self.profile['credit_score'] < 600 else 'Fair' if self.profile['credit_score'] < 700 else 'Good'}", 
                              styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Account Summary
        story.append(Paragraph("<b>ACCOUNT SUMMARY</b>", styles['Heading2']))
        story.append(Paragraph(f"Total Outstanding Debt: AED {self.profile['total_liabilities']:,.2f}", styles['Normal']))
        story.append(Paragraph(f"Number of Missed Payments (Last 12 months): {self.profile['missed_payments']}", styles['Normal']))
        story.append(Paragraph(f"Active Credit Cards: 2", styles['Normal']))
        story.append(Paragraph(f"Active Loans: {1 if self.profile['personal_loan'] > 0 else 0}", styles['Normal']))
        story.append(Spacer(1, 0.3*inch))
        
        # Account Details
        story.append(Paragraph("<b>ACCOUNT DETAILS</b>", styles['Heading2']))
        accounts = []
        if self.profile['credit_card_debt'] > 0:
            story.append(Paragraph(f"• <b>Credit Card</b> - Balance: AED {self.profile['credit_card_debt']:,.2f}", styles['Normal']))
        if self.profile['personal_loan'] > 0:
            story.append(Paragraph(f"• <b>Personal Loan</b> - Outstanding: AED {self.profile['personal_loan']:,.2f}", styles['Normal']))
        if self.profile['car_loan'] > 0:
            story.append(Paragraph(f"• <b>Car Loan</b> - Outstanding: AED {self.profile['car_loan']:,.2f}", styles['Normal']))
        if self.profile['mortgage'] > 0:
            story.append(Paragraph(f"• <b>Mortgage</b> - Outstanding: AED {self.profile['mortgage']:,.2f}", styles['Normal']))
        
        story.append(Spacer(1, 0.3*inch))
        story.append(Paragraph(f"<b>Household Dependents:</b> {self.profile['credit_report_dependents']}", styles['Normal']))
        
        doc.build(story)
        print(f"  ✓ Credit report: {output_path.name}")

    def generate_assets_excel(self):
        """Generate assets_liabilities.xlsx"""
        output_path = self.profile_dir / "assets_liabilities.xlsx"
        
        wb = Workbook()
        
        # Assets sheet
        ws_assets = wb.active
        ws_assets.title = "Assets"
        
        # Header
        ws_assets['A1'] = "Asset Type"
        ws_assets['B1'] = "Description"
        ws_assets['C1'] = "Value (AED)"
        
        header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)
        
        for cell in ['A1', 'B1', 'C1']:
            ws_assets[cell].fill = header_fill
            ws_assets[cell].font = header_font
            ws_assets[cell].alignment = Alignment(horizontal='center')
        
        # Assets data
        row = 2
        if self.profile['property_value'] > 0:
            ws_assets[f'A{row}'] = "Property"
            ws_assets[f'B{row}'] = "Residential Property"
            ws_assets[f'C{row}'] = self.profile['property_value']
            row += 1
        
        if self.profile['car_value'] > 0:
            ws_assets[f'A{row}'] = "Vehicle"
            ws_assets[f'B{row}'] = "Personal Vehicle"
            ws_assets[f'C{row}'] = self.profile['car_value']
            row += 1
        
        if self.profile['savings'] > 0:
            ws_assets[f'A{row}'] = "Savings"
            ws_assets[f'B{row}'] = "Bank Savings Account"
            ws_assets[f'C{row}'] = self.profile['savings']
            row += 1
        
        # Total
        ws_assets[f'A{row}'] = "TOTAL ASSETS"
        ws_assets[f'A{row}'].font = Font(bold=True)
        ws_assets[f'C{row}'] = self.profile['total_assets']
        ws_assets[f'C{row}'].font = Font(bold=True)
        
        # Liabilities sheet
        ws_liab = wb.create_sheet("Liabilities")
        
        # Header
        ws_liab['A1'] = "Liability Type"
        ws_liab['B1'] = "Description"
        ws_liab['C1'] = "Amount (AED)"
        
        for cell in ['A1', 'B1', 'C1']:
            ws_liab[cell].fill = header_fill
            ws_liab[cell].font = header_font
            ws_liab[cell].alignment = Alignment(horizontal='center')
        
        # Liabilities data
        row = 2
        if self.profile['mortgage'] > 0:
            ws_liab[f'A{row}'] = "Mortgage"
            ws_liab[f'B{row}'] = "Home Loan"
            ws_liab[f'C{row}'] = self.profile['mortgage']
            row += 1
        
        if self.profile['personal_loan'] > 0:
            ws_liab[f'A{row}'] = "Personal Loan"
            ws_liab[f'B{row}'] = "Personal Loan"
            ws_liab[f'C{row}'] = self.profile['personal_loan']
            row += 1
        
        if self.profile['car_loan'] > 0:
            ws_liab[f'A{row}'] = "Car Loan"
            ws_liab[f'B{row}'] = "Auto Finance"
            ws_liab[f'C{row}'] = self.profile['car_loan']
            row += 1
        
        if self.profile['credit_card_debt'] > 0:
            ws_liab[f'A{row}'] = "Credit Card"
            ws_liab[f'B{row}'] = "Credit Card Balance"
            ws_liab[f'C{row}'] = self.profile['credit_card_debt']
            row += 1
        
        # Total
        ws_liab[f'A{row}'] = "TOTAL LIABILITIES"
        ws_liab[f'A{row}'].font = Font(bold=True)
        ws_liab[f'C{row}'] = self.profile['total_liabilities']
        ws_liab[f'C{row}'].font = Font(bold=True)
        
        # Summary sheet
        ws_summary = wb.create_sheet("Summary")
        ws_summary['A1'] = "Financial Summary"
        ws_summary['A1'].font = Font(size=14, bold=True)
        
        ws_summary['A3'] = "Total Assets"
        ws_summary['B3'] = self.profile['total_assets']
        ws_summary['B3'].font = Font(color="008000")
        
        ws_summary['A4'] = "Total Liabilities"
        ws_summary['B4'] = self.profile['total_liabilities']
        ws_summary['B4'].font = Font(color="FF0000")
        
        ws_summary['A5'] = "Net Worth"
        ws_summary['B5'] = self.profile['net_worth']
        ws_summary['B5'].font = Font(bold=True)
        
        # Adjust column widths
        for ws in [ws_assets, ws_liab, ws_summary]:
            ws.column_dimensions['A'].width = 20
            ws.column_dimensions['B'].width = 30
            ws.column_dimensions['C'].width = 15
        
        wb.save(output_path)
        print(f"  ✓ Assets/Liabilities Excel: {output_path.name}")


def main():
    """Generate documents for all applicants"""
    output_dir = Path(__file__).parent / "output"
    
    if not output_dir.exists():
        print(f"Error: Output directory not found: {output_dir}")
        return
    
    # Get all applicant directories
    applicant_dirs = sorted([d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith('applicant_')])
    
    if not applicant_dirs:
        print("No applicant directories found!")
        return
    
    print(f"Found {len(applicant_dirs)} applicant profiles")
    print("Generating missing documents...\n")
    
    for i, applicant_dir in enumerate(applicant_dirs, 1):
        print(f"[{i}/{len(applicant_dirs)}] {applicant_dir.name}")
        try:
            generator = DocumentGenerator(applicant_dir)
            generator.generate_all()
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\n✓ Document generation complete!")
    print(f"Generated documents for {len(applicant_dirs)} applicants")


if __name__ == "__main__":
    main()

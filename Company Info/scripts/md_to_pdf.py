import os
import glob
from fpdf import FPDF

# Path to your docs folder
docs_path = "/home/swati23cseds006/Gen AI/Ask-HR/Company Info/docs"

# Find all .md files
md_files = glob.glob(os.path.join(docs_path, "*.md"))

for md_file in md_files:
    # Read the markdown file
    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Create PDF
    pdf = FPDF()
    pdf.add_page()

    # Use a Unicode font (DejaVuSans)
    pdf.add_font("DejaVu", "", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", uni=True)
    pdf.set_font("DejaVu", size=12)

    # Add content line by line
    for line in content.splitlines():
        pdf.multi_cell(0, 10, line)

    # Save PDF with same name
    pdf_file = md_file.replace(".md", ".pdf")
    pdf.output(pdf_file)

    # Delete old .md file
    os.remove(md_file)

    print(f"Converted and deleted: {md_file} -> {pdf_file}")

print("✅ Conversion complete! All .md files converted to PDF and originals deleted.")

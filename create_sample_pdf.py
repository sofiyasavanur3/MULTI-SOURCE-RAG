from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Create a simple PDF
c = canvas.Canvas("data/pdfs/company_report.pdf", pagesize=letter)

# Add title
c.setFont("Helvetica-Bold", 16)
c.drawString(100, 750, "TechCorp Annual Report 2024")

# Add content
c.setFont("Helvetica", 12)
y_position = 700

content = [
    "",
    "Company Overview:",
    "TechCorp is an AI company founded in 2020.",
    "We have 3 main products:",
    "  1. AI Assistant Platform",
    "  2. Data Analytics Suite",
    "  3. Cloud Infrastructure",
    "",
    "Team: 150 employees across 5 countries",
    "Revenue 2024: $10 million",
    "Growth: 200% YoY",
    "",
    "Leadership:",
    "CEO: Priya Sharma",
    "CTO: Raj Kumar"
]

for line in content:
    c.drawString(100, y_position, line)
    y_position -= 20

c.save()
print("✅ PDF created: data/pdfs/company_report.pdf")
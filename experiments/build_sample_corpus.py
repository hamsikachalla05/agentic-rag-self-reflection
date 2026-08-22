#!/usr/bin/env python3
"""Create small synthetic annual-report PDFs for the end-to-end prototype.

These documents are original research fixtures, not copies of real 10-Ks.
Facts are unique per page so retrieval-to-page alignment can be checked.
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "reports"


def _styles():
    styles = getSampleStyleSheet()
    return styles


def build_northstar(path: Path) -> None:
    styles = _styles()
    doc = SimpleDocTemplate(str(path), pagesize=LETTER, title="Northstar Financial Inc. 2023 Annual Report")
    story = []

    story.append(Paragraph("Northstar Financial Inc.", styles["Title"]))
    story.append(Paragraph("Annual Report for the fiscal year ended December 31, 2023", styles["Heading2"]))
    story.append(Paragraph("Commission file number 001-88001", styles["Normal"]))
    story.append(Paragraph(
        "This report is a prototype fixture used to test retrieval over long financial documents. "
        "Northstar Financial Inc. is a diversified financial services holding company.",
        styles["Normal"],
    ))
    story.append(PageBreak())

    story.append(Paragraph("Item 1. Business Overview", styles["Heading1"]))
    story.append(Paragraph(
        "Northstar Financial Inc. operates three reportable segments: Wealth Management, "
        "Commercial Lending, and Insurance Brokerage. The Wealth Management segment provides "
        "advisory and custody services to high-net-worth clients. Commercial Lending originates "
        "senior secured loans to middle-market companies. Insurance Brokerage places property "
        "and casualty coverage for corporate clients. The company was incorporated in Delaware in 1998.",
        styles["Normal"],
    ))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "Principal executive offices are located at 400 Harbor Plaza, Stamford, Connecticut 06902. "
        "The common stock trades on the NYSE under the ticker NSFI.",
        styles["Normal"],
    ))
    story.append(PageBreak())

    story.append(Paragraph("Item 6. Selected Financial Data", styles["Heading1"]))
    story.append(Paragraph(
        "The following table presents selected consolidated financial data for Northstar Financial Inc. "
        "for the year ended December 31, 2023. Total revenues were $12.40 billion. Net income attributable "
        "to common shareholders was $1.85 billion.",
        styles["Normal"],
    ))
    table = Table(
        [
            ["Metric", "2023", "2022"],
            ["Total revenues ($ millions)", "12,400", "11,150"],
            ["Wealth Management revenue", "6,210", "5,740"],
            ["Commercial Lending revenue", "4,080", "3,690"],
            ["Insurance Brokerage revenue", "2,110", "1,720"],
            ["Net income", "1,850", "1,420"],
            ["Diluted EPS ($)", "4.12", "3.18"],
            ["Effective tax rate", "21.4%", "22.1%"],
        ],
        colWidths=[3.2 * inch, 1.5 * inch, 1.5 * inch],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ]
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(table)
    story.append(PageBreak())

    story.append(Paragraph("Item 1A. Risk Factors", styles["Heading1"]))
    story.append(Paragraph(
        "Credit losses in the Commercial Lending segment could increase if middle-market default rates rise. "
        "A 100 basis-point increase in credit loss provisions would have reduced 2023 pre-tax income by "
        "approximately $210 million. Market declines of 15% in equity values would reduce Wealth Management "
        "fee revenue by an estimated $430 million. Cybersecurity incidents remain a material operational risk.",
        styles["Normal"],
    ))
    story.append(PageBreak())

    story.append(Paragraph("Item 7. Management's Discussion and Analysis", styles["Heading1"]))
    story.append(Paragraph(
        "Wealth Management was the largest contributor to 2023 revenue, generating $6.21 billion, or 50.1% "
        "of total revenues. Assets under management ended the year at $412 billion, up 9% year over year. "
        "The Board of Directors declared a cash dividend of $1.48 per common share in 2023, paid quarterly "
        "at $0.37 per share. Share repurchases totaled $640 million.",
        styles["Normal"],
    ))
    story.append(PageBreak())

    story.append(Paragraph("Item 1. Employees and Properties", styles["Heading1"]))
    story.append(Paragraph(
        "As of December 31, 2023, Northstar Financial Inc. employed 18,420 people, of whom 11,200 were in "
        "the United States. The company leased 2.4 million square feet of office space. No single office "
        "other than the Stamford headquarters accounted for more than 10% of total square footage.",
        styles["Normal"],
    ))
    story.append(PageBreak())

    story.append(Paragraph("Item 9B. Other Information", styles["Heading1"]))
    story.append(Paragraph(
        "On January 12, 2024, subsequent to year end, Northstar announced an agreement to acquire Harbor "
        "Trust Partners for $780 million in cash. The transaction is expected to close in the third quarter "
        "of 2024, subject to regulatory approvals. Harbor Trust managed $28 billion of client assets as of "
        "December 31, 2023.",
        styles["Normal"],
    ))
    doc.build(story)


def build_cedartech(path: Path) -> None:
    styles = _styles()
    doc = SimpleDocTemplate(str(path), pagesize=LETTER, title="CedarTech PLC 2022 Annual Report")
    story = []

    story.append(Paragraph("CedarTech PLC", styles["Title"]))
    story.append(Paragraph("Annual Report and Accounts 2022", styles["Heading2"]))
    story.append(Paragraph(
        "CedarTech PLC is a United Kingdom-incorporated industrial software company. This prototype "
        "annual report is used for retrieval evaluation. Registered office: 17 King William Street, "
        "London EC4N 7BL.",
        styles["Normal"],
    ))
    story.append(PageBreak())

    story.append(Paragraph("Strategic Report", styles["Heading1"]))
    story.append(Paragraph(
        "CedarTech sells plant-operations software and related services. The Chief Executive Officer is "
        "Priya Raman. The Chief Financial Officer is Henrik Olsen. Revenue is recognized over the contract "
        "term for subscription licenses and at a point in time for perpetual licenses.",
        styles["Normal"],
    ))
    story.append(PageBreak())

    story.append(Paragraph("Financial Highlights", styles["Heading1"]))
    story.append(Paragraph(
        "For the year ended 31 December 2022, CedarTech reported revenue of £2.76 billion and operating "
        "income of £414 million. Free cash flow was £301 million. Recurring subscription revenue was 81% "
        "of total revenue.",
        styles["Normal"],
    ))
    table = Table(
        [
            ["Metric", "2022", "2021"],
            ["Revenue (£ millions)", "2,760", "2,410"],
            ["Operating income", "414", "355"],
            ["Net income", "328", "271"],
            ["Recurring revenue %", "81%", "77%"],
        ],
        colWidths=[3.0 * inch, 1.6 * inch, 1.6 * inch],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b6e4f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ]
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(table)
    story.append(PageBreak())

    story.append(Paragraph("People", styles["Heading1"]))
    story.append(Paragraph(
        "CedarTech employed 9,860 people at 31 December 2022. Research and development headcount was 3,140. "
        "The company opened a new engineering centre in Manchester during the year.",
        styles["Normal"],
    ))
    doc.build(story)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    northstar = OUT_DIR / "Northstar-Financial-Inc_2023_annual-report.pdf"
    cedar = OUT_DIR / "CedarTech-PLC_2022_annual-report.pdf"
    build_northstar(northstar)
    build_cedartech(cedar)
    print(f"Wrote {northstar}")
    print(f"Wrote {cedar}")


if __name__ == "__main__":
    main()

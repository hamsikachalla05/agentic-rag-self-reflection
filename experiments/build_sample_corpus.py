#!/usr/bin/env python3
"""Create synthetic annual-report PDFs with one topic per page.

Pages are filled to a realistic 10-K density (~400-500 tokens) so 512-token
chunks map onto source pages instead of swallowing the whole filing. Facts
used by the prototype questions stay on fixed page numbers.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas


OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "reports"
WIDTH, HEIGHT = LETTER
LEFT = 72
RIGHT = WIDTH - 72
TOP = HEIGHT - 64
LINE = 12


def wrap_line(text: str, width: int = 98) -> list[str]:
    return textwrap.wrap(text, width=width) or [""]


def filler(topic: str, n: int) -> list[str]:
    templates = [
        f"Management continues to monitor {topic} in the ordinary course of business and reports material changes to the board.",
        f"Internal controls related to {topic} are reviewed annually by the audit committee and by independent registered public accountants.",
        f"Forward-looking statements concerning {topic} are subject to risks and uncertainties described in Item 1A.",
        f"The company believes that disclosure about {topic} is material to an understanding of results of operations and financial condition.",
        f"Comparability of {topic} may be affected by acquisitions, divestitures, foreign-currency translation, and changes in accounting estimates.",
        f"Unless otherwise indicated, figures discussing {topic} are presented in millions of local reporting currency.",
        f"Non-GAAP measures that relate to {topic} are reconciled to the nearest GAAP measure in the financial statements.",
        f"Seasonality has not historically caused material variation in {topic} beyond the ranges discussed in this report.",
        f"Regulation, litigation, and competitive dynamics can affect {topic} and may differ from management's current expectations.",
        f"Employees, vendors, and clients associated with {topic} are subject to the company's code of conduct and risk policies.",
    ]
    out = []
    for i in range(n):
        out.append(templates[i % len(templates)] + f" Additional narrative item {i+1} for {topic}.")
    return out


def draw_page(c: canvas.Canvas, heading: str, paragraphs: list[str]) -> None:
    y = TOP
    c.setFont("Times-Bold", 13)
    for line in wrap_line(heading, 90):
        c.drawString(LEFT, y, line)
        y -= 16
    y -= 6
    c.setFont("Times-Roman", 10)
    for para in paragraphs:
        for line in wrap_line(para):
            if y < 64:
                break
            c.drawString(LEFT, y, line)
            y -= LINE
        else:
            y -= 6
            continue
        break
    c.showPage()


def build_northstar(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=LETTER)
    c.setTitle("Northstar Financial Inc. 2023 Annual Report")

    draw_page(
        c,
        "Northstar Financial Inc. — Annual Report for the year ended December 31, 2023",
        [
            "Commission file number 001-88001. Common stock trades on the NYSE under the ticker NSFI.",
            "This document is an original research fixture used to evaluate retrieval over long financial reports. It is not a copy of a real SEC filing.",
            "Northstar Financial Inc. is a diversified financial services holding company incorporated in Delaware in 1998.",
            *filler("the cover page and corporate profile", 6),
        ],
    )
    draw_page(
        c,
        "Item 1. Business Overview",
        [
            "Northstar Financial Inc. operates three reportable segments: Wealth Management, Commercial Lending, and Insurance Brokerage. Principal executive offices are located at 400 Harbor Plaza, Stamford, Connecticut 06902.",
            "The Wealth Management segment provides advisory and custody services to high-net-worth clients. Commercial Lending originates senior secured loans to middle-market companies. Insurance Brokerage places property and casualty coverage for corporate clients.",
            *filler("business overview and segment operations", 6),
        ],
    )
    draw_page(
        c,
        "Item 6. Selected Financial Data",
        [
            "The following selected consolidated financial data is for Northstar Financial Inc. for the year ended December 31, 2023.",
            "Total revenues were $12.40 billion in 2023, compared with $11.15 billion in 2022.",
            "Net income attributable to common shareholders was $1.85 billion. Diluted earnings per share were $4.12.",
            "Segment revenues: Wealth Management $6.21 billion; Commercial Lending $4.08 billion; Insurance Brokerage $2.11 billion.",
            "The effective tax rate was 21.4% in 2023, compared with 22.1% in 2022.",
            "[TABLE] Metric | 2023 | 2022",
            "Total revenues ($ millions) | 12,400 | 11,150",
            "Net income | 1,850 | 1,420",
            "Diluted EPS ($) | 4.12 | 3.18",
            "Effective tax rate | 21.4% | 22.1% [/TABLE]",
            *filler("selected financial data and results of operations", 5),
        ],
    )
    draw_page(
        c,
        "Item 1A. Risk Factors",
        [
            "Credit losses in the Commercial Lending segment could increase if middle-market default rates rise.",
            "A 100 basis-point increase in credit loss provisions would have reduced 2023 pre-tax income by approximately $210 million.",
            "Market declines of 15% in equity values would reduce Wealth Management fee revenue by an estimated $430 million.",
            "Cybersecurity incidents remain a material operational risk for client data and transaction processing.",
            *filler("risk factors and market, credit, and operational risk", 6),
        ],
    )
    draw_page(
        c,
        "Item 7. Management's Discussion and Analysis",
        [
            "Wealth Management was the largest contributor to 2023 revenue, generating $6.21 billion, or 50.1% of total revenues.",
            "Assets under management ended the year at $412 billion, up 9% year over year.",
            "The Board of Directors declared a cash dividend of $1.48 per common share in 2023, paid quarterly at $0.37 per share.",
            "Share repurchases totaled $640 million during 2023.",
            *filler("management discussion and analysis of financial condition", 6),
        ],
    )
    draw_page(
        c,
        "Item 1. Employees and Properties",
        [
            "As of December 31, 2023, Northstar Financial Inc. employed 18,420 people, of whom 11,200 were in the United States.",
            "The company leased 2.4 million square feet of office space. No single office other than the Stamford headquarters accounted for more than 10% of total square footage.",
            *filler("human capital, properties, and leased facilities", 6),
        ],
    )
    draw_page(
        c,
        "Item 9B. Other Information — Subsequent Events",
        [
            "On January 12, 2024, subsequent to year end, Northstar announced an agreement to acquire Harbor Trust Partners for $780 million in cash.",
            "The transaction is expected to close in the third quarter of 2024, subject to regulatory approvals.",
            "Harbor Trust managed $28 billion of client assets as of December 31, 2023.",
            *filler("subsequent events and pending acquisitions", 6),
        ],
    )
    c.save()


def build_cedartech(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=LETTER)
    c.setTitle("CedarTech PLC 2022 Annual Report")

    draw_page(
        c,
        "CedarTech PLC — Annual Report and Accounts 2022",
        [
            "CedarTech PLC is a United Kingdom-incorporated industrial software company. This prototype annual report is an original research fixture.",
            "Registered office: 17 King William Street, London EC4N 7BL.",
            *filler("the group overview and statutory information", 6),
        ],
    )
    draw_page(
        c,
        "Strategic Report",
        [
            "CedarTech sells plant-operations software and related services.",
            "The Chief Executive Officer is Priya Raman. The Chief Financial Officer is Henrik Olsen.",
            "Revenue is recognized over the contract term for subscription licenses and at a point in time for perpetual licenses.",
            *filler("strategy, software subscriptions, and leadership", 6),
        ],
    )
    draw_page(
        c,
        "Financial Highlights",
        [
            "For the year ended 31 December 2022, CedarTech reported revenue of £2.76 billion and operating income of £414 million.",
            "Free cash flow was £301 million. Recurring subscription revenue was 81% of total revenue.",
            "[TABLE] Metric | 2022 | 2021",
            "Revenue (£ millions) | 2,760 | 2,410",
            "Operating income | 414 | 355",
            "Net income | 328 | 271",
            "Recurring revenue % | 81% | 77% [/TABLE]",
            *filler("financial highlights and recurring revenue", 6),
        ],
    )
    draw_page(
        c,
        "People",
        [
            "CedarTech employed 9,860 people at 31 December 2022. Research and development headcount was 3,140.",
            "The company opened a new engineering centre in Manchester during the year.",
            *filler("people, culture, and engineering centres", 6),
        ],
    )
    c.save()


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

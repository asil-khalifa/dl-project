"""Build the final report PDF: institute cover page + the exported report body.

Usage:  python report/build_pdf.py <exported_report.pdf>
Writes report/Team3_CSPE72_Report_Automatic_Question_Generation.pdf
The cover page reproduces the layout of the team's earlier course-report cover (A4, Arial).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pymupdf as fitz

HERE = Path(__file__).resolve().parent
OUT = HERE / "Team3_CSPE72_Report_Automatic_Question_Generation.pdf"
LOGO = HERE / "cover_logo.png"

PAGE_W, PAGE_H = 595.3, 841.9          # A4 in points
LEFT = RIGHT = 59.0                    # 1181 twips
TOP = 55.5                             # 1109 twips
TEXT_W = PAGE_W - LEFT - RIGHT


def centered(page, y, text, size, bold=False, color=(0, 0, 0)):
    """Insert a centred, wrapped text block; return the y below it."""
    font = "hebo" if bold else "helv"
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if fitz.get_text_length(trial, fontname=font, fontsize=size) <= TEXT_W:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    for line in lines:
        w = fitz.get_text_length(line, fontname=font, fontsize=size)
        page.insert_text(((PAGE_W - w) / 2, y + size), line, fontname=font, fontsize=size, color=color)
        y += size * 1.3
    return y


def build_cover() -> fitz.Document:
    doc = fitz.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    # institute logo: 6.5 in x 3.19 in anchored at the top-left of the text area
    logo_w, logo_h = 468.0, 230.0
    page.insert_image(fitz.Rect(LEFT - 3.7, TOP + 9, LEFT - 3.7 + logo_w, TOP + 9 + logo_h), filename=str(LOGO))
    y = TOP + 9 + logo_h + 28
    y = centered(page, y, "CSPE 72", 31) + 12
    y = centered(page, y, "Deep Learning Techniques", 24, bold=True) + 56
    y = centered(page, y, "Automatic Question Generation from Study Material", 25, bold=True) + 14
    y = centered(page, y, "PROJECT REPORT", 23) + 70
    names = ["Asil Khalifa (106123015)", "Lankireddy Hitesh Reddy (106123073)", "P Kamal Nag (106123093)"]
    for i, line in enumerate(names):
        page.insert_text((LEFT, y + 16), line, fontname="hebo", fontsize=16)
        y += 16 * 1.5
    return doc


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    body = fitz.open(sys.argv[1])
    final = build_cover()
    final.insert_pdf(body)
    final.save(OUT, garbage=3, deflate=True)
    print(f"wrote {OUT} ({len(final)} pages)")


if __name__ == "__main__":
    main()

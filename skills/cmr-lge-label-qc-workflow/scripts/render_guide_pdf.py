#!/usr/bin/env python3
"""Render the workflow HTML guide to PDF with the local headless office tool."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", type=Path, required=True)
    ap.add_argument("--pdf", type=Path, required=True)
    args = ap.parse_args()
    libreoffice = shutil.which("libreoffice")
    if not libreoffice:
        raise SystemExit("libreoffice is required to render the guide PDF")
    args.pdf.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [libreoffice, "--headless", "--convert-to", "pdf", "--outdir", str(args.pdf.parent), str(args.html)],
        capture_output=True,
        text=True,
    )
    generated = args.pdf.parent / (args.html.stem + ".pdf")
    if result.returncode or not generated.exists():
        _render_with_reportlab(args.html, args.pdf)
        generated = args.pdf
    if generated != args.pdf:
        generated.replace(args.pdf)
    print(args.pdf)


class _GuideParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.items = []
        self.tag = None
        self.buf = []

    def handle_starttag(self, tag, attrs):
        if tag in {"h1", "h2", "h3", "p", "li", "pre"}:
            self.tag, self.buf = tag, []
        elif tag == "tr":
            self.items.append(("tr", []))
        elif tag in {"td", "th"} and self.items and self.items[-1][0] == "tr":
            self.tag, self.buf = tag, []

    def handle_endtag(self, tag):
        if tag in {"h1", "h2", "h3", "p", "li", "pre"} and self.tag == tag:
            text = " ".join("".join(self.buf).split())
            if text:
                self.items.append((tag, text))
            self.tag, self.buf = None, []
        elif tag in {"td", "th"} and self.tag == tag and self.items and self.items[-1][0] == "tr":
            self.items[-1][1].append(" ".join("".join(self.buf).split()))
            self.tag, self.buf = None, []

    def handle_data(self, data):
        if self.tag:
            self.buf.append(data)


def _render_with_reportlab(html_path: Path, pdf_path: Path) -> None:
    # Debian's bundled ReportLab is old and still imports removed Python 3.10
    # base64 aliases.  Provide compatibility aliases without modifying Python.
    import base64
    import collections
    import collections.abc
    if not hasattr(base64, "decodestring"):
        base64.decodestring = base64.decodebytes
    if not hasattr(base64, "encodestring"):
        base64.encodestring = base64.encodebytes
    for name in ("Mapping", "MutableMapping", "Sequence"):
        if not hasattr(collections, name):
            setattr(collections, name, getattr(collections.abc, name))
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, Preformatted

    # STSong-Light is a built-in CID font and avoids platform-specific TTC
    # parsing problems while preserving Chinese text in the generated PDF.
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    parser = _GuideParser()
    parser.feed(html_path.read_text(encoding="utf-8"))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="CNBody", fontName="STSong-Light", fontSize=8.5, leading=13, spaceAfter=5))
    styles.add(ParagraphStyle(name="CNH1", fontName="STSong-Light", fontSize=16, leading=22, alignment=TA_CENTER, spaceAfter=10))
    styles.add(ParagraphStyle(name="CNH2", fontName="STSong-Light", fontSize=12, leading=17, textColor=colors.HexColor("#1f5f9e"), spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle(name="CNH3", fontName="STSong-Light", fontSize=10, leading=14, spaceBefore=5, spaceAfter=3))
    styles.add(ParagraphStyle(name="CNPre", fontName="STSong-Light", fontSize=7, leading=9, backColor=colors.HexColor("#f1f5f9"), borderPadding=5, spaceAfter=6))
    story = []
    for kind, value in parser.items:
        if kind == "tr":
            continue
        if kind in {"h1", "h2", "h3"}:
            story.append(Paragraph(value, styles["CN" + kind.upper()]))
        elif kind == "pre":
            story.append(Preformatted(value, styles["CNPre"]))
        elif kind == "li":
            story.append(Paragraph("• " + value, styles["CNBody"]))
        else:
            story.append(Paragraph(value, styles["CNBody"]))
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=16 * mm, leftMargin=16 * mm, topMargin=15 * mm, bottomMargin=15 * mm)
    doc.build(story)


if __name__ == "__main__":
    main()

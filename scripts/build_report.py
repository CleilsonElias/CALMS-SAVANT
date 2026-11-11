"""Render the technical review to PDF. Optional: reportlab and Open Sans fonts."""
import html
from pathlib import Path
import re
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether


def main():
    root = Path(__file__).resolve().parents[1]
    fontdir = Path("/usr/share/fonts/truetype/open-sans")
    regular, bold, italic = "Helvetica", "Helvetica-Bold", "Helvetica-Oblique"
    if (fontdir/"OpenSans-Regular.ttf").exists():
        for name, filename in (("Review", "OpenSans-Regular.ttf"),
                               ("ReviewBold", "OpenSans-Bold.ttf"),
                               ("ReviewItalic", "OpenSans-Italic.ttf")):
            pdfmetrics.registerFont(TTFont(name, str(fontdir/filename)))
        pdfmetrics.registerFontFamily("Review", normal="Review", bold="ReviewBold", italic="ReviewItalic", boldItalic="ReviewBold")
        regular, bold, italic = "Review", "ReviewBold", "ReviewItalic"
    navy, teal = colors.HexColor("#16374c"), colors.HexColor("#247084")
    styles = {
        "body": ParagraphStyle("Body", fontName=regular, fontSize=9.1, leading=13.7,
                               textColor=colors.HexColor("#263744"), spaceAfter=8, splitLongWords=True,
                               allowWidows=0, allowOrphans=0),
        "title": ParagraphStyle("Title", fontName=bold, fontSize=23, leading=29,
                                textColor=navy, spaceAfter=18),
        "h2": ParagraphStyle("Section", fontName=bold, fontSize=13, leading=17,
                             textColor=teal, spaceBefore=15, spaceAfter=8, keepWithNext=True),
        "cell": ParagraphStyle("Cell", fontName=regular, fontSize=7.6, leading=10.4,
                               textColor=colors.HexColor("#263744")),
        "cellhead": ParagraphStyle("CellHead", fontName=bold, fontSize=7.5, leading=10.3, textColor=colors.white),
        "caption": ParagraphStyle("Caption", fontName=italic, fontSize=6.8, leading=9, spaceAfter=5),
    }
    def rich(text):
        text = html.escape(text.replace("\u2013", "-").replace("\u2014", "-"))
        text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"`([^`]+)`", r"<b>\1</b>", text)
        return text
    lines = (root/"docs/ANALISE_TECNICA.md").read_text(encoding="utf-8").splitlines()
    story, index = [], 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1; continue
        if line.startswith("# "):
            story.append(Paragraph(rich(line[2:]), styles["title"]))
            index += 1; continue
        if line.startswith("## "):
            story.append(Paragraph(rich(line[3:]), styles["h2"]))
            index += 1; continue
        if line.startswith("|"):
            rows = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                cells = [x.strip() for x in lines[index].strip().strip("|").split("|")]
                if not all(re.fullmatch(r"[: -]+", c) for c in cells):
                    rows.append(cells)
                index += 1
            width = A4[0]-100
            widths = [width*0.35, width*0.65] if len(rows[0]) == 2 else [width*0.28, width*0.28, width*0.22, width*0.22]
            table = Table([[Paragraph(rich(cell), styles["cellhead" if r == 0 else "cell"])
                            for cell in row] for r, row in enumerate(rows)], colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0,0), (-1,0), navy), ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f0f5f7"), colors.white]),
                ("LEFTPADDING", (0,0), (-1,-1), 7), ("RIGHTPADDING", (0,0), (-1,-1), 7),
                ("TOPPADDING", (0,0), (-1,-1), 7), ("BOTTOMPADDING", (0,0), (-1,-1), 7),
                ("LINEBELOW", (0,0), (-1,0), 0.5, navy)]))
            story.extend([table, Spacer(1, 10)])
            continue
        paragraph = [line]
        index += 1
        while index < len(lines) and lines[index].strip() and not lines[index].lstrip().startswith(("#", "|")):
            paragraph.append(lines[index].strip()); index += 1
        text = " ".join(paragraph)
        story.append(Paragraph(rich(text), styles["body"]))
        if text.startswith("Nenhuma execução deste benchmark") and (root/"validation/benchmark_linear.png").exists():
            figure_width = (A4[0]-100)*0.95
            story.append(KeepTogether([
                Image(str(root/"validation/benchmark_linear.png"), width=figure_width, height=figure_width*4.5/12.6),
                Paragraph("Figura 1. Surrogate linear; barras indicam médias e SEM entre cinco sementes. O eixo do cosseno factual é ampliado. Não constitui reprodução do PPP-Bench.", styles["caption"])]))
    output = root/"docs/Analise_CALMS_SAVANT_2026-11-11.pdf"
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d0dce2"))
        canvas.line(50, 42, A4[0]-50, 42)
        canvas.setFillColor(navy)
        canvas.setFont(regular, 7)
        canvas.drawString(50, 29, "CALMS-SAVANT | Análise técnica e software de referência | 11/11/2026")
        canvas.drawRightString(A4[0]-50, 29, str(doc.page))
        canvas.restoreState()
    document = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=50, rightMargin=50,
        topMargin=49, bottomMargin=57, title="CALMS-SAVANT: análise técnica e implementação verificável",
        author="Cleilson Elias de Sousa", pageCompression=1)
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    print(output)


if __name__ == "__main__":
    main()

"""PDF audit report. All text is generated from the actual scored data.
No conclusions are hard-coded. Simulated (demo) results get a red warning banner."""

import os
from datetime import datetime

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from trustbench.config import results_paths


def generate_pdf_report(mode: str = "real", output_pdf_path: str = None) -> str:
    paths = results_paths(mode)
    if not os.path.exists(paths["scored"]):
        raise FileNotFoundError("Cannot find scored data at " + paths["scored"])
    df = pd.read_csv(paths["scored"])
    out = output_pdf_path or paths["report"]
    demo = mode == "demo" or bool(df.get("simulated", pd.Series([False])).astype(bool).any())

    doc = SimpleDocTemplate(out, pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=20, textColor=colors.HexColor("#1e3a8a"))
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, textColor=colors.HexColor("#1e3a8a"))
    body = ParagraphStyle("b", parent=ss["Normal"], fontSize=9, leading=12)
    warn = ParagraphStyle("w", parent=body, textColor=colors.white, backColor=colors.HexColor("#b91c1c"),
                          borderPadding=6, fontSize=10, leading=14)
    cell = ParagraphStyle("c", parent=body, fontSize=8, leading=10)

    story = [Paragraph("Trust Bench: Audit Report", h1)]
    story.append(Paragraph("Generated: %s | Records: %s | Questions: %d | Runs per question: %d" % (
        datetime.now().strftime("%Y-%m-%d"), format(len(df), ","), df["qid"].nunique(), df["run"].nunique()), body))
    story.append(Spacer(1, 6))
    if demo:
        story.append(Paragraph("<b>DEMO / SIMULATED DATA.</b> These results come from a scripted simulator, "
                               "not from real models. They only prove that the pipeline works. "
                               "Do not quote any number from this report.", warn))
        story.append(Spacer(1, 8))

    answerable, traps = df[df["kind"] != "abstain"], df[df["kind"] == "abstain"]
    g = ["model", "approach"]
    summ = pd.DataFrame({
        "overall": df.groupby(g)["is_correct"].mean(),
        "answerable": answerable.groupby(g)["is_correct"].mean(),
        "trap_handled": traps.groupby(g)["is_correct"].mean(),
        "halluc": traps.groupby(g)["outcome"].apply(lambda s: (s == "hallucinated").mean()),
        "n": df.groupby(g).size(),
    }).reset_index()

    def pct(x):
        return "n/a" if pd.isna(x) else "%.1f%%" % (x * 100)

    story.append(Paragraph("1. Results", h2))
    data = [["Model", "Approach", "Overall", "Answerable", "Trap handled", "Hallucination", "Evals"]]
    for _, r in summ.iterrows():
        data.append([r["model"], r["approach"], pct(r["overall"]), pct(r["answerable"]),
                     pct(r["trap_handled"]), pct(r["halluc"]), str(int(r["n"]))])
    t = Table([[Paragraph(str(c), cell) for c in row] for row in data], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#94a3b8"))]))
    story.append(t)
    story.append(Spacer(1, 8))

    story.append(Paragraph("2. Reading guide", h2))
    for line in [
        "Overall = share of answers that were correct, or correctly refused on trap questions.",
        "n/a means that the group has no questions of that type in this run.",
        "Differences of a few points on only %d questions can be chance. See the confidence "
        "intervals and the paired comparison in the dashboard before claiming a winner." % df["qid"].nunique(),
    ]:
        story.append(Paragraph("- " + line, body))
    doc.build(story)
    return out

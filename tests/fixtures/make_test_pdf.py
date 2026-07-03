"""Generate a synthetic multi-page PDF fixture for testing the PDF ingestion skill stack."""
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
import os

OUT = os.path.join(os.path.dirname(__file__), "hydra-settlement-test-fixture.pdf")

styles = getSampleStyleSheet()
h1 = ParagraphStyle("H1", parent=styles["Heading1"], spaceAfter=12)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], spaceAfter=8)
body = styles["BodyText"]

story = []

story.append(Paragraph("Hydra Settlement: A Synchronous Layer-2 for eUTXO Chains", h1))
story.append(Paragraph("Working Paper, Test Fixture Edition", styles["Italic"]))
story.append(Spacer(1, 0.3 * inch))

story.append(Paragraph("1. Introduction", h1))
story.append(Paragraph(
    "Hydra is a family of Layer-2 protocols designed to scale Cardano-style "
    "eUTXO ledgers by allowing a small set of participants to run an "
    "isolated, high-throughput state channel called a Hydra Head. "
    "Transactions inside a Head settle instantly among participants and are "
    "only reconciled with the main chain when the Head is closed.",
    body))
story.append(Spacer(1, 0.2 * inch))

story.append(Paragraph("2. Architecture", h1))
story.append(Paragraph("2.1 Head Lifecycle", h2))
story.append(Paragraph(
    "A Hydra Head moves through four phases: Init, Open, Close, and "
    "Fanout. During Init, participants commit UTXOs from the main chain. "
    "During Open, participants exchange signed snapshots off-chain. Close "
    "posts the latest snapshot on-chain, and Fanout distributes the final "
    "UTXO set back to the main ledger.",
    body))
story.append(Spacer(1, 0.15 * inch))

story.append(Paragraph("2.2 Settlement Guarantees", h2))
story.append(Paragraph(
    "Because every state transition inside the Head requires unanimous "
    "signatures from all participants, Hydra can be used as a synchronous "
    "settlement layer over Cardano-style eUTXO state: once all parties "
    "sign a snapshot, that state is final among them even before it "
    "touches the main chain.",
    body))
story.append(Spacer(1, 0.2 * inch))

story.append(Paragraph("3. Throughput Comparison", h1))
story.append(Paragraph(
    "The table below compares theoretical transaction throughput across "
    "configurations tested in the reference implementation.",
    body))
story.append(Spacer(1, 0.15 * inch))

table_data = [
    ["Configuration", "Participants", "TPS", "Finality"],
    ["Baseline L1", "-", "250", "~20s"],
    ["Hydra Head (small)", "3", "1,000", "<1s"],
    ["Hydra Head (large)", "10", "800", "<1s"],
]
t = Table(table_data, colWidths=[1.8 * inch, 1.2 * inch, 0.9 * inch, 0.9 * inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b2b2b")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("GRID", (0, 0), (-1, -1), 0.75, colors.grey),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
]))
story.append(t)
story.append(Spacer(1, 0.3 * inch))

story.append(PageBreak())

story.append(Paragraph("4. Threat Model", h1))
story.append(Paragraph(
    "Hydra Heads assume an honest majority is not required; instead, "
    "safety relies on unanimity and a contestation period during Close. "
    "If a participant posts a stale snapshot, any other participant can "
    "contest it on-chain within the contestation window by presenting a "
    "newer, validly signed snapshot.",
    body))
story.append(Spacer(1, 0.2 * inch))

story.append(Paragraph("5. Open Questions", h1))
story.append(Paragraph(
    "It remains an open question how Hydra Heads compose with external "
    "delegated-signing standards such as the Open Wallet Standard (OWS), "
    "particularly when an autonomous agent — rather than a human "
    "participant — holds one of the Head's signing keys.",
    body))
story.append(Spacer(1, 0.2 * inch))

story.append(Paragraph("Figure 1: Head Lifecycle (placeholder)", styles["Italic"]))

doc = SimpleDocTemplate(OUT, pagesize=letter, title="Hydra Settlement Test Fixture")
doc.build(story)
print(f"wrote {OUT}")

"""
PDF Report Generator — generates a professional match report PDF.

Uses fpdf2 to create a branded PDF with:
- Score summary
- Skill matches
- Prioritized improvements
- Recommendations
- ML details
"""

from __future__ import annotations

import io
from datetime import datetime

from fpdf import FPDF


def _safe(text: str) -> str:
    """Sanitize text for fpdf2 core fonts (latin-1 only)."""
    replacements = {
        "\u2014": "-",  # em-dash
        "\u2013": "-",  # en-dash
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u2026": "...",  # ellipsis
        "\u2022": "*",  # bullet
        "\u00a0": " ",  # non-breaking space
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Fallback: replace any remaining non-latin-1 chars
    return text.encode("latin-1", errors="replace").decode("latin-1")


# Brand colors
NAVY = (30, 58, 95)
BLUE = (37, 99, 235)
GREEN = (5, 150, 105)
YELLOW = (245, 158, 11)
RED = (220, 38, 38)
GRAY = (107, 114, 128)
LIGHT_GRAY = (243, 244, 246)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class MatchReportPDF(FPDF):
    """Custom PDF class for match reports."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=20)

    def header(self):
        """Branded header on every page."""
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(*NAVY)
        self.cell(0, 8, "CV Match Engine", align="L")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 8, datetime.now().strftime("%B %d, %Y"), align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*BLUE)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        """Page number footer."""
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, title: str):
        """Add a section title."""
        self.ln(5)
        self.set_font("Helvetica", "B", 12)
        self.set_text_color(*NAVY)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*BLUE)
        self.set_line_width(0.3)
        self.line(10, self.get_y(), 80, self.get_y())
        self.ln(3)

    def add_score_summary(self, overall_percent: int, band: str, ml_label: str, ml_score: float):
        """Add the main score summary box."""
        self.ln(5)

        # Score box background
        self.set_fill_color(*LIGHT_GRAY)
        self.rect(10, self.get_y(), 190, 35, style="F")

        # Score
        self.set_font("Helvetica", "B", 28)
        self.set_text_color(*NAVY)
        self.set_xy(15, self.get_y() + 3)
        self.cell(40, 15, f"{overall_percent}%", align="C")

        # Band
        self.set_font("Helvetica", "", 14)
        self.set_text_color(*GRAY)
        self.set_xy(60, self.get_y())
        self.cell(130, 8, band, align="L")

        # ML Label
        self.set_font("Helvetica", "", 10)
        ml_color = GREEN if ml_label == "Good Fit" else (YELLOW if ml_label == "Potential Fit" else RED)
        self.set_text_color(*ml_color)
        self.set_xy(60, self.get_y() + 10)
        self.cell(130, 8, f"ML Prediction: {ml_label} ({ml_score:.0%})", align="L")

        self.set_y(self.get_y() + 25)

    def add_skill_matches(self, skill_matches: list[dict]):
        """Add skill match table."""
        self.section_title("Skill Matches")

        # Table header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(*NAVY)
        self.set_text_color(*WHITE)
        self.cell(50, 7, "Required Skill", fill=True, align="C")
        self.cell(50, 7, "Your Skill", fill=True, align="C")
        self.cell(30, 7, "Match", fill=True, align="C")
        self.cell(50, 7, "Evidence", fill=True, align="C")
        self.ln()

        # Table rows
        self.set_font("Helvetica", "", 8)
        for i, match in enumerate(skill_matches):
            bg = WHITE if i % 2 == 0 else LIGHT_GRAY
            self.set_fill_color(*bg)

            required = _safe(match.get("required_skill", ""))
            candidate = _safe(match.get("candidate_skill", "") or "—")
            level = _safe(match.get("match_class", ""))
            evidence = _safe(match.get("evidence", "")[:60])

            # Color code match level
            if level == "matched":
                self.set_text_color(*GREEN)
            elif level == "partial":
                self.set_text_color(*YELLOW)
            else:
                self.set_text_color(*RED)

            self.cell(50, 6, required, fill=True, align="C")
            self.set_text_color(*BLACK)
            self.cell(50, 6, candidate or "—", fill=True, align="C")
            self.cell(30, 6, level, fill=True, align="C")
            self.set_text_color(*GRAY)
            self.cell(50, 6, evidence, fill=True, align="C")
            self.set_text_color(*BLACK)
            self.ln()

    def add_prioritized_improvements(self, improvements: list[dict]):
        """Add prioritized improvement actions."""
        self.section_title("How to Improve Your CV")

        for i, imp in enumerate(improvements[:7]):
            priority = imp.get("priority", "medium")
            action = imp.get("action", "")
            impact = imp.get("impact", 0)

            # Priority badge
            if priority == "high":
                self.set_fill_color(*RED)
                badge = "HIGH"
            elif priority == "medium":
                self.set_fill_color(*YELLOW)
                badge = "MED"
            else:
                self.set_fill_color(*GRAY)
                badge = "LOW"

            y_start = self.get_y()
            if y_start > 260:
                self.add_page()
                y_start = self.get_y()

            # Badge
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(*WHITE)
            self.set_xy(10, y_start)
            self.cell(15, 6, badge, fill=True, align="C")

            # Impact
            self.set_font("Helvetica", "", 8)
            self.set_text_color(*GRAY)
            self.set_xy(26, y_start)
            self.cell(20, 6, f"+{impact:.0f}pts", align="L")

            # Action text
            self.set_text_color(*BLACK)
            self.set_xy(46, y_start)
            self.multi_cell(144, 6, _safe(action))
            self.ln(1)

    def add_recommendations(self, recommendations: list[str]):
        """Add recommendations list."""
        self.section_title("Recommendations")

        self.set_font("Helvetica", "", 9)
        for i, rec in enumerate(recommendations[:8]):
            if self.get_y() > 260:
                self.add_page()
            self.set_text_color(*NAVY)
            self.cell(5, 6, f"{i+1}.", align="R")
            self.set_text_color(*BLACK)
            self.multi_cell(175, 6, _safe(f" {rec}"))
            self.ln(1)

    def add_ats_details(self, ats_details: dict):
        """Add ATS friendliness section."""
        if not ats_details:
            return

        self.section_title("ATS Friendliness")

        score = ats_details.get("overall_score", 0)
        pass_est = ats_details.get("pass_estimate", "unknown")

        # Score box
        self.set_fill_color(*LIGHT_GRAY)
        self.rect(10, self.get_y(), 190, 15, style="F")

        self.set_font("Helvetica", "B", 14)
        if score >= 75:
            self.set_text_color(*GREEN)
            status = "LIKELY TO PASS"
        elif score >= 50:
            self.set_text_color(*YELLOW)
            status = "BORDERLINE"
        else:
            self.set_text_color(*RED)
            status = "LIKELY TO FAIL"

        self.set_xy(15, self.get_y() + 2)
        self.cell(40, 10, f"{score}/100", align="C")
        self.set_font("Helvetica", "", 10)
        self.cell(140, 10, status, align="C")
        self.ln(15)

        # Suggestions
        suggestions = ats_details.get("suggestions", [])
        if suggestions:
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(*NAVY)
            self.cell(0, 6, "ATS Improvements:", new_x="LMARGIN", new_y="NEXT")
            self.set_font("Helvetica", "", 8)
            for sug in suggestions[:5]:
                if self.get_y() > 260:
                    self.add_page()
                self.set_text_color(*BLACK)
                self.multi_cell(0, 5, _safe(f"  * {sug}"))
                self.ln(1)

    def add_ml_details(self, ml_details: dict):
        """Add ML model details."""
        if not ml_details:
            return

        self.section_title("ML Model Details")

        self.set_font("Helvetica", "", 9)
        details = [
            ("Model Version", ml_details.get("trained_model_version", "N/A")),
            ("Prediction", ml_details.get("label", "N/A")),
            ("Fit Score", f"{ml_details.get('fit_score', 0):.2%}"),
            ("Calibration", ml_details.get("calibration_method", "N/A")),
            ("Confidence", ml_details.get("confidence_note", "N/A")),
        ]

        for label, value in details:
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(*GRAY)
            self.cell(40, 6, f"{label}:", align="L")
            self.set_font("Helvetica", "", 9)
            self.set_text_color(*BLACK)
            self.cell(0, 6, str(value), align="L", new_x="LMARGIN", new_y="NEXT")

        # Probabilities
        probs = ml_details.get("probabilities", {})
        if probs:
            self.ln(3)
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(*NAVY)
            self.cell(0, 6, "Class Probabilities:", new_x="LMARGIN", new_y="NEXT")
            for cls, prob in probs.items():
                self.set_font("Helvetica", "", 9)
                self.set_text_color(*GRAY)
                self.cell(40, 6, f"  {cls}:", align="L")
                self.set_text_color(*BLACK)
                # Color code
                if prob > 0.5:
                    self.set_text_color(*GREEN)
                elif prob > 0.3:
                    self.set_text_color(*YELLOW)
                else:
                    self.set_text_color(*RED)
                self.cell(0, 6, f"{prob:.1%}", align="L", new_x="LMARGIN", new_y="NEXT")
                self.set_text_color(*BLACK)

    def add_disclaimer(self, disclaimer: str):
        """Add disclaimer."""
        self.ln(5)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*GRAY)
        self.multi_cell(0, 4, _safe(disclaimer))


def generate_match_report_pdf(
    overall_percent: int,
    band: str,
    ml_label: str,
    ml_score: float,
    skill_matches: list[dict],
    prioritized_improvements: list[dict],
    recommendations: list[str],
    ml_details: dict | None = None,
    ats_details: dict | None = None,
    disclaimer: str = "",
) -> bytes:
    """
    Generate a match report PDF and return bytes.

    Returns:
        PDF file content as bytes.
    """
    pdf = MatchReportPDF()
    pdf.alias_nb_pages()
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(*NAVY)
    pdf.cell(0, 12, "CV Match Report", align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # Score summary
    pdf.add_score_summary(overall_percent, band, ml_label, ml_score)

    # ATS details
    if ats_details:
        pdf.add_ats_details(ats_details)

    # Skill matches
    if skill_matches:
        pdf.add_skill_matches(skill_matches)

    # Prioritized improvements
    if prioritized_improvements:
        if pdf.get_y() > 200:
            pdf.add_page()
        pdf.add_prioritized_improvements(prioritized_improvements)

    # Recommendations
    if recommendations:
        if pdf.get_y() > 200:
            pdf.add_page()
        pdf.add_recommendations(recommendations)

    # ML details
    if ml_details:
        if pdf.get_y() > 200:
            pdf.add_page()
        pdf.add_ml_details(ml_details)

    # Disclaimer
    if disclaimer:
        pdf.add_disclaimer(disclaimer)

    # Get bytes
    return bytes(pdf.output())

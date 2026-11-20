"""Export the full solution to Markdown, JSON and PDF."""

from __future__ import annotations

import io
import json
from fractions import Fraction

from solver.models import Result, Problem
from solver.steps import StepLog


def result_to_dict(result: Result, problem: Problem) -> dict:
    return {
        "method": result.method,
        "status": result.status,
        "solution": {k: str(v) for k, v in result.solution.items()},
        "objective_value": (str(result.objective_value)
                            if result.objective_value is not None else None),
        "problem": {
            "sense": problem.sense,
            "objective": {k: str(v) for k, v in problem.objective.items()},
            "constraints": [
                {"coeffs": {k: str(v) for k, v in c.coeffs.items()},
                 "op": c.op, "rhs": str(c.rhs)}
                for c in problem.constraints
            ],
            "variables": [{"name": v.name, "vtype": v.vtype}
                          for v in problem.variables],
        },
        "verification": [
            {"expr": v.expr, "lhs": str(v.lhs), "op": v.op, "rhs": str(v.rhs),
             "slack": str(v.slack), "satisfied": v.satisfied}
            for v in result.verification
        ],
        "steps": [s.to_dict() for s in result.steps],
        "bb_tree": result.bb_tree,
        "notes": result.notes,
    }


def to_json(result: Result, problem: Problem) -> str:
    return json.dumps(result_to_dict(result, problem), indent=2, default=str)


def to_markdown(result: Result, problem: Problem) -> str:
    lines = []
    lines.append(f"# LP/ILP Solution — {result.method}")
    lines.append("")
    lines.append(f"**Status:** {result.status}")
    if result.objective_value is not None:
        lines.append(f"**Objective value Z =** `{result.objective_value}`")
    if result.solution:
        sol = ", ".join(f"{k} = {v}" for k, v in result.solution.items())
        lines.append(f"**Solution:** {sol}")
    lines.append("")
    lines.append("## Problem")
    lines.append(f"```\n{problem_str(problem)}\n```")
    lines.append("")
    lines.append("## Step-by-step")
    for i, step in enumerate(result.steps):
        lines.append(f"### {step.title}")
        if step.description:
            lines.append(step.description)
        for snap in step.snapshots:
            lines.append("")
            header = " | ".join(snap.columns)
            sep = " | ".join(["---"] * len(snap.columns))
            lines.append(header)
            lines.append(sep)
            zrow = [snap.objective_label] + snap.z_row
            lines.append(" | ".join(zrow))
            for r, basis in enumerate(snap.basis):
                row = [basis] + snap.rows[r] + [snap.rhs[r]]
                ratio = snap.ratios[r + 1] if r + 1 < len(snap.ratios) else ""
                row.append(ratio)
                lines.append(" | ".join(row))
            if snap.reason:
                lines.append("")
                lines.append(f"> {snap.reason}")
        for note in getattr(step, "notes", []) or []:
            lines.append(f"_{note}_")
        lines.append("")
    lines.append("## Verification")
    for v in result.verification:
        mark = "PASS" if v.satisfied else "FAIL"
        lines.append(f"- `{v.expr}`  ->  LHS = {v.lhs}, RHS = {v.rhs}, "
                     f"slack = {v.slack}  [{mark}]")
    return "\n".join(lines)


def problem_str(problem: Problem) -> str:
    parts = [f"{problem.sense} Z = "
             + " + ".join(f"{v}{n}" for n, v in problem.objective.items() if v != 0)
             or f"{problem.sense} Z = 0"]
    parts.append("subject to:")
    for c in problem.constraints:
        terms = " + ".join(f"{v}{n}" for n, v in c.coeffs.items() if v != 0)
        parts.append(f"  {terms} {c.op} {c.rhs}")
    vt = ", ".join(f"{v.name}={v.vtype}" for v in problem.variables)
    parts.append(f"variables: {vt}")
    return "\n".join(parts)


def to_pdf(result: Result, problem: Problem) -> bytes:
    """Render the Markdown solution to a PDF using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                    Preformatted, Table, TableStyle)
    from reportlab.lib import colors

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                           leftMargin=1.5*cm, rightMargin=1.5*cm,
                           topMargin=1.5*cm, bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    normal = styles["Normal"]
    code = ParagraphStyle("code", parent=normal, fontName="Courier",
                          fontSize=8, leading=10)
    story = []
    story.append(Paragraph(f"LP/ILP Solution — {result.method}", h1))
    story.append(Spacer(1, 6))
    story.append(Paragraph(f"<b>Status:</b> {result.status}", normal))
    if result.objective_value is not None:
        story.append(Paragraph(
            f"<b>Objective value Z =</b> {result.objective_value}", normal))
    if result.solution:
        sol = ", ".join(f"{k} = {v}" for k, v in result.solution.items())
        story.append(Paragraph(f"<b>Solution:</b> {sol}", normal))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Problem", h2))
    story.append(Preformatted(problem_str(problem), code))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Step-by-step", h2))
    for step in result.steps:
        story.append(Paragraph(step.title, styles["Heading3"]))
        if step.description:
            story.append(Paragraph(step.description.replace("<", "&lt;"),
                                   normal))
        for snap in step.snapshots:
            story.append(_snapshot_table(snap, code))
            if snap.reason:
                story.append(Paragraph(f"<i>{snap.reason}</i>", normal))
            story.append(Spacer(1, 4))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Verification", h2))
    for v in result.verification:
        mark = "PASS" if v.satisfied else "FAIL"
        story.append(Paragraph(
            f"{v.expr}  ->  LHS={v.lhs}, RHS={v.rhs}, slack={v.slack} [{mark}]",
            normal))
    doc.build(story)
    return buf.getvalue()


def _snapshot_table(snap, code_style):
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors
    header = ["Basis"] + snap.columns
    rows = [[snap.objective_label] + snap.z_row]
    for r, basis in enumerate(snap.basis):
        ratio = snap.ratios[r + 1] if r + 1 < len(snap.ratios) else ""
        rows.append([basis] + snap.rows[r] + [snap.rhs[r], ratio])
    data = [header] + rows
    t = Table(data, repeatRows=1)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Courier"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
    ]
    if snap.entering_col is not None:
        style.append(("BACKGROUND", (snap.entering_col + 1, 1),
                      (snap.entering_col + 1, -1), colors.HexColor("#fde68a")))
    if snap.leaving_row is not None:
        style.append(("BACKGROUND", (0, snap.leaving_row + 1),
                      (-1, snap.leaving_row + 1), colors.HexColor("#fde68a")))
    if snap.pivot_cell is not None:
        lr, ec = snap.pivot_cell
        style.append(("BACKGROUND", (ec + 1, lr + 1), (ec + 1, lr + 1),
                      colors.HexColor("#f59e0b")))
    t.setStyle(TableStyle(style))
    return t

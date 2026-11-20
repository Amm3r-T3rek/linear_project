"""Render a :class:`TableauSnapshot` as a highlighted pandas DataFrame."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from solver.steps import TableauSnapshot


def render_snapshot(snap: TableauSnapshot, fmt: str = "frac") -> None:
    """Display one tableau snapshot with the entering column, leaving row,
    pivot cell and ratio column highlighted."""
    cols = list(snap.columns)
    # Build a matrix: row 0 = objective row, rows 1..m = constraint rows.
    data = [list(snap.z_row)]
    for row in snap.rows:
        data.append(list(row))
    df = pd.DataFrame(data, columns=cols)

    # Basis column: objective label for row 0, then the basic variable names.
    df.insert(0, "Basis", [snap.objective_label] + list(snap.basis))
    # Ratio column: one entry per row (objective row gets "").
    df.insert(len(df.columns), "Ratio", list(snap.ratios))

    styler = df.style
    # Highlight the entering column (variable columns are at df positions
    # 1 .. len(cols)-1, i.e. excluding the leading Basis and trailing RHS).
    if snap.entering_col is not None:
        entering_df_col = df.columns[snap.entering_col + 1]
        styler = styler.map(_highlight_cell,
                            subset=(slice(None), [entering_df_col]))
    # Highlight the leaving row (row index snap.leaving_row + 1 in df).
    if snap.leaving_row is not None:
        leaving_df_row = snap.leaving_row + 1
        styler = styler.map(_highlight_cell,
                            subset=([leaving_df_row], slice(None)))
    # Pivot cell (strongest highlight).
    if snap.pivot_cell is not None:
        lr, ec = snap.pivot_cell
        entering_df_col = df.columns[ec + 1]
        styler = styler.map(_highlight_pivot,
                            subset=([lr + 1], [entering_df_col]))

    st.dataframe(styler, use_container_width=True, hide_index=True)
    if snap.reason:
        st.caption(snap.reason)


def _highlight_cell(val):
    return "background-color: #fde68a; color: #1f2937"


def _highlight_pivot(val):
    return "background-color: #f59e0b; color: #ffffff; font-weight: bold"


def render_steps(steps, fmt: str = "frac") -> None:
    """Render every step in the log with expanders."""
    for i, step in enumerate(steps):
        title = step.title or f"Step {i+1}"
        with st.expander(title, expanded=(i < 3)):
            if step.description:
                st.markdown(step.description)
            for snap in step.snapshots:
                render_snapshot(snap, fmt)
            for note in getattr(step, "notes", []) or []:
                st.markdown(f"_{note}_")

"""Dual simplex method.

Used after adding a cutting plane: the Z-row stays dual-feasible (optimal
reduced costs) while a new row makes the primal infeasible (negative RHS).
The dual simplex restores primal feasibility *without* losing dual feasibility.

Rules (course conventions):
  * Pivot row  = the row with the most negative RHS.
  * Pivot col  = argmin |z_j| / |a_rj|  over columns j with ``a_rj < 0``
                 (ties -> lowest index).
  * If the chosen row has no negative entry, the problem is infeasible.
"""

from __future__ import annotations

from typing import Optional

from fractions import Fraction

from .numbers import BigM
from .steps import Step, StepLog
from .tableau import Tableau


def run_dual_simplex(
    tab: Tableau,
    log: StepLog,
    forbidden: Optional[set[int]] = None,
    max_iter: int = 1000,
    iter_title: str = "Dual iteration",
) -> tuple[str, Tableau]:
    forbidden = forbidden or set()
    iteration = 0
    while iteration < max_iter:
        # pivot row: most negative RHS
        piv_row: Optional[int] = None
        for i in range(tab.num_rows):
            if tab.rhs[i] < 0:
                if piv_row is None or tab.rhs[i] < tab.rhs[piv_row]:
                    piv_row = i
        if piv_row is None:
            # feasible -> optimal
            return "optimal", tab

        # pivot column: min |z_j| / |a_rj| over a_rj < 0
        piv_col: Optional[int] = None
        best_ratio: Optional[Fraction] = None
        for j in range(tab.num_cols):
            if j in forbidden:
                continue
            a = tab.body[piv_row][j]
            if a >= 0:
                continue
            zj = tab.z_row[j]
            if zj.b != 0:
                # should not happen post-LP; skip defensively
                continue
            ratio = abs(zj.to_fraction()) / abs(a.to_fraction())
            if best_ratio is None or ratio < best_ratio or (
                ratio == best_ratio and j < piv_col  # type: ignore[operator]
            ):
                best_ratio = ratio
                piv_col = j

        if piv_col is None:
            reason = (
                f"Cut row {tab.basis[piv_row]} has RHS {tab.rhs[piv_row]} < 0 "
                f"but no negative coefficient; the cut makes the problem "
                f"infeasible (or a stronger cut is needed)."
            )
            snap = tab.snapshot(reason=reason, forbidden=forbidden)
            log.add(Step(f"{iter_title} {iteration+1}: infeasible",
                         reason, [snap]))
            return "infeasible", tab

        ratio = abs(tab.z_row[piv_col].to_fraction()) / abs(
            tab.body[piv_row][piv_col].to_fraction())
        reason = (
            f"Pivot row: {tab.basis[piv_row]} (RHS {tab.rhs[piv_row]} most "
            f"negative). Entering column: {tab.columns[piv_col]} "
            f"(min |z_j|/|a_rj| = {ratio}). Pivot element "
            f"{tab.body[piv_row][piv_col]} at (row {tab.basis[piv_row]}, "
            f"col {tab.columns[piv_col]})."
        )
        snap = tab.snapshot(reason=reason, entering=piv_col, leaving=piv_row,
                            forbidden=forbidden)
        log.add(Step(f"{iter_title} {iteration+1}", reason, [snap]))
        tab.pivot(piv_col, piv_row)
        iteration += 1

    return "maxiter", tab

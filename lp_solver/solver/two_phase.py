"""The two-phase simplex method.

Phase I minimises ``r = sum of artificials`` with a *plain* (M-free) simplex.
If ``r > 0`` at the optimum the original problem is infeasible.  Otherwise the
Phase-I optimal tableau is a feasible starting basis for Phase II, where the
real objective ``Z`` is substituted in and optimised.

The Phase-II setup pivots out any artificial that is still basic (at value 0)
so the displayed basis never contains an artificial variable.
"""

from __future__ import annotations

from typing import Optional

from .models import Problem, Result
from .numbers import BigM
from .simplex import build_tableau, run_simplex, _finalise, verify
from .standardize import StandardForm, standardize
from .steps import Step, StepLog
from .tableau import Tableau


def two_phase_core(problem: Problem, max_iter: int = 1000):
    """Run two-phase simplex and return ``(tab, sf, status, log)``.

    ``status`` is one of ``optimal``, ``infeasible``, ``unbounded``,
    ``maxiter``.  The returned tableau is the final one (with artificials
    forbidden in Phase II).
    """
    sf = standardize(problem)
    log = StepLog()

    log.add(Step("Standardisation (Two-Phase)",
                 f"Sense = {sf.sense}. Structural variables added: "
                 f"slacks {sorted(sf.slack_names)}, surplus "
                 f"{sorted(sf.surplus_names)}, artificials "
                 f"{sorted(sf.artificial_names)}.", []))

    if not sf.artificial_names:
        log.add(Step("Phase I", "No artificial variables are required; the "
                     "origin is already feasible. Skipping to Phase II.", []))
        tab = build_tableau(sf, sense=sf.sense, obj_label="Z")
        entering_word = ("most positive (min)" if sf.sense == "min"
                         else "most negative (max)")
        status, tab = run_simplex(tab, log, max_iter=max_iter,
                                  entering_word=entering_word)
        return tab, sf, status, log

    # ----- Phase I -------------------------------------------------------
    phase1_obj = {c: (BigM(1) if c in sf.artificial_names else BigM(0))
                  for c in sf.columns}
    tab = build_tableau(sf, sense="min", obj_label="r", objective=phase1_obj)

    pre = tab.snapshot(reason="Phase-I objective r = sum of artificials, "
                      "before substituting the basics out of the r-row.")
    log.add(Step("Phase I - setup", "Minimise r = R1 + R2 + ... (sum of "
                 "artificials).", [pre]))
    tab.cleanse_objective_row()
    post = tab.snapshot(reason="After substituting the basic artificials out "
                      "of the r-row.")
    log.add(Step("Phase I - r-row substituted",
                 "r is now expressed in nonbasic variables only.", [post]))

    status, tab = run_simplex(tab, log, max_iter=max_iter,
                              iter_title="Phase I iteration",
                              entering_word="most positive (min)")

    r_val = tab.z_rhs.to_fraction() if tab.z_rhs.b == 0 else None
    if status != "optimal":
        return tab, sf, status, log
    if r_val is not None and r_val > 0:
        reason = (
            f"Phase I optimum has r = {tab.z_rhs} > 0, so the artificial "
            f"variables cannot all be driven to zero. The feasible region is "
            f"empty -> INFEASIBLE."
        )
        snap = tab.snapshot(reason=reason, forbidden=tab.artificial_cols)
        log.add(Step("Infeasible", reason, [snap]))
        return tab, sf, "infeasible", log

    log.add(Step("Phase I complete",
                 f"r = 0; a basic feasible solution has been found. Proceed "
                 f"to Phase II with the original objective Z.", []))

    _drive_out_artificials(tab, sf, log)

    # ----- Phase II ------------------------------------------------------
    tab.obj_label = "Z"
    tab.sense = sf.sense
    real_obj = dict(sf.obj_coeffs)
    tab.z_row = [-BigM(real_obj.get(c, 0)) for c in tab.columns]
    tab.z_rhs = BigM(0)
    tab.cleanse_objective_row()

    forbidden = tab.artificial_cols
    setup2 = tab.snapshot(reason="Phase II start tableau: Z-row rebuilt from "
                          "the original objective and substituted against the "
                          "current basis. Artificial columns are forbidden.",
                          forbidden=forbidden)
    log.add(Step("Phase II - setup",
                 "Substitute the current basic variables into the original "
                 "objective Z; forbid the artificial columns.", [setup2]))

    entering_word = ("most positive (min)" if sf.sense == "min"
                     else "most negative (max)")
    status, tab = run_simplex(tab, log, forbidden=forbidden, max_iter=max_iter,
                              iter_title="Phase II iteration",
                              entering_word=entering_word)
    return tab, sf, status, log


def solve_two_phase(problem: Problem, max_iter: int = 1000) -> Result:
    tab, sf, status, log = two_phase_core(problem, max_iter=max_iter)
    if status == "infeasible":
        return Result(status="infeasible", solution={}, objective_value=None,
                      steps=log.steps, verification=verify(problem, {}),
                      method="Two-Phase")
    if status == "unbounded":
        return Result(status="unbounded", solution={}, objective_value=None,
                      steps=log.steps, verification=verify(problem, {}),
                      method="Two-Phase")
    return _finalise(tab, sf, status, log, method="Two-Phase", problem=problem)


def _drive_out_artificials(tab: Tableau, sf: StandardForm, log: StepLog) -> None:
    """Pivot out any artificial still basic after Phase I (value 0)."""
    changed = True
    while changed:
        changed = False
        for r in range(tab.num_rows):
            if tab.basis[r] not in sf.artificial_names:
                continue
            pivot_col: Optional[int] = None
            for j in range(tab.num_cols):
                if tab.columns[j] in sf.artificial_names:
                    continue
                if tab.body[r][j] != 0:
                    pivot_col = j
                    break
            if pivot_col is not None:
                name = tab.columns[pivot_col]
                tab.pivot(pivot_col, r)
                log.add(Step("Phase II - basis cleanup",
                             f"Artificial was basic at 0; pivoted in {name} "
                             f"to clean the basis.", []))
                changed = True
                break
            else:
                log.add(Step("Phase II - redundant row",
                             f"Row {r+1} (basic {tab.basis[r]}) is all zero "
                             f"outside the artificial; it is redundant and "
                             f"left in place.", []))

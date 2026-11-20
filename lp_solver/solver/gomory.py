"""Gomory fractional cutting-plane method (pure ILP).

1. Solve the LP relaxation.
2. If every integer-required variable is already integer, done.
3. Pick the basic integer variable whose RHS has the *largest* fractional
   part (ties -> lowest row); build the fractional cut
   ``sum f_ij x_j >= f_i``  ->  ``-sum f_ij x_j + s = -f_i``.
4. Add the cut row and re-optimise with the dual simplex.
5. Repeat (bounded by ``max_cuts``).
"""

from __future__ import annotations

import math
from fractions import Fraction

from .models import Problem, Result
from .numbers import BigM
from .simplex import solve_lp, _finalise, verify
from .standardize import StandardForm
from .steps import Step, StepLog
from .tableau import Tableau
from .dual_simplex import run_dual_simplex


def solve_gomory(problem: Problem, max_cuts: int = 50, max_iter: int = 1000) -> Result:
    log = StepLog()
    log.add(Step("Gomory cut method",
                 "Solve the LP relaxation, then repeatedly add fractional cuts "
                 "and re-optimise with the dual simplex until every integer "
                 "variable is integral.", []))

    tab, sf, status, sublog = solve_lp(problem, max_iter=max_iter)
    log.steps.extend(sublog.steps)

    if status not in ("optimal", "alternative"):
        msg = {"infeasible": "LP relaxation is infeasible -> ILP infeasible.",
               "unbounded": "LP relaxation is unbounded.",
               "maxiter": "LP relaxation did not converge."}.get(status, status)
        log.add(Step("Stop", msg, []))
        return Result(status=status, solution={}, objective_value=None,
                      steps=log.steps, verification=verify(problem, {}),
                      method="Gomory")

    log.add(Step("LP relaxation solved",
                 f"LP optimum Z = {tab.z_rhs}. Checking integrality of the "
                 f"integer variables {sorted(sf.integer_names)}.", []))

    for cut_no in range(1, max_cuts + 1):
        sol = sf.recover(tab.basic_solution())
        if _all_integer(sf, sol):
            log.add(Step("Integer solution found",
                         f"All integer variables are integral after "
                         f"{cut_no - 1} cut(s).", []))
            res = _finalise(tab, sf, "optimal", log, method="Gomory",
                            problem=problem)
            res.status = "optimal"  # integer optimum; ignore LP alt-optima flag
            return res

        row = _select_cut_row(tab, sf)
        if row is None:
            log.add(Step("No fractional integer basic variable",
                         "Nothing left to cut; returning current solution.", []))
            break

        coeffs, rhs, frac_parts, f_i = _build_gomory_row(tab, row)
        basis_name = tab.basis[row]
        cut_desc = (
            f"From the {basis_name}-row: f_i = {f_i} (largest fractional part). "
            f"Cut: {' + '.join(_cut_term(c, n) for c, n in zip(frac_parts, tab.columns) if c != 0) or '0'} "
            f">= {f_i}.  Added as a new row with slack s{cut_no}: RHS = -{f_i}."
        )
        log.add(Step(f"Cut {cut_no}", cut_desc, []))
        tab.add_cut_row(coeffs, BigM(rhs), f"s{cut_no}")

        snap = tab.snapshot(reason=f"Tableau after adding cut {cut_no} (slack "
                          f"s{cut_no}). RHS is negative -> run dual simplex.",
                          forbidden=tab.artificial_cols)
        log.add(Step(f"Cut {cut_no} added", "Dual simplex re-optimisation.", [snap]))

        dstatus, tab = run_dual_simplex(tab, log, max_iter=max_iter,
                                        iter_title=f"Cut {cut_no} dual iter")
        if dstatus == "infeasible":
            log.add(Step("Infeasible", "A cut made the problem infeasible; "
                         "the ILP has no feasible integer solution.", []))
            return Result(status="infeasible", solution={}, objective_value=None,
                          steps=log.steps, verification=verify(problem, {}),
                          method="Gomory")
        if dstatus == "maxiter":
            log.add(Step("Max iterations", "Dual simplex did not converge.", []))
            break

    sol = sf.recover(tab.basic_solution())
    if _all_integer(sf, sol):
        res = _finalise(tab, sf, "optimal", log, method="Gomory", problem=problem)
        res.status = "optimal"
        return res
    log.add(Step("Cut limit reached",
                 f"Reached the {max_cuts}-cut limit without an all-integer "
                 f"solution; returning the best relaxation found.", []))
    res = _finalise(tab, sf, "optimal", log, method="Gomory", problem=problem)
    res.status = "noninteger"
    return res


# --------------------------------------------------------------------------- #
def _frac(f: Fraction) -> Fraction:
    """Fractional part ``f - floor(f)`` in ``[0, 1)``."""
    return f - math.floor(f)


def _all_integer(sf: StandardForm, sol: dict[str, Fraction]) -> bool:
    for name in sf.integer_names:
        v = sol.get(name, Fraction(0))
        if v.denominator != 1:
            return False
        if name in sf.binary_names and v not in (0, 1):
            return False
    return True


def _select_cut_row(tab: Tableau, sf: StandardForm):
    """Pick the basic integer variable with the largest fractional RHS."""
    best_row = None
    best_frac = Fraction(0)
    for r in range(tab.num_rows):
        name = tab.basis[r]
        # map a basic column name back to an original variable, if possible
        orig = _orig_name(name)
        if orig not in sf.integer_names:
            continue
        val = tab.rhs[r]
        if val.b != 0:
            continue
        f = _frac(val.to_fraction())
        if f > 0 and (f > best_frac or (f == best_frac and best_row is None)):
            best_frac = f
            best_row = r
    return best_row


def _orig_name(col_name: str) -> str:
    """Recover the original variable name from a (possibly split) column."""
    if col_name.endswith("+") or col_name.endswith("-"):
        return col_name[:-1]
    if col_name.endswith("'"):
        return col_name[:-1]
    return col_name


def _build_gomory_row(tab: Tableau, row: int):
    """Return (coeffs list, rhs, frac_parts list, f_i)."""
    b = tab.rhs[row].to_fraction()
    f_i = _frac(b)
    coeffs: list[BigM] = []
    frac_parts: list[Fraction] = []
    for j in range(tab.num_cols):
        a = tab.body[row][j].to_fraction()
        f = _frac(a)
        frac_parts.append(f)
        coeffs.append(BigM(-f))
    return coeffs, -f_i, frac_parts, f_i


def _cut_term(c: Fraction, name: str) -> str:
    if c == 0:
        return ""
    if c.denominator == 1:
        cs = str(c.numerator)
    else:
        cs = f"{c.numerator}/{c.denominator}"
    return f"{cs} {name}"

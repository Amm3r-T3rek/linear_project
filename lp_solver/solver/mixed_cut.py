"""Mixed-integer cutting-plane method.

For a basic *integer* variable ``x_k`` with non-integer value ``d_k`` (row
``x_k + sum_{j in J} a_kj x_j = d_k``), let ``f_k = d_k - floor(d_k)`` and
split the nonbasic set into ``J+ = {a_kj > 0}`` and ``J- = {a_kj < 0}``.
The mixed-integer cut (course notes, eq. *) is::

    sum_{j in J-}  (f_k / (1 - f_k)) * a_kj * x_j
  - sum_{j in J+}  a_kj * x_j        <=  -f_k

i.e. in equality form with a slack ``s``::

    [those coefficients] . x + s = -f_k ,  s >= 0.

The cut is added as a new row and the tableau is re-optimised with the dual
simplex.  Only integer variables are cut; continuous variables are left free.
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


def solve_mixed_integer(problem: Problem, max_cuts: int = 50,
                        max_iter: int = 1000) -> Result:
    log = StepLog()
    log.add(Step("Mixed-integer cut method",
                 "Solve the LP relaxation, then add mixed cuts (only on "
                 "integer variables) and re-optimise with the dual simplex "
                 "until every integer variable is integral.", []))

    tab, sf, status, sublog = solve_lp(problem, max_iter=max_iter)
    log.steps.extend(sublog.steps)

    if status not in ("optimal", "alternative"):
        return Result(status=status, solution={}, objective_value=None,
                      steps=log.steps, verification=verify(problem, {}),
                      method="Mixed-Integer")

    for cut_no in range(1, max_cuts + 1):
        sol = sf.recover(tab.basic_solution())
        if _all_integer(sf, sol):
            log.add(Step("Integer solution found",
                         f"All integer variables integral after "
                         f"{cut_no - 1} cut(s).", []))
            res = _finalise(tab, sf, "optimal", log, method="Mixed-Integer",
                             problem=problem)
            res.status = "optimal"
            return res

        row = _select_cut_row(tab, sf)
        if row is None:
            log.add(Step("No fractional integer basic variable", "", []))
            break

        coeffs, rhs, f_k, desc = _build_mixed_row(tab, row)
        log.add(Step(f"Cut {cut_no}", desc, []))
        tab.add_cut_row(coeffs, BigM(rhs), f"s{cut_no}")
        snap = tab.snapshot(reason=f"Tableau after adding mixed cut {cut_no}.",
                            forbidden=tab.artificial_cols)
        log.add(Step(f"Cut {cut_no} added", "Dual simplex re-optimisation.",
                     [snap]))

        dstatus, tab = run_dual_simplex(tab, log, max_iter=max_iter,
                                        iter_title=f"Cut {cut_no} dual iter")
        if dstatus == "infeasible":
            return Result(status="infeasible", solution={}, objective_value=None,
                          steps=log.steps, verification=verify(problem, {}),
                          method="Mixed-Integer")
        if dstatus == "maxiter":
            break

    sol = sf.recover(tab.basic_solution())
    if _all_integer(sf, sol):
        res = _finalise(tab, sf, "optimal", log, method="Mixed-Integer",
                         problem=problem)
        res.status = "optimal"
        return res
    res = _finalise(tab, sf, "optimal", log, method="Mixed-Integer",
                    problem=problem)
    res.status = "noninteger"
    return res


def _frac(f: Fraction) -> Fraction:
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
    best_row = None
    best_frac = Fraction(0)
    for r in range(tab.num_rows):
        name = tab.basis[r]
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
    if col_name and (col_name[-1] in "+-'"):
        return col_name[:-1]
    return col_name


def _build_mixed_row(tab: Tableau, row: int):
    """Build the mixed cut row from the given body row."""
    d = tab.rhs[row].to_fraction()
    f_k = _frac(d)
    one_minus = 1 - f_k
    coeffs: list[BigM] = []
    basis_set = set(tab.basis)
    terms: list[str] = []
    for j in range(tab.num_cols):
        col_name = tab.columns[j]
        a = tab.body[row][j].to_fraction()
        if col_name in basis_set:
            # basic variables are excluded from the cut (J is the nonbasic set)
            coeffs.append(BigM(0))
            continue
        if a > 0:  # J+
            c = -a
            terms.append(f"{_fmt(c)} {col_name}")
        elif a < 0:  # J-
            c = (f_k * a) / one_minus  # negative
            terms.append(f"{_fmt(c)} {col_name}")
        else:
            c = Fraction(0)
        coeffs.append(BigM(c))
    desc = (
        f"From the {tab.basis[row]}-row: f_k = {f_k}. Mixed cut: "
        f"{' + '.join(t for t in terms if t) or '0'} <= -{f_k}, i.e. RHS = "
        f"-{f_k}."
    )
    return coeffs, -f_k, f_k, desc


def _fmt(c: Fraction) -> str:
    if c.denominator == 1:
        return str(c.numerator)
    return f"{c.numerator}/{c.denominator}"

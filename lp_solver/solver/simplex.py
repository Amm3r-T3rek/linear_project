"""Simplex method (pure LP, ``<=`` constraints, nonnegative variables).

Also provides the shared engine :func:`run_simplex` used by the Big-M and
two-phase methods.  The engine works on a :class:`Tableau` whose entries are
:class:`BigM`, so the same code drives exact-rational Simplex and the symbolic
Big-M iterations.
"""

from __future__ import annotations

from typing import Optional

from fractions import Fraction

from .models import Problem, Result, VerificationRow
from .numbers import BigM
from .standardize import StandardForm, standardize
from .steps import Step, StepLog, TableauSnapshot
from .tableau import Tableau


# --------------------------------------------------------------------------- #
#  Tableau construction
# --------------------------------------------------------------------------- #
def build_tableau(sf: StandardForm, sense: str = "max",
                  obj_label: str = "Z", objective: Optional[dict[str, BigM]] = None,
                  basis: Optional[list[str]] = None) -> Tableau:
    """Build an initial tableau from a :class:`StandardForm`.

    ``objective`` overrides the per-column objective coefficients (used by
    Big-M to inject the ``M`` penalties).  ``basis`` overrides the initial
    basis (used by two-phase when an artificial leaves during phase I).
    """
    if objective is None:
        objective = sf.obj_coeffs
    cols = sf.columns
    # z_row stores z_j so that  Z + sum z_j x_j = z_rhs.
    # max:  z_j = -c_j
    # min:  z_j = -c_j   (same formula; the sign convention differences are
    #       handled by the entering/optimality rules in Tableau.entering_col)
    z_row = [-BigM(objective.get(c, 0)) for c in cols]
    z_rhs = BigM(0)
    body = [list(row) for row in sf.body]
    rhs = list(sf.rhs)
    tab = Tableau(
        columns=cols,
        basis=basis or list(sf.basis),
        z_row=z_row,
        z_rhs=z_rhs,
        body=body,
        rhs=rhs,
        sense=sense,
        obj_label=obj_label,
        artificial_cols=sf.artificial_cols(),
        integer_cols=sf.integer_cols(),
    )
    # zero-out the z_row entries of the initial basic variables (so that the
    # basic columns form a unit vector in z_row as well).  This is what
    # "substitute the basic variables into Z" does algebraically.
    tab.cleanse_objective_row()
    return tab


# --------------------------------------------------------------------------- #
#  Core engine
# --------------------------------------------------------------------------- #
def run_simplex(
    tab: Tableau,
    log: StepLog,
    forbidden: Optional[set[int]] = None,
    max_iter: int = 1000,
    iter_title: str = "Iteration",
    entering_word: str = "most negative",
) -> tuple[str, Tableau]:
    """Run the primal simplex until optimal / unbounded / max-iter.

    Returns ``(status, tableau)`` where status is one of
    ``"optimal"``, ``"unbounded"``, ``"maxiter"``.
    """
    forbidden = forbidden or set()
    visited: set[tuple] = set()
    bland = False
    iteration = 0

    while iteration < max_iter:
        ent = tab.entering_col(forbidden, bland=bland)
        if ent is None:
            return "optimal", tab

        leav = tab.leaving_row(ent)
        if leav is None:
            reason = (
                f"Entering variable {tab.columns[ent]} has no positive entry in "
                f"any constraint row, so Z can grow without bound -> UNBOUNDED."
            )
            snap = tab.snapshot(reason=reason, entering=ent, forbidden=forbidden)
            log.add(Step(f"{iter_title} {iteration + 1}: unbounded", reason, [snap]))
            return "unbounded", tab

        ratio = tab.rhs[leav] / tab.body[leav][ent]
        zc = tab.z_row[ent]
        if iteration == 0:
            title = f"Initial tableau & {iter_title.lower()} 1"
        else:
            title = f"{iter_title} {iteration + 1}"
        reason = (
            f"Entering: {tab.columns[ent]} (Z-row coeff {zc}, {entering_word}). "
            f"Leaving: {tab.basis[leav]} (min ratio {tab.rhs[leav]} / "
            f"{tab.body[leav][ent]} = {ratio}). "
            f"Pivot element = {tab.body[leav][ent]} at (row {tab.basis[leav]}, "
            f"col {tab.columns[ent]})."
        )
        snap = tab.snapshot(reason=reason, entering=ent, leaving=leav,
                            forbidden=forbidden)
        log.add(Step(title, reason, [snap]))

        tab.pivot(ent, leav)

        # anti-cycling
        key = tuple(tab.basis)
        if key in visited and not bland:
            bland = True
            log.add(Step("Anti-cycling", "Repeated basis detected; switching to Bland's rule.",
                         []))
        visited.add(key)
        iteration += 1

    return "maxiter", tab


# --------------------------------------------------------------------------- #
#  Public API: solve with the plain Simplex method
# --------------------------------------------------------------------------- #
def solve_simplex(problem: Problem, max_iter: int = 1000) -> Result:
    """Solve a ``<=``-only LP with the plain Simplex method.

    Raises :class:`ValueError` if the problem needs artificial variables
    (use Big-M or Two-Phase instead).
    """
    sf = standardize(problem)
    if sf.artificial_names:
        raise ValueError(
            "Plain Simplex cannot handle '>=' or '=' constraints (needs "
            "artificial variables). Use Big-M or Two-Phase instead."
        )
    log = StepLog()
    setup = Step(
        "Standardisation",
        f"Sense = {sf.sense}. Every '<=' constraint received a slack variable "
        f"(S1..). No artificial variables are needed.",
        [],
    )
    log.add(setup)

    tab = build_tableau(sf, sense=sf.sense, obj_label="Z")
    entering_word = ("most positive (min)" if sf.sense == "min"
                     else "most negative (max)")
    status, tab = run_simplex(tab, log, max_iter=max_iter,
                               entering_word=entering_word)

    result = _finalise(tab, sf, status, log, method="Simplex", problem=problem)
    return result


# --------------------------------------------------------------------------- #
#  LP relaxation (used by Gomory / mixed-cut / branch-and-bound)
# --------------------------------------------------------------------------- #
def solve_lp(problem: Problem, max_iter: int = 1000):
    """Solve the continuous LP relaxation.

    Returns ``(tab, sf, status, log)``.  Picks plain simplex when no
    artificials are needed, otherwise two-phase.
    """
    from .two_phase import two_phase_core
    sf = standardize(problem)
    if sf.artificial_names:
        return two_phase_core(problem, max_iter=max_iter)
    log = StepLog()
    tab = build_tableau(sf, sense=sf.sense, obj_label="Z")
    entering_word = ("most positive (min)" if sf.sense == "min"
                     else "most negative (max)")
    status, tab = run_simplex(tab, log, max_iter=max_iter,
                               entering_word=entering_word)
    return tab, sf, status, log


# --------------------------------------------------------------------------- #
#  Helpers shared by every method
# --------------------------------------------------------------------------- #
def _finalise(tab: Tableau, sf: StandardForm, status: str, log: StepLog,
              method: str, problem: Problem) -> Result:
    if status == "optimal":
        alt = tab.has_alternative_optimum(forbidden=tab.artificial_cols)
        deg = tab.is_degenerate()
        opt_reason = (
            "All nonbasic Z-row coefficients are >= 0 (max) / <= 0 (min); "
            "the current basic feasible solution is optimal."
        )
        if alt:
            opt_reason += (" A nonbasic variable has a zero reduced cost and can "
                           "enter without changing Z -> alternative optima exist.")
            status = "alternative"
        if deg:
            opt_reason += " (Degenerate: at least one basic variable equals 0.)"
        snap = tab.snapshot(reason=opt_reason, forbidden=tab.artificial_cols)
        log.add(Step("Optimal tableau", opt_reason, [snap]))
    # solution
    raw = tab.basic_solution()
    recovered = sf.recover(raw)
    obj_val = None
    if status in ("optimal", "alternative"):
        try:
            obj_val = tab.z_rhs.to_fraction()
        except ValueError:
            obj_val = None
    result = Result(
        status=status,
        solution={k: v for k, v in recovered.items()},
        objective_value=obj_val,
        steps=log.steps,
        verification=verify(problem, recovered),
        method=method,
    )
    return result


def verify(problem: Problem, solution: dict[str, Fraction]) -> list[VerificationRow]:
    """Plug ``solution`` into every original constraint and report slack."""
    rows = []
    for idx, con in enumerate(problem.constraints):
        lhs = sum(Fraction(c) * Fraction(solution.get(n, 0))
                  for n, c in con.coeffs.items())
        rhs = Fraction(con.rhs)
        slack = lhs - rhs
        ok = _satisfied(con.op, slack)
        rows.append(VerificationRow(
            expr=_constraint_str(con, idx + 1),
            lhs=lhs, op=con.op, rhs=rhs, slack=slack, satisfied=ok,
        ))
    # objective row
    obj_val = sum(Fraction(c) * Fraction(solution.get(n, 0))
                  for n, c in problem.objective.items())
    rows.append(VerificationRow(
        expr=f"{problem.sense} Z = {_obj_str(problem)}",
        lhs=obj_val, op="=", rhs=obj_val, slack=Fraction(0), satisfied=True,
    ))
    return rows


def _satisfied(op: str, slack: Fraction) -> bool:
    # slack = lhs - rhs
    if op == "<=":
        return slack <= 0
    if op == ">=":
        return slack >= 0
    return slack == 0


def _constraint_str(con, idx: int) -> str:
    parts = [f"{_fmt(c)}{n}" for n, c in con.coeffs.items() if c != 0]
    return f"{' + '.join(parts)} {con.op} {_fmt(con.rhs)}"


def _fmt(c) -> str:
    c = Fraction(c)
    if c.denominator == 1:
        return str(c.numerator) if c == 1 else (f"{c.numerator}" if c == -1 else f"{c.numerator}")
    return f"{c.numerator}/{c.denominator}"


def _obj_str(problem: Problem) -> str:
    parts = [f"{_fmt(c)}{n}" for n, c in problem.objective.items() if c != 0]
    return " + ".join(parts) if parts else "0"


def problem_from_standard(sf: StandardForm) -> Problem:
    """Reconstruct a minimal Problem object for verification.  Not used for
    re-solving; only for constraint checking on the original variables."""
    from .models import Variable
    vs = [Variable(n) for n in sf.original_names]
    # constraints unknown here -> verification handled by callers with the real
    # problem; this stub is only a fallback.
    return Problem(sense=sf.sense, objective={}, constraints=[], variables=vs)

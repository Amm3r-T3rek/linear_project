"""The Big-M (penalty) method.

Artificial variables are penalised in the objective with a symbolic ``M``:

  * ``min`` problem  ->  ``+ M`` per artificial
  * ``max`` problem  ->  ``- M`` per artificial

``M`` is never given a numeric value; :class:`BigM` carries it symbolically
and the simplex engine compares coefficients lexicographically.  After the
optimal tableau, any artificial still positive (equivalently a nonzero
``M``-part in ``Z``) signals infeasibility.
"""

from __future__ import annotations

from .models import Problem, Result
from .numbers import BigM
from .simplex import build_tableau, run_simplex, _finalise, verify
from .standardize import standardize
from .steps import Step, StepLog
from .tableau import Tableau


def solve_big_m(problem: Problem, max_iter: int = 1000) -> Result:
    sf = standardize(problem)
    log = StepLog()

    # Objective coefficients with the M-penalty on artificials.
    objective = dict(sf.obj_coeffs)
    M = BigM(0, 1)
    for name in sf.artificial_names:
        objective[name] = M if sf.sense == "min" else -M

    # Standardisation narrative
    parts = []
    for i, con in enumerate(problem.constraints):
        tag = _constraint_tag(con.op)
        parts.append(f"constraint {i+1}: {tag}")
    setup_desc = (
        f"Sense = {sf.sense}. Artificial variables "
        f"{sorted(sf.artificial_names) or '(none)'} receive a "
        f"{'+' if sf.sense=='min' else '-'}M penalty in the objective so the "
        f"optimizer drives them to zero.  "
        + "; ".join(parts) + "."
    )
    log.add(Step("Standardisation (Big-M)", setup_desc, []))

    tab = build_tableau(sf, sense=sf.sense, obj_label="Z", objective=objective)

    # Show the setup tableau *before* substituting the artificials out of Z.
    pre = tab.snapshot(reason="Initial Big-M tableau before substituting the "
                      "basic artificials out of the Z-row.")
    log.add(Step("Initial tableau", "Z still contains the M-penalised artificials.",
                 [pre]))

    # Algebraically substitute R_i out of Z (zero their Z-row coefficients).
    tab.cleanse_objective_row()
    post = tab.snapshot(reason="After substituting the basic artificials out of "
                      "the Z-row, every coefficient has the form a + bM; the "
                      "M-part governs the entering choice (largest M wins).")
    log.add(Step("Z-row substitution", "Z is now expressed in terms of the "
                 "nonbasic variables only.", [post]))

    entering_word = "most positive (min)" if sf.sense == "min" else "most negative (max)"
    status, tab = run_simplex(tab, log, max_iter=max_iter,
                              entering_word=entering_word)

    # Infeasibility test: any artificial still positive.
    infeasible = _artificial_positive(tab, sf)
    if infeasible:
        reason = (
            "At the optimal Big-M tableau at least one artificial variable is "
            "still positive (equivalently Z carries a residual +M term). The "
            "original constraints are inconsistent -> INFEASIBLE."
        )
        snap = tab.snapshot(reason=reason, forbidden=tab.artificial_cols)
        log.add(Step("Infeasible", reason, [snap]))
        result = Result(status="infeasible", solution={}, objective_value=None,
                        steps=log.steps, verification=verify(problem, {}),
                        method="Big-M")
        return result

    return _finalise(tab, sf, status, log, method="Big-M", problem=problem)


def _artificial_positive(tab: Tableau, sf) -> bool:
    sol = tab.basic_solution()
    for name in sf.artificial_names:
        v = sol.get(name, BigM(0))
        if v > 0:
            return True
    # also check Z's M-part (an artificial contributing to Z signals trouble)
    try:
        if tab.z_rhs.b != 0:
            return True
    except AttributeError:
        pass
    return False


def _constraint_tag(op: str) -> str:
    if op == "<=":
        return "+ slack S"
    if op == ">=":
        return "- surplus S + artificial R"
    if op == "=":
        return "+ artificial R"
    return op

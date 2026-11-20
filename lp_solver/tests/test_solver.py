"""Pytest suite: 12 worked test cases + randomized property tests.

Run with:  pytest -q

The 12 worked cases pin every algorithm against the exact fractional answers
from the course PDFs.  The randomized cases cross-check the simplex family
against ``scipy.optimize.linprog`` and the integer solvers against brute-force
enumeration.
"""

from __future__ import annotations

import random
from fractions import Fraction as F

import pytest

from solver import (
    solve_simplex, solve_big_m, solve_two_phase,
    solve_gomory, solve_mixed_integer, solve_branch_bound,
    Problem, Constraint, Variable,
)
import examples as ex


def _c(coeffs, op, rhs):
    return Constraint({k: F(v) for k, v in coeffs.items()}, op, F(rhs))


# --------------------------------------------------------------------------- #
#  The 12 worked test cases
# --------------------------------------------------------------------------- #
class TestWorkedCases:
    def test_01_simplex(self):
        r = solve_simplex(ex.EX1)
        assert r.status == "optimal"
        assert r.solution["x1"] == F(10, 3)
        assert r.solution["x2"] == F(4, 3)
        assert r.objective_value == F(38, 3)

    def test_02_big_m(self):
        r = solve_big_m(ex.EX2)
        assert r.status == "optimal"
        assert r.solution["x1"] == F(2, 5)
        assert r.solution["x2"] == F(9, 5)
        assert r.objective_value == F(17, 5)

    def test_03_two_phase_same_lp(self):
        r = solve_two_phase(ex.EX2)
        assert r.status in ("optimal", "alternative")
        assert r.solution["x1"] == F(2, 5)
        assert r.solution["x2"] == F(9, 5)
        assert r.objective_value == F(17, 5)

    def test_04_two_phase(self):
        r = solve_two_phase(ex.EX4)
        assert r.status == "optimal"
        assert r.solution["x1"] == F(9, 2)
        assert r.solution["x2"] == F(5, 4)
        assert r.objective_value == F(41, 4)

    def test_05_max_x1_equality(self):
        r = solve_two_phase(ex.EX5)
        assert r.status == "optimal"
        assert r.solution["x1"] == F(4)
        assert r.solution["x2"] == F(0)
        assert r.objective_value == F(4)

    def test_06_infeasible_phase1(self):
        r = solve_two_phase(ex.EX6)
        assert r.status == "infeasible"

    def test_07a_O3_infeasible(self):
        r = solve_two_phase(ex.O3)
        assert r.status == "infeasible"

    def test_07b_O4_infeasible(self):
        r = solve_two_phase(ex.O4)
        assert r.status == "infeasible"

    def test_08_unbounded(self):
        r = solve_simplex(ex.EX8)
        assert r.status == "unbounded"

    def test_09a_O1_big_m_vs_scipy(self):
        r = solve_big_m(ex.O1)
        assert r.status == "optimal"
        scipy_val = _scipy_lp(ex.O1)
        assert r.objective_value == scipy_val

    def test_09b_O2_two_phase_vs_scipy(self):
        r = solve_two_phase(ex.O2)
        assert r.status in ("optimal", "alternative")
        scipy_val = _scipy_lp(ex.O2)
        assert r.objective_value == scipy_val

    def test_10_gomory(self):
        r = solve_gomory(ex.EX10)
        assert r.status == "optimal"
        assert r.solution["x"] == F(4)
        assert r.solution["y"] == F(3)
        assert r.objective_value == F(55)
        # first cut must be -(7/22) S1 - (1/22) S2 <= -1/2  (equivalently
        # 7/22 S1 + 1/22 S2 >= 1/2)
        cut = next(s for s in r.steps if s.title.startswith("Cut 1"))
        assert "7/22" in cut.description and "1/22" in cut.description
        assert "1/2" in cut.description

    def test_11_rounding_trap(self):
        # LP relaxation first to show the rounding trap
        r_lp = solve_two_phase(ex.EX11.as_lp_relaxation())
        assert r_lp.objective_value == F(11)
        assert r_lp.solution["x1"] == F(2)
        assert r_lp.solution["x2"] == F(9, 5)
        # integer optimum is (0, 2), Z = 10  (rounding would give (2,2) Z=12
        # which is infeasible, or (2,1) Z=7 which is suboptimal)
        r = solve_gomory(ex.EX11)
        assert r.status == "optimal"
        assert r.solution["x1"] == F(0)
        assert r.solution["x2"] == F(2)
        assert r.objective_value == F(10)

    def test_12_branch_and_bound(self):
        r = solve_branch_bound(ex.EX12)
        assert r.status == "optimal"
        assert r.objective_value == F(16)
        assert r.solution["x"] == F(4)
        assert r.solution["y"] == F(2)
        # cross-check against brute force
        bf = brute_force_integer(ex.EX12, bounds={"x": (0, 8), "y": (0, 3)})
        assert r.objective_value == bf


# --------------------------------------------------------------------------- #
#  Consistency: Simplex vs Big-M vs Two-Phase give the same answer
# --------------------------------------------------------------------------- #
class TestCompareMethods:
    def test_compare_on_ex1(self):
        # plain simplex works; two-phase should agree (origin feasible)
        rs = solve_simplex(ex.EX1)
        rt = solve_two_phase(ex.EX1)
        assert rs.objective_value == rt.objective_value == F(38, 3)

    def test_compare_on_big_m_lp(self):
        rb = solve_big_m(ex.EX2)
        rt = solve_two_phase(ex.EX2)
        assert rb.objective_value == rt.objective_value == F(17, 5)
        assert rb.solution == rt.solution

    def test_compare_on_O2(self):
        rb = solve_big_m(ex.O2)
        rt = solve_two_phase(ex.O2)
        assert rb.objective_value == rt.objective_value


# --------------------------------------------------------------------------- #
#  Verification rows must all be satisfied at the optimum
# --------------------------------------------------------------------------- #
class TestVerification:
    @pytest.mark.parametrize("solver,problem", [
        (solve_simplex, ex.EX1),
        (solve_big_m, ex.EX2),
        (solve_two_phase, ex.EX4),
        (solve_two_phase, ex.EX5),
        (solve_big_m, ex.O1),
        (solve_two_phase, ex.O2),
        (solve_gomory, ex.EX10),
        (solve_gomory, ex.EX11),
        (solve_branch_bound, ex.EX12),
    ])
    def test_all_constraints_satisfied(self, solver, problem):
        r = solver(problem)
        if r.status not in ("optimal", "alternative"):
            pytest.skip(f"status={r.status}")
        for row in r.verification[:-1]:  # last is the objective echo
            assert row.satisfied, f"{row.expr} not satisfied (slack={row.slack})"


# --------------------------------------------------------------------------- #
#  Randomized property tests vs scipy / brute force
# --------------------------------------------------------------------------- #
class TestRandomized:
    def test_lp_max_vs_scipy(self):
        random.seed(2024)
        for _ in range(40):
            problem = _random_leq_lp(seed=random.randint(0, 10**9))
            r = solve_simplex(problem)
            ref = _scipy_lp(problem)
            if ref is None:
                assert r.status in ("infeasible", "unbounded")
                continue
            assert r.status in ("optimal", "alternative")
            assert r.objective_value == ref

    def test_two_phase_vs_scipy(self):
        random.seed(7)
        for _ in range(40):
            problem = _random_mixed_lp(seed=random.randint(0, 10**9))
            r = solve_two_phase(problem)
            ref = _scipy_lp(problem)
            if ref is None:
                assert r.status in ("infeasible", "unbounded")
                continue
            assert r.status in ("optimal", "alternative")
            assert r.objective_value == ref

    def test_big_m_vs_scipy(self):
        random.seed(99)
        for _ in range(40):
            problem = _random_mixed_lp(seed=random.randint(0, 10**9))
            r = solve_big_m(problem)
            ref = _scipy_lp(problem)
            if ref is None:
                assert r.status in ("infeasible", "unbounded")
                continue
            assert r.status in ("optimal", "alternative")
            assert r.objective_value == ref

    def test_ilp_vs_brute_force(self):
        random.seed(321)
        for _ in range(25):
            problem, bounds = _random_ilp(seed=random.randint(0, 10**9))
            rg = solve_gomory(problem)
            rb = solve_branch_bound(problem)
            bf = brute_force_integer(problem, bounds)
            if bf is None:
                assert rg.status == "infeasible"
                assert rb.status == "infeasible"
                continue
            assert rg.status == "optimal"
            assert rb.status == "optimal"
            assert rg.objective_value == bf
            assert rb.objective_value == bf


# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def _scipy_lp(problem: Problem):
    """Return the optimal objective value as a Fraction, or None if
    infeasible/unbounded.  Uses scipy.linprog only for cross-checking."""
    try:
        from scipy.optimize import linprog
        import numpy as np
    except Exception:
        pytest.skip("scipy not available")
    names = [v.name for v in problem.variables]
    c = [float(problem.objective.get(n, 0)) for n in names]
    if problem.sense == "max":
        c = [-x for x in c]
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for con in problem.constraints:
        row = [float(con.coeffs.get(n, 0)) for n in names]
        b = float(con.rhs)
        if con.op == "<=":
            A_ub.append(row); b_ub.append(b)
        elif con.op == ">=":
            A_ub.append([-x for x in row]); b_ub.append(-b)
        else:
            A_eq.append(row); b_eq.append(b)
    res = linprog(c, A_ub=A_ub or None, b_ub=b_ub or None,
                  A_eq=A_eq or None, b_eq=b_eq or None,
                  bounds=[(0, None)] * len(names), method="highs")
    if not res.success:
        return None
    val = -res.fun if problem.sense == "max" else res.fun
    # round to recover the exact fraction
    return F(val).limit_denominator(1000)


def brute_force_integer(problem: Problem, bounds: dict[str, tuple[int, int]]):
    """Enumerate every integer point in the given bounds; return the optimal
    objective value (Fraction) or None if infeasible everywhere."""
    names = [v.name for v in problem.variables if v.is_integer]
    ranges = [range(bounds[n][0], bounds[n][1] + 1) for n in names]
    best = None
    import itertools
    for combo in itertools.product(*ranges):
        sol = dict(zip(names, [F(v) for v in combo]))
        ok = True
        for con in problem.constraints:
            lhs = sum(F(c) * sol.get(n, F(0))
                      for n, c in con.coeffs.items())
            rhs = F(con.rhs)
            if con.op == "<=" and not (lhs <= rhs): ok = False; break
            if con.op == ">=" and not (lhs >= rhs): ok = False; break
            if con.op == "=" and not (lhs == rhs): ok = False; break
        if not ok:
            continue
        val = sum(F(c) * sol.get(n, F(0))
                  for n, c in problem.objective.items())
        if best is None or (problem.sense == "max" and val > best) \
                or (problem.sense == "min" and val < best):
            best = val
    return best


def _random_leq_lp(seed):
    rng = random.Random(seed)
    n = 2
    m = rng.randint(2, 4)
    obj = {f"x{i+1}": F(rng.randint(1, 9)) for i in range(n)}
    cons = []
    for _ in range(m):
        coeffs = {f"x{i+1}": F(rng.randint(-3, 6)) for i in range(n)}
        rhs = F(rng.randint(1, 20))
        cons.append(_c(coeffs, "<=", rhs))
    vars_ = [Variable(f"x{i+1}") for i in range(n)]
    return Problem("max", obj, cons, vars_)


def _random_mixed_lp(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 3)
    m = rng.randint(2, 4)
    obj = {f"x{i+1}": F(rng.randint(1, 6)) for i in range(n)}
    cons = []
    for _ in range(m):
        op = rng.choice(["<=", ">=", "="])
        coeffs = {f"x{i+1}": F(rng.randint(-2, 5)) for i in range(n)}
        rhs = F(rng.randint(1, 15))
        cons.append(_c(coeffs, op, rhs))
    sense = rng.choice(["max", "min"])
    vars_ = [Variable(f"x{i+1}") for i in range(n)]
    return Problem(sense, obj, cons, vars_)


def _random_ilp(seed):
    rng = random.Random(seed)
    n = 2
    obj = {f"x{i+1}": F(rng.randint(1, 9)) for i in range(n)}
    cons = []
    # explicit upper bounds keep the LP bounded and inside the brute-force box
    for i in range(n):
        ub = rng.randint(4, 8)
        cons.append(_c({f"x{i+1}": 1}, "<=", ub))
    for _ in range(rng.randint(1, 3)):
        op = rng.choice(["<=", "<=", ">="])
        coeffs = {f"x{i+1}": F(rng.randint(1, 6)) for i in range(n)}
        rhs = F(rng.randint(2, 12))
        cons.append(_c(coeffs, op, rhs))
    sense = rng.choice(["max", "min"])
    vars_ = [Variable(f"x{i+1}", "integer") for i in range(n)]
    bounds = {f"x{i+1}": (0, 8) for i in range(n)}
    return Problem(sense, obj, cons, vars_), bounds


# small helper used by test_11
def _as_lp_relaxation(self):
    from copy import deepcopy
    p = deepcopy(self)
    for v in p.variables:
        if v.vtype in ("integer", "binary"):
            v.vtype = "nonneg"
    return p
Problem.as_lp_relaxation = _as_lp_relaxation  # type: ignore[attr-defined]

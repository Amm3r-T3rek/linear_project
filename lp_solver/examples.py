"""All worked examples from the course PDFs plus O1-O4.

Each example is a :class:`Problem` (or a small dict describing the expected
answer) and is referenced both by the test-suite and by the Streamlit UI's
"Examples" dropdown.
"""

from __future__ import annotations

from fractions import Fraction

from solver.models import Problem, Constraint, Variable

F = Fraction


def _c(coeffs, op, rhs):
    return Constraint({k: F(v) for k, v in coeffs.items()}, op, F(rhs))


# --------------------------------------------------------------------------- #
#  Test-case 1: plain Simplex
# --------------------------------------------------------------------------- #
EX1 = Problem(
    sense="max",
    objective={"x1": F(3), "x2": F(2)},
    constraints=[
        _c({"x1": 1, "x2": 2}, "<=", 6),
        _c({"x1": 2, "x2": 1}, "<=", 8),
        _c({"x1": -1, "x2": 1}, "<=", 1),
        _c({"x2": 1}, "<=", 2),
    ],
    variables=[Variable("x1"), Variable("x2")],
)
EX1_EXPECTED = {"x1": F(10, 3), "x2": F(4, 3), "Z": F(38, 3)}

# --------------------------------------------------------------------------- #
#  Test-case 2 / 3: Big-M and Two-Phase on the same LP
# --------------------------------------------------------------------------- #
EX2 = Problem(
    sense="min",
    objective={"x1": F(4), "x2": F(1)},
    constraints=[
        _c({"x1": 3, "x2": 1}, "=", 3),
        _c({"x1": 4, "x2": 3}, ">=", 6),
        _c({"x1": 1, "x2": 2}, "<=", 4),
    ],
    variables=[Variable("x1"), Variable("x2")],
)
EX2_EXPECTED = {"x1": F(2, 5), "x2": F(9, 5), "Z": F(17, 5)}

# --------------------------------------------------------------------------- #
#  Test-case 4: Two-Phase
# --------------------------------------------------------------------------- #
EX4 = Problem(
    sense="max",
    objective={"x1": F(2), "x2": F(1)},
    constraints=[
        _c({"x1": 1, "x2": -2}, "<=", 2),
        _c({"x1": -1, "x2": 1}, "<=", 2),
        _c({"x1": 1, "x2": 2}, "=", 7),
        _c({"x1": 1, "x2": 1}, ">=", 2),
    ],
    variables=[Variable("x1"), Variable("x2")],
)
EX4_EXPECTED = {"x1": F(9, 2), "x2": F(5, 4), "Z": F(41, 4)}

# --------------------------------------------------------------------------- #
#  Test-case 5: max x1 s.t. x1 + 2 x2 = 4   -> (4, 0), Z = 4
# --------------------------------------------------------------------------- #
EX5 = Problem(
    sense="max",
    objective={"x1": F(1)},
    constraints=[_c({"x1": 1, "x2": 2}, "=", 4)],
    variables=[Variable("x1"), Variable("x2")],
)
EX5_EXPECTED = {"x1": F(4), "x2": F(0), "Z": F(4)}

# --------------------------------------------------------------------------- #
#  Test-case 6: infeasible (Phase I r = 4)
# --------------------------------------------------------------------------- #
EX6 = Problem(
    sense="max",
    objective={"x1": F(3), "x2": F(2)},
    constraints=[
        _c({"x1": 2, "x2": 1}, "<=", 2),
        _c({"x1": 3, "x2": 4}, ">=", 12),
    ],
    variables=[Variable("x1"), Variable("x2")],
)
EX6_EXPECTED = {"status": "infeasible", "r": F(4)}

# --------------------------------------------------------------------------- #
#  Test-case 8: unbounded
# --------------------------------------------------------------------------- #
EX8 = Problem(
    sense="max",
    objective={"x1": F(2), "x2": F(1)},
    constraints=[
        _c({"x1": 1, "x2": -1}, "<=", 10),
        _c({"x1": 2}, "<=", 40),
    ],
    variables=[Variable("x1"), Variable("x2")],
)
EX8_EXPECTED = {"status": "unbounded"}

# --------------------------------------------------------------------------- #
#  O1: Big-M   max 2x1+3x2 ; x1+2x2<=4 ; x1+x2=3
# --------------------------------------------------------------------------- #
O1 = Problem(
    sense="max",
    objective={"x1": F(2), "x2": F(3)},
    constraints=[
        _c({"x1": 1, "x2": 2}, "<=", 4),
        _c({"x1": 1, "x2": 1}, "=", 3),
    ],
    variables=[Variable("x1"), Variable("x2")],
)

# --------------------------------------------------------------------------- #
#  O2: Two-Phase  min 2x1+3x2+x3 ;
#       x1+4x2+2x3>=8 ; 3x1+2x2>=6
# --------------------------------------------------------------------------- #
O2 = Problem(
    sense="min",
    objective={"x1": F(2), "x2": F(3), "x3": F(1)},
    constraints=[
        _c({"x1": 1, "x2": 4, "x3": 2}, ">=", 8),
        _c({"x1": 3, "x2": 2}, ">=", 6),
    ],
    variables=[Variable("x1"), Variable("x2"), Variable("x3")],
)

# --------------------------------------------------------------------------- #
#  O3: infeasible  max 5x1+4x2 ; 3x1+2x2<=6 ; 2x1-x2>=6
# --------------------------------------------------------------------------- #
O3 = Problem(
    sense="max",
    objective={"x1": F(5), "x2": F(4)},
    constraints=[
        _c({"x1": 3, "x2": 2}, "<=", 6),
        _c({"x1": 2, "x2": -1}, ">=", 6),
    ],
    variables=[Variable("x1"), Variable("x2")],
)

# --------------------------------------------------------------------------- #
#  O4: infeasible  max 4x1+5x2+3x3 ;
#       x1+x2+2x3>=20 ; 15x1+6x2-5x3<=50 ; x1+3x2+5x3<=30
# --------------------------------------------------------------------------- #
O4 = Problem(
    sense="max",
    objective={"x1": F(4), "x2": F(5), "x3": F(3)},
    constraints=[
        _c({"x1": 1, "x2": 1, "x3": 2}, ">=", 20),
        _c({"x1": 15, "x2": 6, "x3": -5}, "<=", 50),
        _c({"x1": 1, "x2": 3, "x3": 5}, "<=", 30),
    ],
    variables=[Variable("x1"), Variable("x2"), Variable("x3")],
)

# --------------------------------------------------------------------------- #
#  Test-case 10: Gomory cut   max 7x+9y ; -x+3y<=6 ; 7x+y<=35  (integers)
#     LP optimum (9/2, 7/2) Z=63 ; integer (4,3) Z=55 ;
#     first cut  -(7/22)x1-(1/22)x2 <= -1/2
# --------------------------------------------------------------------------- #
EX10 = Problem(
    sense="max",
    objective={"x": F(7), "y": F(9)},
    constraints=[
        _c({"x": -1, "y": 3}, "<=", 6),
        _c({"x": 7, "y": 1}, "<=", 35),
    ],
    variables=[Variable("x", "integer"), Variable("y", "integer")],
)
EX10_LP = {"x": F(9, 2), "y": F(7, 2), "Z": F(63)}
EX10_INT = {"x": F(4), "y": F(3), "Z": F(55)}

# --------------------------------------------------------------------------- #
#  Test-case 11: ILP rounding trap
#     max x1+5x2 ; x1+10x2<=20 ; x1<=2  (integers)
#     LP (2, 1.8) Z=11 ; integer (0, 2) Z=10
# --------------------------------------------------------------------------- #
EX11 = Problem(
    sense="max",
    objective={"x1": F(1), "x2": F(5)},
    constraints=[
        _c({"x1": 1, "x2": 10}, "<=", 20),
        _c({"x1": 1}, "<=", 2),
    ],
    variables=[Variable("x1", "integer"), Variable("x2", "integer")],
)
EX11_LP = {"x1": F(2), "x2": F(9, 5), "Z": F(11)}
EX11_INT = {"x1": F(0), "x2": F(2), "Z": F(10)}

# --------------------------------------------------------------------------- #
#  Test-case 12: Branch & Bound (notes p.76)
#     max 2x+4y ; 5x+9y<=40 ; 2y<=5 ; x,y>=0 integers
#     LP (7/2, 5/2) Z=17 ; integer (4,2) Z=16
# --------------------------------------------------------------------------- #
EX12 = Problem(
    sense="max",
    objective={"x": F(2), "y": F(4)},
    constraints=[
        _c({"x": 5, "y": 9}, "<=", 40),
        _c({"y": 2}, "<=", 5),
    ],
    variables=[Variable("x", "integer"), Variable("y", "integer")],
)
EX12_INT = {"x": F(4), "y": F(2), "Z": F(16)}

# --------------------------------------------------------------------------- #
#  Catalogue for the UI dropdown
# --------------------------------------------------------------------------- #
EXAMPLES = [
    ("Simplex - Example (notes p.18)", EX1, "simplex",
     {"x1": F(10, 3), "x2": F(4, 3), "Z": F(38, 3)}),
    ("Big-M - Example (notes p.20)", EX2, "big_m", EX2_EXPECTED),
    ("Two-Phase on Big-M LP", EX2, "two_phase", EX2_EXPECTED),
    ("Two-Phase - Example (notes p.24)", EX4, "two_phase", EX4_EXPECTED),
    ("max x1 ; x1+2x2=4", EX5, "two_phase", EX5_EXPECTED),
    ("Infeasible - Phase I r=4 (notes p.27)", EX6, "two_phase", EX6_EXPECTED),
    ("Unbounded (notes p.29)", EX8, "simplex", EX8_EXPECTED),
    ("O1 - Big-M", O1, "big_m", None),
    ("O2 - Two-Phase", O2, "two_phase", None),
    ("O3 - Infeasible", O3, "two_phase", {"status": "infeasible"}),
    ("O4 - Infeasible", O4, "two_phase", {"status": "infeasible"}),
    ("Gomory cut (notes p.72)", EX10, "gomory", EX10_INT),
    ("ILP rounding trap (notes p.65)", EX11, "gomory", EX11_INT),
    ("Branch & Bound (notes p.76)", EX12, "branch_bound", EX12_INT),
]

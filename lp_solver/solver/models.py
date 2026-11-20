"""Dataclasses describing an LP / ILP problem and the solver output."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional


@dataclass
class Variable:
    """A decision variable.

    ``vtype`` is one of:
      * ``"nonneg"``  -- x >= 0 (default)
      * ``"free"``    -- unrestricted in sign (x = x' - x'')
      * ``"leq0"``    -- x <= 0  (x = -x')
      * ``"integer"`` -- x >= 0 and integer
      * ``"binary"``  -- x in {0, 1}
    """

    name: str
    vtype: str = "nonneg"

    @property
    def is_integer(self) -> bool:
        return self.vtype in ("integer", "binary")

    @property
    def is_binary(self) -> bool:
        return self.vtype == "binary"


@dataclass
class Constraint:
    """A single linear constraint ``sum(coeffs[v] * v)  op  rhs``."""

    coeffs: dict[str, Fraction]
    op: str  # "<=", ">=", "="
    rhs: Fraction


@dataclass
class Problem:
    """A complete LP / ILP problem."""

    sense: str  # "max" or "min"
    objective: dict[str, Fraction]  # var name -> coefficient
    constraints: list[Constraint]
    variables: list[Variable]

    def var_names(self) -> list[str]:
        return [v.name for v in self.variables]

    def has_integrality(self) -> bool:
        return any(v.is_integer for v in self.variables)


# --------------------------------------------------------------------------- #
#  Solver result
# --------------------------------------------------------------------------- #
@dataclass
class VerificationRow:
    expr: str
    lhs: Fraction
    op: str
    rhs: Fraction
    slack: Fraction  # lhs - rhs (<=0 means satisfied for <=)
    satisfied: bool


@dataclass
class Result:
    status: str  # "optimal" | "infeasible" | "unbounded" | "alternative" | ...
    solution: dict[str, Fraction] = field(default_factory=dict)
    objective_value: Optional[Fraction] = None
    steps: list = field(default_factory=list)  # list[Step]
    verification: list[VerificationRow] = field(default_factory=list)
    bb_tree: Optional[dict] = None  # branch-and-bound tree
    notes: list[str] = field(default_factory=list)
    method: str = ""

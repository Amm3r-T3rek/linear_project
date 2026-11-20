"""Standardise an LP into equality form with nonnegative variables.

Course conventions:

  * ``<=``  ->  +S                       (slack)
  * ``=``   ->  +R                       (artificial)
  * ``>=``  ->  -S + R                   (surplus + artificial)
  * negative RHS  ->  multiply row by -1 and flip the operator
  * free variable x   ->  x = x' - x''   (both >= 0)
  * ``x <= 0`` variable ->  x = -x'      (x' >= 0)

The output :class:`StandardForm` is consumed by the simplex / Big-M / two-phase
engines.  ``recovery`` records, for each original variable, how to read its
value back out of the solved tableau.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

from .models import Problem, Variable
from .numbers import BigM


@dataclass
class StandardForm:
    sense: str  # original "max" / "min"
    columns: list[str]
    obj_coeffs: dict[str, BigM]  # original-objective coefficient per column
    body: list[list[BigM]]
    rhs: list[BigM]
    basis: list[str]
    artificial_names: set[str] = field(default_factory=set)
    slack_names: set[str] = field(default_factory=set)
    surplus_names: set[str] = field(default_factory=set)
    integer_names: set[str] = field(default_factory=set)
    binary_names: set[str] = field(default_factory=set)
    # recovery: for each original variable (in order), how to compute its value
    recovery: list[tuple] = field(default_factory=list)
    original_names: list[str] = field(default_factory=list)

    @property
    def n_original(self) -> int:
        return len(self.original_names)

    def artificial_cols(self) -> set[int]:
        return {i for i, c in enumerate(self.columns) if c in self.artificial_names}

    def integer_cols(self) -> set[int]:
        return {i for i, c in enumerate(self.columns) if c in self.integer_names}

    def recover(self, values: dict[str, object]) -> dict[str, Fraction]:
        """Turn a {column: value} dict into {original_name: Fraction}."""
        out: dict[str, Fraction] = {}
        for rec in self.recovery:
            if rec[0] == "nonneg":
                out[rec[1]] = _frac(values.get(rec[1], 0))
            elif rec[0] == "free":
                out[rec[1]] = _frac(values.get(rec[2], 0)) - _frac(values.get(rec[3], 0))
            elif rec[0] == "leq0":
                out[rec[1]] = -_frac(values.get(rec[2], 0))
        return out


def _frac(x) -> Fraction:
    if isinstance(x, BigM):
        return x.to_fraction()
    return Fraction(x)


def standardize(problem: Problem) -> StandardForm:
    """Convert ``problem`` to equality form with all variables nonnegative."""
    sense = problem.sense
    columns: list[str] = []
    obj_coeffs: dict[str, BigM] = {}
    recovery: list[tuple] = []
    original_names: list[str] = []
    integer_names: set[str] = set()
    binary_names: set[str] = set()

    # -- variables -------------------------------------------------------
    for v in problem.variables:
        original_names.append(v.name)
        if v.vtype in ("integer", "binary"):
            integer_names.add(v.name)
            if v.vtype == "binary":
                binary_names.add(v.name)
        if v.vtype == "free":
            p, n = f"{v.name}+", f"{v.name}-"
            columns.extend([p, n])
            obj_coeffs[p] = BigM(problem.objective.get(v.name, 0))
            obj_coeffs[n] = -BigM(problem.objective.get(v.name, 0))
            recovery.append(("free", v.name, p, n))
        elif v.vtype == "leq0":
            neg = f"{v.name}'"
            columns.append(neg)
            # x = -x', so objective coeff of x' is -c_x
            obj_coeffs[neg] = -BigM(problem.objective.get(v.name, 0))
            recovery.append(("leq0", v.name, neg))
        else:  # nonneg (or integer/binary treated as nonneg columns)
            columns.append(v.name)
            obj_coeffs[v.name] = BigM(problem.objective.get(v.name, 0))
            recovery.append(("nonneg", v.name))

    # -- constraints -----------------------------------------------------
    # First pass: collect all structural variable names (slacks / surplus /
    # artificials) so every body row can be built against the full column set.
    body: list[list[BigM]] = []
    rhs: list[BigM] = []
    basis: list[str] = []
    artificial_names: set[str] = set()
    slack_names: set[str] = set()
    surplus_names: set[str] = set()

    constraint_specs: list[tuple[str, Fraction, dict[str, Fraction], str, str]] = []
    for idx, con in enumerate(problem.constraints):
        op = con.op
        b = Fraction(con.rhs)
        coeffs = {k: Fraction(v) for k, v in con.coeffs.items()}
        if b < 0:
            b = -b
            coeffs = {k: -v for k, v in coeffs.items()}
            op = {"<=": ">=", ">=": "<=", "=": "="}[op]
        if op == "<=":
            s = f"S{idx + 1}"
            columns.append(s)
            slack_names.add(s)
            obj_coeffs[s] = BigM(0)
            basis.append(s)
            extra = {s: BigM(1)}
            basis_var = s
        elif op == ">=":
            s = f"S{idx + 1}"
            r = f"R{idx + 1}"
            columns.extend([s, r])
            surplus_names.add(s)
            artificial_names.add(r)
            obj_coeffs[s] = BigM(0)
            obj_coeffs[r] = BigM(0)
            basis.append(r)
            extra = {s: BigM(-1), r: BigM(1)}
            basis_var = r
        elif op == "=":
            r = f"R{idx + 1}"
            columns.append(r)
            artificial_names.add(r)
            obj_coeffs[r] = BigM(0)
            basis.append(r)
            extra = {r: BigM(1)}
            basis_var = r
        else:
            raise ValueError(f"Unknown operator {op!r}")
        constraint_specs.append((op, b, coeffs, extra, basis_var))

    # Second pass: build every body row against the final column list.
    col_index = {c: i for i, c in enumerate(columns)}
    for op, b, coeffs, extra, _basis in constraint_specs:
        row = [BigM(0)] * len(columns)
        for v in problem.variables:
            c = coeffs.get(v.name, 0)
            if c == 0:
                continue
            if v.vtype == "free":
                row[col_index[f"{v.name}+"]] = BigM(c)
                row[col_index[f"{v.name}-"]] = BigM(-c)
            elif v.vtype == "leq0":
                row[col_index[f"{v.name}'"]] = BigM(-c)
            else:
                row[col_index[v.name]] = BigM(c)
        for name, val in extra.items():
            row[col_index[name]] = val
        body.append(row)
        rhs.append(BigM(b))

    return StandardForm(
        sense=sense,
        columns=columns,
        obj_coeffs=obj_coeffs,
        body=body,
        rhs=rhs,
        basis=basis,
        artificial_names=artificial_names,
        slack_names=slack_names,
        surplus_names=surplus_names,
        integer_names=integer_names,
        binary_names=binary_names,
        recovery=recovery,
        original_names=original_names,
    )

"""Branch-and-bound for (mixed/pure/0-1) integer programs.

Algorithm (course notes, p.75):
  1. Relax integrality, solve the LP. If integer, done.
  2. Pick a fractional integer variable ``x_r = k``; branch into
     ``x_r <= floor(k)`` and ``x_r >= floor(k) + 1``.
  3. For each child solve the LP relaxation (with the extra bound).
  4. Fathom when the child is infeasible, integer, or its LP bound is no
     better than the incumbent.
  5. Stop when no node remains open; the incumbent is optimal.

A depth-first search with the ``<=`` (floor) child explored first is used, so
the first integer solution found tends to be the one reported in the notes.
A ``node_limit`` guards against pathological growth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

from .models import Problem, Constraint, Result, VerificationRow
from .simplex import solve_lp, verify
from .standardize import StandardForm
from .steps import Step, StepLog
from .tableau import Tableau


@dataclass
class BBNode:
    node_id: int
    parent_id: Optional[int]
    branch_desc: str  # human-readable, e.g. "y <= 2"
    extra_constraints: list[Constraint] = field(default_factory=list)
    lp_value: Optional[Fraction] = None
    lp_solution: dict[str, Fraction] = field(default_factory=dict)
    status: str = "open"  # open | optimal | infeasible | unbounded | fathomed
    fathom_reason: str = ""
    children: list[int] = field(default_factory=list)
    is_integer: bool = False


def solve_branch_bound(problem: Problem, node_limit: int = 1000,
                       max_iter: int = 1000) -> Result:
    log = StepLog()
    log.add(Step("Branch-and-Bound",
                 f"Sense = {problem.sense}. Relax integrality and solve the "
                 f"LP; branch on fractional integer variables; fathom by "
                 f"infeasibility / integrality / bound. DFS with the '<=' "
                 f"(floor) child explored first.", []))

    sense = problem.sense
    int_names = [v.name for v in problem.variables if v.is_integer]

    if not int_names:
        # no integrality -> just solve the LP
        tab, sf, status, sublog = solve_lp(problem, max_iter=max_iter)
        log.steps.extend(sublog.steps)
        from .simplex import _finalise
        return _finalise(tab, sf, status, log, method="Branch-and-Bound",
                         problem=problem)

    # incumbent initialisation
    if sense == "max":
        best_val: Optional[Fraction] = None  # we want the largest
        better = lambda a, b: b is None or a > b
    else:
        best_val = None
        better = lambda a, b: b is None or a < b
    best_sol: Optional[dict[str, Fraction]] = None

    nodes: dict[int, BBNode] = {}
    root = BBNode(0, None, "root", [])
    nodes[0] = root
    stack: list[int] = [0]
    counter = 1

    while stack and counter < node_limit + 1:
        nid = stack.pop()
        node = nodes[nid]
        sub = _with_constraints(problem, node.extra_constraints)
        tab, sf, status, _ = solve_lp(sub, max_iter=max_iter)

        if status == "infeasible":
            node.status = "infeasible"
            node.fathom_reason = "LP relaxation infeasible"
            log.add(Step(f"Node {nid}", f"Branch {node.branch_desc}: LP "
                         f"infeasible -> fathom.", []))
            continue
        if status == "unbounded":
            node.status = "unbounded"
            node.fathom_reason = "LP relaxation unbounded"
            continue
        if status not in ("optimal", "alternative"):
            node.status = "fathomed"
            node.fathom_reason = f"LP status {status}"
            continue

        lp_val = tab.z_rhs.to_fraction()
        node.lp_value = lp_val
        node.lp_solution = sf.recover(tab.basic_solution())

        # bound fathom
        if best_val is not None:
            if sense == "max" and lp_val <= best_val:
                node.status = "fathomed"
                node.fathom_reason = f"LP bound {lp_val} <= incumbent {best_val}"
                log.add(Step(f"Node {nid}", f"Branch {node.branch_desc}: LP "
                             f"value {lp_val} <= incumbent {best_val} -> "
                             f"fathom by bound.", []))
                continue
            if sense == "min" and lp_val >= best_val:
                node.status = "fathomed"
                node.fathom_reason = f"LP bound {lp_val} >= incumbent {best_val}"
                log.add(Step(f"Node {nid}", f"Branch {node.branch_desc}: LP "
                             f"value {lp_val} >= incumbent {best_val} -> "
                             f"fathom by bound.", []))
                continue

        # integrality test
        if _all_integer(node.lp_solution, int_names, problem):
            node.is_integer = True
            node.status = "optimal"
            node.fathom_reason = "integer solution"
            if better(lp_val, best_val):
                best_val = lp_val
                best_sol = dict(node.lp_solution)
                log.add(Step(f"Node {nid} - new incumbent",
                             f"Branch {node.branch_desc}: integer solution "
                             f"{_fmt_sol(node.lp_solution)} with Z = {lp_val} "
                             f"-> new incumbent.", []))
            else:
                log.add(Step(f"Node {nid}", f"Branch {node.branch_desc}: "
                             f"integer solution Z = {lp_val} not better than "
                             f"incumbent {best_val} -> fathom.", []))
            continue

        # branch
        var, val = _pick_branch_var(node.lp_solution, int_names, problem)
        floor_v = math.floor(val)
        lo_val = Fraction(floor_v)
        hi_val = Fraction(floor_v + 1)
        left = BBNode(counter, nid, f"{var} <= {floor_v}",
                      node.extra_constraints + [Constraint({var: Fraction(1)}, "<=", lo_val)])
        counter += 1
        right = BBNode(counter, nid, f"{var} >= {floor_v + 1}",
                       node.extra_constraints + [Constraint({var: Fraction(1)}, ">=", hi_val)])
        counter += 1
        node.children = [left.node_id, right.node_id]
        nodes[left.node_id] = left
        nodes[right.node_id] = right
        log.add(Step(f"Node {nid} - branch",
                     f"LP value {lp_val}, solution {_fmt_sol(node.lp_solution)}. "
                     f"Branch on {var} = {val}: {var} <= {floor_v}  /  "
                     f"{var} >= {floor_v + 1}.", []))
        # push right first so left (<=) is explored first
        stack.append(right.node_id)
        stack.append(left.node_id)

    if best_sol is None:
        log.add(Step("No integer solution", "Branch-and-bound found no "
                     "feasible integer solution.", []))
        return Result(status="infeasible", solution={}, objective_value=None,
                      steps=log.steps, verification=verify(problem, {}),
                      method="Branch-and-Bound")

    tree = _serialise_tree(nodes)
    log.add(Step("Optimal integer solution",
                 f"Best integer solution found: {_fmt_sol(best_sol)} with "
                 f"Z = {best_val}.", [],
                 notes=[f"Explored {len(nodes)} nodes."]))
    res = Result(
        status="optimal",
        solution=best_sol,
        objective_value=best_val,
        steps=log.steps,
        verification=verify(problem, best_sol),
        method="Branch-and-Bound",
    )
    res.bb_tree = tree
    return res


# --------------------------------------------------------------------------- #
def _with_constraints(problem: Problem, extra: list[Constraint]) -> Problem:
    return Problem(
        sense=problem.sense,
        objective=dict(problem.objective),
        constraints=list(problem.constraints) + list(extra),
        variables=list(problem.variables),
    )


def _all_integer(sol: dict[str, Fraction], int_names: list[str],
                 problem: Problem) -> bool:
    binary = {v.name for v in problem.variables if v.is_binary}
    for n in int_names:
        v = sol.get(n, Fraction(0))
        if v.denominator != 1:
            return False
        if n in binary and v not in (0, 1):
            return False
    return True


def _pick_branch_var(sol: dict[str, Fraction], int_names: list[str],
                     problem: Problem) -> tuple[str, Fraction]:
    """Pick the integer variable whose fractional part is closest to 1/2
    (ties: lowest index).  This is the standard 'most fractional' rule."""
    best = None
    best_dist = None
    for n in int_names:
        v = sol.get(n, Fraction(0))
        if v.denominator == 1:
            continue
        f = v - math.floor(v)
        d = abs(f - Fraction(1, 2))
        if best is None or d < best_dist:
            best = n
            best_dist = d
    if best is None:
        # fall back to first fractional
        for n in int_names:
            v = sol.get(n, Fraction(0))
            if v.denominator != 1:
                return n, v
    return best, sol[best]


def _fmt_sol(sol: dict[str, Fraction]) -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(sol.items()))


def _serialise_tree(nodes: dict[int, BBNode]) -> dict:
    return {
        "nodes": [
            {
                "id": n.node_id,
                "parent": n.parent_id,
                "branch": n.branch_desc,
                "lp_value": str(n.lp_value) if n.lp_value is not None else None,
                "lp_solution": {k: str(v) for k, v in n.lp_solution.items()},
                "status": n.status,
                "fathom_reason": n.fathom_reason,
                "children": n.children,
                "is_integer": n.is_integer,
            }
            for n in sorted(nodes.values(), key=lambda x: x.node_id)
        ]
    }

"""Serializable step log.

Every solver method appends :class:`Step` objects to a list.  Each step carries
one or more :class:`TableauSnapshot` instances (most steps carry exactly one)
together with a plain-English description.  The snapshots store *string*
representations of the numbers so that the log is JSON-serialisable and can be
re-rendered with any number-formatting mode (fractions / decimals).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TableauSnapshot:
    """A frozen view of a tableau at one point in time.

    * ``columns`` -- header labels of every tableau column (variables + RHS).
    * ``basis``   -- label of the basic variable for each constraint row.
    * ``z_row``   -- the objective (Z or r) row, as strings.
    * ``rows``    -- constraint rows, each a list of strings (no RHS).
    * ``rhs``     -- the constant column for every row (Z first, then basis).
    * ``ratios``  -- ratio for each constraint row; ``""`` for the Z row.
    * ``entering_col`` / ``leaving_row`` -- 0-based indices into ``columns``-1
      and the constraint-row block; ``None`` when not applicable.
    * ``entering_name`` / ``leaving_name`` -- human-readable labels.
    * ``highlight`` -- extra cells to highlight, e.g. the pivot cell.
    """

    columns: list[str]
    basis: list[str]
    z_row: list[str]
    rows: list[list[str]]
    rhs: list[str]
    ratios: list[str]
    objective_label: str = "Z"
    entering_col: Optional[int] = None
    leaving_row: Optional[int] = None
    entering_name: Optional[str] = None
    leaving_name: Optional[str] = None
    pivot_cell: Optional[tuple[int, int]] = None  # (constraint_row_index, col_index)
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "columns": self.columns,
            "basis": self.basis,
            "z_row": self.z_row,
            "rows": self.rows,
            "rhs": self.rhs,
            "ratios": self.ratios,
            "objective_label": self.objective_label,
            "entering_col": self.entering_col,
            "leaving_row": self.leaving_row,
            "entering_name": self.entering_name,
            "leaving_name": self.leaving_name,
            "pivot_cell": self.pivot_cell,
            "reason": self.reason,
        }


@dataclass
class Step:
    """A titled, narrated group of one or more tableau snapshots."""

    title: str
    description: str
    snapshots: list[TableauSnapshot] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "description": self.description,
            "snapshots": [s.to_dict() for s in self.snapshots],
            "notes": self.notes,
        }


class StepLog:
    """A simple append-only container of :class:`Step` objects."""

    def __init__(self) -> None:
        self.steps: list[Step] = []

    def add(self, step: Step) -> None:
        self.steps.append(step)

    def snapshot(self, title: str, description: str, snap: TableauSnapshot,
                notes: Optional[list[str]] = None) -> None:
        self.steps.append(Step(title, description, [snap], notes or []))

    def to_list(self) -> list[dict]:
        return [s.to_dict() for s in self.steps]

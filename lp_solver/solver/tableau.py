"""The tableau and the pivot operation.

A :class:`Tableau` stores the objective row (``z_row`` + ``z_rhs``) and the
constraint body (``body`` + ``rhs``) using :class:`BigM` entries, so the same
engine handles the plain Simplex (pure rationals) and the Big-M method
(symbolic ``M``).

Course conventions (see NOTES.md):
  * ``<=``  ->  +S  (slack)
  * ``=``   ->  +R  (artificial)
  * ``>=``  ->  -S + R  (surplus + artificial)
  * Entering variable = most-negative Z-row coefficient (max form); for a
    ``min`` problem the row is stored so that the entering variable is the
    most-positive coefficient (this makes the displayed tableau identical to
    the course PDFs).
  * Leaving variable = smallest positive ratio (ties: lowest index).
  * Bland's rule is used as an anti-cycling fallback.
"""

from __future__ import annotations

from typing import Optional

from .numbers import BigM, to_bigm
from .steps import TableauSnapshot


class Tableau:
    """A simplex tableau with exact :class:`BigM` arithmetic."""

    def __init__(
        self,
        columns: list[str],
        basis: list[str],
        z_row: list[BigM],
        z_rhs: BigM,
        body: list[list[BigM]],
        rhs: list[BigM],
        sense: str = "max",
        obj_label: str = "Z",
        artificial_cols: Optional[set[int]] = None,
        integer_cols: Optional[set[int]] = None,
    ) -> None:
        assert sense in ("max", "min")
        self.columns = list(columns)
        self.basis = list(basis)
        self.z_row = [to_bigm(x) for x in z_row]
        self.z_rhs = to_bigm(z_rhs)
        self.body = [[to_bigm(x) for x in row] for row in body]
        self.rhs = [to_bigm(x) for x in rhs]
        self.sense = sense
        self.obj_label = obj_label
        self.artificial_cols = set(artificial_cols or ())
        self.integer_cols = set(integer_cols or ())
        # the variable index of every column
        self._col_index = {name: i for i, name in enumerate(self.columns)}

    # ------------------------------------------------------------------ #
    #  Basic queries
    # ------------------------------------------------------------------ #
    @property
    def num_cols(self) -> int:
        return len(self.columns)

    @property
    def num_rows(self) -> int:
        return len(self.basis)

    def col_index(self, name: str) -> int:
        return self._col_index[name]

    def is_basic(self, col: int) -> bool:
        return self.columns[col] in self.basis

    def basic_row_of(self, col: int) -> Optional[int]:
        """Return the body-row index where ``columns[col]`` is basic, or None."""
        name = self.columns[col]
        for i, b in enumerate(self.basis):
            if b == name:
                return i
        return None

    # ------------------------------------------------------------------ #
    #  Entering / leaving selection
    # ------------------------------------------------------------------ #
    def entering_col(self, forbidden: Optional[set[int]] = None,
                     bland: bool = False) -> Optional[int]:
        """Return the entering column index, or ``None`` if optimal.

        ``forbidden`` columns (e.g. artificials in phase II) are never chosen.
        When ``bland`` is true we use Bland's rule: the eligible column with
        the lowest index.
        """
        forbidden = forbidden or set()
        best_idx: Optional[int] = None
        best_val: Optional[BigM] = None
        for j in range(self.num_cols):
            if j in forbidden:
                continue
            zj = self.z_row[j]
            if self.sense == "max":
                # most negative
                if zj < 0:
                    if bland:
                        return j
                    if best_val is None or zj < best_val or (
                        zj == best_val and j < best_idx  # type: ignore[operator]
                    ):
                        best_val = zj
                        best_idx = j
            else:  # min
                # most positive
                if zj > 0:
                    if bland:
                        return j
                    if best_val is None or zj > best_val or (
                        zj == best_val and j < best_idx  # type: ignore[operator]
                    ):
                        best_val = zj
                        best_idx = j
        return best_idx

    def is_optimal(self, forbidden: Optional[set[int]] = None) -> bool:
        return self.entering_col(forbidden) is None

    def ratios(self, entering: int) -> list[Optional[BigM]]:
        """Ratio ``rhs[i] / body[i][entering]`` for each body row (None if not
        eligible: denominator <= 0)."""
        out: list[Optional[BigM]] = []
        for i in range(self.num_rows):
            a = self.body[i][entering]
            if a > 0:
                out.append(self.rhs[i] / a)
            else:
                out.append(None)
        return out

    def leaving_row(self, entering: int) -> Optional[int]:
        """Smallest positive ratio; ties broken by lowest index."""
        best_idx: Optional[int] = None
        best_ratio: Optional[BigM] = None
        for i in range(self.num_rows):
            a = self.body[i][entering]
            if a > 0:
                r = self.rhs[i] / a
                if best_ratio is None or r < best_ratio or (
                    r == best_ratio and i < best_idx  # type: ignore[operator]
                ):
                    best_ratio = r
                    best_idx = i
        return best_idx

    # ------------------------------------------------------------------ #
    #  Pivot
    # ------------------------------------------------------------------ #
    def pivot(self, entering: int, leaving: int) -> None:
        """Perform one pivot: ``columns[entering]`` enters the basis, the
        variable basic in body-row ``leaving`` leaves."""
        pivot_el = self.body[leaving][entering]
        if pivot_el == 0:
            raise ValueError("Pivot element is zero")
        # normalise pivot row
        self.body[leaving] = [x / pivot_el for x in self.body[leaving]]
        self.rhs[leaving] = self.rhs[leaving] / pivot_el
        # eliminate entering column from every other row, incl. z_row
        fz = self.z_row[entering]
        if fz != 0:
            self.z_row = [self.z_row[j] - fz * self.body[leaving][j]
                          for j in range(self.num_cols)]
            self.z_rhs = self.z_rhs - fz * self.rhs[leaving]
        for i in range(self.num_rows):
            if i == leaving:
                continue
            fi = self.body[i][entering]
            if fi == 0:
                continue
            self.body[i] = [self.body[i][j] - fi * self.body[leaving][j]
                            for j in range(self.num_cols)]
            self.rhs[i] = self.rhs[i] - fi * self.rhs[leaving]
        self.basis[leaving] = self.columns[entering]

    # ------------------------------------------------------------------ #
    #  Solution extraction
    # ------------------------------------------------------------------ #
    def basic_solution(self) -> dict[str, BigM]:
        """Return {column_name: value} for every column; nonbasics are 0."""
        sol = {name: BigM(0) for name in self.columns}
        for i, name in enumerate(self.basis):
            sol[name] = self.rhs[i]
        return sol

    def solution_for(self, names: list[str]) -> dict[str, "object"]:
        """Values of the given (original) variable names, as Fractions when
        possible."""
        sol = self.basic_solution()
        out: dict[str, object] = {}
        for n in names:
            v = sol.get(n, BigM(0))
            try:
                out[n] = v.to_fraction()
            except ValueError:
                out[n] = v  # keep BigM if M-part nonzero (shouldn't happen at opt)
        return out

    def objective_value(self):
        return self.z_rhs

    # ------------------------------------------------------------------ #
    #  Snapshot
    # ------------------------------------------------------------------ #
    def snapshot(
        self,
        reason: str = "",
        entering: Optional[int] = None,
        leaving: Optional[int] = None,
        forbidden: Optional[set[int]] = None,
        fmt: str = "frac",
    ) -> TableauSnapshot:
        """Build an immutable :class:`TableauSnapshot` (string-valued)."""
        forbidden = forbidden or set()
        cols = list(self.columns) + ["RHS"]
        z_row = [self.z_row[j].format(fmt) for j in range(self.num_cols)]
        z_rhs = self.z_rhs.format(fmt)
        rows = []
        rhs_col = []
        for i in range(self.num_rows):
            rows.append([self.body[i][j].format(fmt) for j in range(self.num_cols)])
            rhs_col.append(self.rhs[i].format(fmt))
        # ratios (only meaningful when an entering column is chosen)
        ratios = [""]  # z-row
        if entering is not None:
            for i in range(self.num_rows):
                a = self.body[i][entering]
                if a > 0:
                    ratios.append((self.rhs[i] / a).format(fmt))
                else:
                    ratios.append("—")
        else:
            ratios.extend([""] * self.num_rows)

        entering_name = self.columns[entering] if entering is not None else None
        leaving_name = self.basis[leaving] if leaving is not None else None
        pivot_cell = (leaving, entering) if (entering is not None and leaving is not None) else None

        snap = TableauSnapshot(
            columns=cols,
            basis=list(self.basis),
            z_row=z_row + [z_rhs],
            rows=[r + [rhs_col[i]] for i, r in enumerate(rows)],
            rhs=rhs_col,
            ratios=ratios,
            objective_label=self.obj_label,
            entering_col=entering,
            leaving_row=leaving,
            entering_name=entering_name,
            leaving_name=leaving_name,
            pivot_cell=pivot_cell,
            reason=reason,
        )
        return snap

    # ------------------------------------------------------------------ #
    #  Objective-row cleansing
    # ------------------------------------------------------------------ #
    def cleanse_objective_row(self) -> None:
        """Substitute the basic variables out of the objective row so that
        every basic column is a unit vector in ``z_row`` as well.

        For each basic variable sitting in body-row ``r`` with column ``k``,
        we perform ``z_row -= z_row[k] * body[r]`` (and the same for ``z_rhs``).
        This is exactly the algebraic "express Z in terms of the nonbasic
        variables" step and is needed whenever the initial basis contains
        variables with a nonzero objective coefficient (e.g. Big-M artificials
        or the two-phase ``r`` objective).
        """
        for r, name in enumerate(self.basis):
            k = self._col_index[name]
            zk = self.z_row[k]
            if zk != 0:
                self.z_row = [self.z_row[j] - zk * self.body[r][j]
                               for j in range(self.num_cols)]
                self.z_rhs = self.z_rhs - zk * self.rhs[r]

    # ------------------------------------------------------------------ #
    #  Adding cut rows (Gomory / mixed cut)
    # ------------------------------------------------------------------ #
    def add_cut_row(self, coeffs: list[BigM], rhs: BigM, slack_name: str) -> int:
        """Append a new constraint row ``coeffs . x + slack = rhs`` and add a
        fresh column for ``slack`` (coefficient 1 in the new row, 0 elsewhere,
        objective coefficient 0).  Returns the body-row index of the new row.
        """
        assert len(coeffs) == self.num_cols
        # extend every existing row with a 0 in the new slack column
        new_col_idx = self.num_cols
        self.columns.append(slack_name)
        self._col_index[slack_name] = new_col_idx
        for i in range(self.num_rows):
            self.body[i].append(BigM(0))
        self.z_row.append(BigM(0))
        # the new row: coeffs + [1] for the slack
        new_body = list(coeffs) + [BigM(1)]
        self.body.append(new_body)
        self.rhs.append(to_bigm(rhs))
        self.basis.append(slack_name)
        return self.num_rows - 1

    # ------------------------------------------------------------------ #
    #  Diagnostics
    # ------------------------------------------------------------------ #
    def has_alternative_optimum(self, forbidden: Optional[set[int]] = None) -> bool:
        """Optimal but some nonbasic has a zero reduced cost (could re-enter)."""
        if not self.is_optimal(forbidden):
            return False
        forbidden = forbidden or set()
        for j in range(self.num_cols):
            if j in forbidden or self.is_basic(j):
                continue
            if self.z_row[j] == 0:
                # would change nothing -> alternative optimum exists only if
                # the column can actually move (has a positive entry)
                for i in range(self.num_rows):
                    if self.body[i][j] > 0:
                        return True
        return False

    def is_degenerate(self) -> bool:
        """A basic variable sitting at zero -> degenerate vertex."""
        return any(r == 0 for r in self.rhs)

# LP / ILP Solver — Pure Python

An exact-arithmetic Linear & Integer Programming solver with a Streamlit web UI.
Implements the **Simplex method**, **Big-M (penalty) method**, **Two-Phase
method**, **Dual Simplex**, **Gomory fractional cut**, **Mixed-integer cut**,
and **Branch-and-Bound**, together with a full step-by-step teaching log.

All arithmetic uses `fractions.Fraction` (and a symbolic `BigM` class for the
penalty constant `M`), so **no rounding ever occurs** — every answer is exact.

---

## Quick start (3 commands — Windows or any OS)

```bat
pip install -r requirements.txt
pytest -q
streamlit run app.py
```

On Windows save the three lines above as commands in a terminal
(`cmd` or PowerShell).  `run.bat` does the install + launch in one go:

```bat
run.bat
```

---

## Project layout

```
lp_solver/
  app.py                  Streamlit entry point  (streamlit run app.py)
  examples.py             all worked examples from the PDFs + O1-O4
  requirements.txt
  run.sh / run.bat        install + launch helpers
  README.md
  NOTES.md                course conventions + PDF discrepancies
  solver/                 pure-logic package (no Streamlit dependency)
    numbers.py            BigM class (a + bM, lexicographic compare)
    models.py             Problem / Constraint / Variable / Result dataclasses
    standardize.py        <=, >=, = handling; negative RHS; free / <=0 vars
    tableau.py            Tableau class + pivot + ratio column + cut-row add
    steps.py              serialisable step log (TableauSnapshot / Step)
    simplex.py            plain Simplex + shared engine + LP relaxation
    big_m.py              Big-M method (symbolic M)
    two_phase.py          Phase I (min r = sum R) + Phase II
    dual_simplex.py       dual simplex (for cutting planes)
    gomory.py             Gomory fractional cut (pure ILP)
    mixed_cut.py          mixed-integer cut
    branch_bound.py       B&B (max/min, pure/mixed, 0-1) + tree log
  ui/                     Streamlit components
    input_panel.py        dynamic problem builder
    tableau_view.py       highlighted tableau rendering (pandas Styler)
    graph.py              2-variable Plotly graph (region, lines, optimum,
                          integer lattice)
    bb_tree.py            branch-and-bound tree visualisation
    export.py             Markdown / JSON / PDF export
  tests/
    test_solver.py        12 worked cases + randomized vs scipy / brute force
```

The `solver/` package is **independent of Streamlit** — it returns a serialisable
step log and can be used from any Python script or notebook.

---

## Algorithms (course conventions)

| Step | Rule |
|------|------|
| `<=` constraint | add slack **S** |
| `=`  constraint | add artificial **R** |
| `>=` constraint | subtract surplus **S**, add artificial **R** |
| negative RHS | multiply row by -1, flip the operator |
| free variable `x` | substitute `x = x' - x''` (both >= 0) |
| `x <= 0` variable | substitute `x = -x'` (x' >= 0) |
| entering variable (max) | most-negative Z-row coefficient |
| entering variable (min) | most-positive Z-row coefficient |
| leaving variable | smallest positive ratio `b_i / a_ik` (ties: lowest index) |
| anti-cycling | Bland's rule fallback on repeated basis |
| optimality (max) | all nonbasic Z-row coefficients >= 0 |
| optimality (min) | all nonbasic Z-row coefficients <= 0 |

**Big-M**: artificials are penalised with `+M` (min) / `-M` (max) in the
objective, where `M` is the symbolic `BigM(0, 1)`.  Comparison is lexicographic
on the M-coefficient, so `7M - 4 > 4M - 1` is decided exactly without picking a
numeric value for `M`.

**Two-Phase**: Phase I minimises `r = sum of artificials`; `r > 0` ⇒
infeasible.  Phase II reuses the Phase-I basis with the real objective and
forbids the artificial columns.

**Gomory cut**: from a basic integer variable `x_Bi + sum a_ij x_j = b_i`,
let `f_ij = a_ij - floor(a_ij)`, `f_i = b_i - floor(b_i)`.  The cut is
`sum f_ij x_j >= f_i`, added as `-sum f_ij x_j + s = -f_i` and re-optimised
with the dual simplex.

**Mixed cut**: for an integer `x_k = d_k`, `f_k = d_k - floor(d_k)`,
`J+ = {a_kj > 0}`, `J- = {a_kj < 0}`:
`sum_{J-} (f_k/(1-f_k)) a_kj x_j - sum_{J+} a_kj x_j <= -f_k`.

**Branch-and-Bound**: relax integrality, solve the LP, branch on the most
fractional integer variable into `x <= floor(k)` / `x >= floor(k)+1`.  DFS
with the floor child first.  Fathom by infeasibility, integrality, or bound.
A node limit guards against pathological growth.

---

## Status detection

The solver reports one of:

* **optimal** — unique optimum reached
* **alternative** — optimal but a nonbasic variable has a zero reduced cost
  (alternative optima exist)
* **infeasible** — Phase-I `r > 0` (or a Big-M artificial stays positive)
* **unbounded** — entering column has no positive entry
* **degenerate** — flagged when a basic variable sits at 0

A **final verification** table plugs the returned `x*` back into every original
constraint and shows the slack (`LHS - RHS`) and a PASS/FAIL mark.

---

## Tests

```
pytest -q
```

runs the 12 worked test cases (exact fractional answers) plus randomized
property tests that cross-check the simplex family against
`scipy.optimize.linprog` and the integer solvers against brute-force
enumeration.  30 tests, all passing.

---

## Export

From the UI: **Markdown**, **JSON**, and **PDF** (reportlab) downloads of the
full solution (problem, every tableau, verification).

---

## Architecture notes

* `BigM(a, b) = a + b*M` carries `M` symbolically.  Because constraint rows in
  a Big-M tableau never carry an M-term, `BigM * BigM` is only ever invoked
  with one factor M-free — the class raises if `M^2` would arise, which
  guards against programming mistakes.
* The same `run_simplex` engine drives the plain Simplex, Big-M, and both
  phases of the two-phase method: only the initial tableau setup differs.
* The step log stores **strings** (already formatted), so it is JSON-
  serialisable and can be re-rendered in fraction or decimal mode without
  re-running the solver.

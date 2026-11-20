# NOTES — course conventions and PDF discrepancies

## Conventions followed (from the course PDFs)

### Standardisation
* `<=`  →  `+ S`            (slack, becomes the initial basic variable)
* `=`   →  `+ R`            (artificial, needs Big-M / two-phase)
* `>=`  →  `- S + R`        (surplus + artificial)
* A constraint with a **negative RHS** is multiplied by `-1` and the operator
  flipped (`<=`↔`>=`, `=` stays `=`) before a slack/surplus/artificial is
  added.
* A **free** variable `x` is replaced by `x = x' - x''` with `x', x'' >= 0`
  (PDF p.24 remark).
* A **`<= 0`** variable `x` is replaced by `x = -x'` with `x' >= 0`.

### Tableau layout (matches the PDFs exactly)
The objective row is written as `Z + sum z_j x_j = rhs`.
* For a **max** problem: `z_j = -c_j`, the entering variable is the **most
  negative** `z_j`, and optimality is reached when all `z_j >= 0`.
* For a **min** problem: after substituting the basic artificials out of the
  Z-row, the entering variable is the **most positive** `z_j`, and optimality
  is reached when all `z_j <= 0`.

(The two conventions are the negation of each other; storing the row the way
the PDF does makes the displayed tableau byte-for-byte identical to the
course notes — see the Big-M example below.)

### Ratio / leaving rule
* Leaving variable = smallest **positive** ratio `b_i / a_{i,entering}`
  (denominator must be `> 0`).
* Ties are broken by **lowest row index**.
* **Bland's rule** is engaged as an anti-cycling fallback whenever a basis
  repeats.

### Big-M
* `min` problem: artificial objective coefficient `= +M`.
* `max` problem: artificial objective coefficient `= -M`.
* `M` is **never** given a numeric value — `BigM(0, 1)` carries it symbolically
  and comparisons are lexicographic on the M-coefficient first.  This avoids
  the “computational error from a very large M” that the two-phase method was
  invented to fix (PDF p.22).

### Two-phase
* Phase I objective: `min r = R_1 + R_2 + …` (sum of artificials).
* `r > 0` at the Phase-I optimum ⇒ **infeasible** (PDF p.27).
* Before Phase II, any artificial still basic at 0 is pivoted out (the basis
  is “cleaned”) and the artificial columns are forbidden from re-entering.

### Dual simplex (used after a cut)
* Pivot row  = most-negative RHS.
* Pivot col  = `argmin |z_j| / |a_rj|` over `a_rj < 0`  (ties: lowest index).
* If the chosen row has no negative coefficient ⇒ infeasible.

### Gomory fractional cut (PDF p.68–69)
From `x_Bi + sum_{j in J} a_ij x_j = b_i` with non-integer `b_i`, let
`f_ij = a_ij - floor(a_ij)` and `f_i = b_i - floor(b_i)`.  The cut is
`sum f_ij x_j >= f_i`, added as `-sum f_ij x_j + s = -f_i` and re-optimised
with the dual simplex.  The basic variable with the **largest fractional
part** of its RHS is chosen (ties: lowest row index).

### Mixed-integer cut (PDF p.71, eq. *)
For an integer `x_k = d_k`, `f_k = d_k - floor(d_k)`, split the nonbasic set
into `J+ = {a_kj > 0}` and `J- = {a_kj < 0}`:
`sum_{J-} (f_k/(1-f_k)) a_kj x_j - sum_{J+} a_kj x_j <= -f_k`.

### Branch-and-Bound (PDF p.74–76)
DFS, floor (`<=`) child explored first.  Fathom by infeasibility, integrality,
or bound (LP relaxation no better than the incumbent).  A node limit guards
against pathological growth.

---

## Discrepancies / clarification found in the PDFs

1. **Big-M min tableau signs (PDF p.21–22).**
   The PDF writes the Z-row of the min example as `(-4+7M, -1+4M, -M, …)` and
   selects the **most positive** coefficient as the entering variable.  This
   is the *negation* of the max-form row `(4-7M, 1-4M, M, …)` on which the
   “most negative” rule would pick the same variable.  Both views are
   consistent; the solver stores the PDF form (`z_j = -(effective c_j)`) so
   the displayed numbers match the notes.

2. **Two-phase example variables (PDF p.24).**
   The problem statement lists the remarks `x2 free`, `x2 <= 0`,
   `x2 = x'2 - x''2`, `x2 = -x'2` — these are **general remarks** about how
   to handle those cases, not part of the worked example (which has
   `x1, x2 >= 0`).  The solution `(9/2, 5/4)` confirms `x2 >= 0`.  The solver
   supports `free` and `<= 0` variables via the substitutions above, but the
   example itself uses plain nonnegative variables.

3. **Branch-and-Bound example (PDF p.76).**
   The OCR of the constraint block is ambiguous; the constraint `2y <= 5`
   (i.e. `y <= 5/2`) is the only reading consistent with the displayed tree:
   * the root LP optimum is `(7/2, 5/2)` with `Z = 17`;
   * the `y >= 3` branch is **infeasible** (because `y <= 5/2`);
   * the `y <= 2`, `x <= 4` branch yields the integer optimum `(4, 2)`,
     `Z = 16`.
   The objective at the root is `2·(7/2) + 4·(5/2) = 17` (the OCR fragment
   “17 1/2” is a layout artefact, not `17.5`).  The integer optimum `Z = 16`
   is verified against brute-force enumeration in the test suite.

4. **Gomory example first cut (PDF p.72).**
   The first cut is `-(7/22) S1 - (1/22) S2 <= -1/2`, i.e.
   `7/22 S1 + 1/22 S2 >= 1/2`, derived from the `y`-row
   (`y + 7/22 S1 + 1/22 S2 = 7/2`, fractional part `1/2`).  Both `x` and `y`
   have fractional part `1/2`; the solver picks `y` (lowest row index),
   reproducing the PDF exactly.  After two cuts the integer optimum
   `(4, 3), Z = 55` is reached.

5. **ILP rounding trap (PDF p.65).**
   The LP relaxation optimum is `(2, 9/5)` with `Z = 11`; rounding to
   `(2, 2)` is infeasible (`x1 + 10·2 = 22 > 20`) and rounding to `(2, 1)`
   gives `Z = 7`, both worse than the true integer optimum `(0, 2), Z = 10`.

6. **Phase-I `r` value for the infeasible example (PDF p.28).**
   The PDF reports `min r = 4` for `max 3x1+2x2; 2x1+x2<=2; 3x1+4x2>=12`.
   The solver reproduces `r = 4` exactly (test case 6).

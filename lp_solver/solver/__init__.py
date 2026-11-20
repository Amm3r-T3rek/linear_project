"""Pure-Python LP / ILP solver package.

The submodules implement, with exact rational arithmetic:

  * :mod:`solver.simplex`      -- the primal Simplex method
  * :mod:`solver.big_m`        -- the Big-M (penalty) method
  * :mod:`solver.two_phase`    -- the two-phase method
  * :mod:`solver.dual_simplex` -- the dual simplex method (for cutting planes)
  * :mod:`solver.gomory`       -- Gomory fractional cut (pure ILP)
  * :mod:`solver.mixed_cut`    -- mixed-integer cut
  * :mod:`solver.branch_bound` -- branch-and-bound (pure / mixed / 0-1)

All methods return a :class:`solver.models.Result` whose ``steps`` list is a
serialisable step log (see :mod:`solver.steps`).

The package has no Streamlit dependency; the UI lives in :mod:`lp_solver.ui`.
"""

from .models import Problem, Constraint, Variable, Result, VerificationRow
from .numbers import BigM
from .standardize import StandardForm, standardize
from .tableau import Tableau
from .steps import Step, StepLog, TableauSnapshot
from .simplex import solve_simplex, run_simplex, build_tableau, verify
from .big_m import solve_big_m
from .two_phase import solve_two_phase
from .dual_simplex import run_dual_simplex
from .gomory import solve_gomory
from .mixed_cut import solve_mixed_integer
from .branch_bound import solve_branch_bound

__all__ = [
    "Problem", "Constraint", "Variable", "Result", "VerificationRow",
    "BigM", "StandardForm", "standardize", "Tableau",
    "Step", "StepLog", "TableauSnapshot",
    "solve_simplex", "run_simplex", "build_tableau", "verify",
    "solve_big_m", "solve_two_phase", "run_dual_simplex",
    "solve_gomory", "solve_mixed_integer", "solve_branch_bound",
]

"""Streamlit entry point for the LP/ILP solver.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
import os
from fractions import Fraction

import streamlit as st

# Make the package importable whether run from inside lp_solver/ or its parent.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from solver import (
    solve_simplex, solve_big_m, solve_two_phase,
    solve_gomory, solve_mixed_integer, solve_branch_bound,
    Problem,
)
import examples as ex
from ui.tableau_view import render_steps, render_snapshot
from ui.graph import render_graph
from ui.bb_tree import render_bb_tree
from ui.export import to_json, to_markdown, to_pdf
from ui.input_panel import input_problem

st.set_page_config(
    page_title="LP / ILP Solver",
    page_icon=":abacus:",
    layout="wide",
)

# --------------------------------------------------------------------------- #
#  Header
# --------------------------------------------------------------------------- #
st.title("Linear & Integer Programming Solver")
st.caption(
    "Exact rational arithmetic (fractions + symbolic Big-M). Simplex, Big-M, "
    "Two-Phase, Dual-Simplex, Gomory cut, Mixed-integer cut, Branch-and-Bound."
)

# --------------------------------------------------------------------------- #
#  Sidebar: method, examples, options
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Settings")
    method = st.selectbox(
        "Method",
        ["Simplex", "Big-M", "Two-Phase", "Gomory cut",
         "Mixed-integer cut", "Branch & Bound", "Compare methods"],
    )
    fmt = st.radio("Number format", ["fraction", "decimal"], horizontal=True)
    use_example = st.checkbox("Load a worked example", value=True)
    if use_example:
        ex_names = [e[0] for e in ex.EXAMPLES]
        chosen = st.selectbox("Example", ex_names)
    st.divider()
    st.markdown("**About**")
    st.markdown(
        "Pure-Python solver. Run `pytest` for the 12 worked test cases "
        "and randomized cross-checks vs scipy / brute force."
    )

SOLVERS = {
    "Simplex": solve_simplex,
    "Big-M": solve_big_m,
    "Two-Phase": solve_two_phase,
    "Gomory cut": solve_gomory,
    "Mixed-integer cut": solve_mixed_integer,
    "Branch & Bound": solve_branch_bound,
}


# --------------------------------------------------------------------------- #
#  Build / load the problem
# --------------------------------------------------------------------------- #
problem: Problem | None = None
if use_example:
    idx = [e[0] for e in ex.EXAMPLES].index(chosen)
    problem = ex.EXAMPLES[idx][1]
    # If the chosen example recommends a method and the user hasn't overridden,
    # we still let the user pick; but show the recommended one.
    recommended = ex.EXAMPLES[idx][2]
    with st.sidebar:
        st.info(f"Recommended method for this example: **{recommended}**")
else:
    problem = input_problem()

if problem is None:
    st.stop()

# --------------------------------------------------------------------------- #
#  Problem summary
# --------------------------------------------------------------------------- #
with st.expander("Problem", expanded=True):
    from ui.export import problem_str
    st.code(problem_str(problem))

# --------------------------------------------------------------------------- #
#  Solve
# --------------------------------------------------------------------------- #
def fmt_mode() -> str:
    return "frac" if fmt == "fraction" else "dec"


if st.button("Solve", type="primary", use_container_width=True):
    st.session_state["result"] = None
    st.session_state["results"] = None
    try:
        if method == "Compare methods":
            results = {}
            for m in ("Simplex", "Big-M", "Two-Phase"):
                try:
                    results[m] = SOLVERS[m](problem)
                except Exception as e:
                    results[m] = e
            st.session_state["results"] = results
        else:
            st.session_state["result"] = SOLVERS[method](problem)
    except Exception as e:
        st.error(f"Error: {e}")
        st.session_state["result"] = None

# --------------------------------------------------------------------------- #
#  Display results
# --------------------------------------------------------------------------- #
left, right = st.columns([3, 2])

with left:
    if st.session_state.get("results"):
        st.subheader("Compare methods")
        data = []
        for m, r in st.session_state["results"].items():
            if isinstance(r, Exception):
                data.append({"Method": m, "Status": "error",
                             "Z": str(r), "Solution": ""})
            else:
                sol = ", ".join(f"{k}={v}" for k, v in r.solution.items())
                data.append({"Method": m, "Status": r.status,
                             "Z": str(r.objective_value) if r.objective_value is not None else "—",
                             "Solution": sol})
        import pandas as pd
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
        st.success("All three primal methods reach the same Z when the problem "
                   "is feasible and bounded.")

    result = st.session_state.get("result")
    if result is not None:
        # Status banner
        if result.status == "optimal":
            st.success(f"**Optimal**  —  Z = `{result.objective_value}`")
        elif result.status == "alternative":
            st.success(f"**Optimal (alternative optima exist)**  —  Z = "
                       f"`{result.objective_value}`")
        elif result.status == "infeasible":
            st.error("**Infeasible** — the feasible region is empty.")
        elif result.status == "unbounded":
            st.warning("**Unbounded** — the objective grows without bound.")
        else:
            st.info(f"Status: {result.status}")

        if result.solution:
            sol = "  ·  ".join(f"{k} = `{v}`"
                               for k, v in result.solution.items())
            st.markdown(f"**Solution:** {sol}")

        st.divider()
        st.subheader("Step-by-step solution")
        render_steps(result.steps, fmt_mode())

        if result.bb_tree:
            st.subheader("Branch-and-Bound tree")
            fig = render_bb_tree(result.bb_tree)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Final verification")
        import pandas as pd
        vrows = [{
            "Constraint": v.expr, "LHS": str(v.lhs), "op": v.op,
            "RHS": str(v.rhs), "slack (LHS-RHS)": str(v.slack),
            "satisfied": "PASS" if v.satisfied else "FAIL",
        } for v in result.verification]
        st.dataframe(pd.DataFrame(vrows), use_container_width=True,
                     hide_index=True)

with right:
    st.subheader("Graphical view")
    names = [v.name for v in problem.variables]
    if len(names) == 2:
        show_lat = st.checkbox("Show integer lattice (ILP)",
                               value=any(v.is_integer for v in problem.variables))
        sol = st.session_state.get("result")
        sol_dict = None
        if sol is not None and sol.solution:
            sol_dict = sol.solution
        fig = render_graph(problem, sol_dict, show_integer_lattice=show_lat)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Graphical view is available for 2-variable problems.")

# --------------------------------------------------------------------------- #
#  Exports
# --------------------------------------------------------------------------- #
result = st.session_state.get("result")
if result is not None:
    st.divider()
    st.subheader("Export")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Markdown", to_markdown(result, problem),
                           file_name="solution.md", mime="text/markdown")
    with c2:
        st.download_button("JSON", to_json(result, problem),
                           file_name="solution.json", mime="application/json")
    with c3:
        try:
            st.download_button("PDF", to_pdf(result, problem),
                               file_name="solution.pdf",
                               mime="application/pdf")
        except Exception as e:
            st.warning(f"PDF export unavailable: {e}")

st.markdown("---")
st.caption("LP/ILP Solver — pure Python (Streamlit + Plotly + pandas + reportlab). "
           "Exact arithmetic via `fractions.Fraction` and a symbolic `BigM` class.")

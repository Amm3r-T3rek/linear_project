"""Streamlit input panel: build a :class:`Problem` interactively."""

from __future__ import annotations

from fractions import Fraction

import streamlit as st

from solver.models import Problem, Constraint, Variable


def input_problem(key_prefix: str = "ip") -> Problem | None:
    """Render the input form and return the constructed :class:`Problem`."""
    st.subheader("Problem definition")

    sense = st.radio("Objective sense", ["max", "min"], horizontal=True,
                     key=f"{key_prefix}_sense")

    n_vars = st.number_input("Number of variables", 1, 6, 2,
                             key=f"{key_prefix}_nvars")
    n_cons = st.number_input("Number of constraints", 1, 8, 3,
                             key=f"{key_prefix}_ncons")

    st.markdown("**Variables** (name, type)")
    var_names: list[str] = []
    var_types: list[str] = []
    cols = st.columns(min(n_vars, 4))
    default_names = ["x1", "x2", "x3", "x4", "x5", "x6"]
    for i in range(n_vars):
        with cols[i % len(cols)]:
            nm = st.text_input("Name", value=default_names[i],
                              key=f"{key_prefix}_vname_{i}")
            vt = st.selectbox("Type",
                             ["nonneg", "free", "leq0", "integer", "binary"],
                             key=f"{key_prefix}_vtype_{i}")
            var_names.append(nm)
            var_types.append(vt)

    st.markdown("**Objective coefficients**")
    obj: dict[str, Fraction] = {}
    ocols = st.columns(min(n_vars, 4))
    for i in range(n_vars):
        with ocols[i % len(ocols)]:
            c = st.number_input(f"c({var_names[i]})", value=1, step=1,
                               key=f"{key_prefix}_obj_{i}")
            obj[var_names[i]] = Fraction(c)

    st.markdown("**Constraints**")
    constraints: list[Constraint] = []
    for j in range(n_cons):
        with st.expander(f"Constraint {j+1}", expanded=(j < 2)):
            ccols = st.columns(min(n_vars, 3) + 1)
            coeffs: dict[str, Fraction] = {}
            for i in range(n_vars):
                with ccols[i]:
                    a = st.number_input(f"a({var_names[i]})", value=0, step=1,
                                        key=f"{key_prefix}_a_{j}_{i}")
                    coeffs[var_names[i]] = Fraction(a)
            with ccols[-1] if n_vars < len(ccols) else st.container():
                op = st.selectbox("op", ["<=", ">=", "="],
                                  key=f"{key_prefix}_op_{j}")
                rhs = st.number_input("RHS", value=0, step=1,
                                     key=f"{key_prefix}_rhs_{j}")
            constraints.append(Constraint(coeffs, op, Fraction(rhs)))

    variables = [Variable(n, t) for n, t in zip(var_names, var_types)]

    # validation
    errors = []
    if len(set(var_names)) != len(var_names):
        errors.append("Variable names must be unique.")
    for c in constraints:
        if all(v == 0 for v in c.coeffs.values()):
            errors.append(f"A constraint has all-zero coefficients.")
    if errors:
        for e in errors:
            st.error(e)
        return None

    return Problem(sense=sense, objective=obj, constraints=constraints,
                  variables=variables)

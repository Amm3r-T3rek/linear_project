"""Interactive Plotly graph for 2-variable LP/ILP problems."""

from __future__ import annotations

from fractions import Fraction

import plotly.graph_objects as go

from solver.models import Problem


def render_graph(problem: Problem, solution: dict[str, Fraction] | None = None,
                 show_integer_lattice: bool = False) -> go.Figure | None:
    """Build a Plotly figure of the feasible region for a 2-variable problem.

    Returns None if the problem does not have exactly 2 decision variables.
    """
    names = [v.name for v in problem.variables]
    if len(names) != 2:
        return None
    x_name, y_name = names
    x0 = problem.variables[0]
    y0 = problem.variables[1]
    is_ilp = x0.is_integer or y0.is_integer

    # Determine plotting bounds: use the tightest constraints to bound the box.
    x_max, y_max = _auto_bounds(problem, x_name, y_name)
    if x_max is None:
        x_max = 20
    if y_max is None:
        y_max = 20

    fig = go.Figure()

    # Feasible region as a polygon via half-plane intersection on a grid.
    region_x, region_y = _feasible_polygon(problem, x_name, y_name, x_max, y_max)
    if region_x:
        fig.add_trace(go.Scatter(
            x=region_x + [region_x[0]],
            y=region_y + [region_y[0]],
            fill="toself", fillcolor="rgba(34,197,94,0.18)",
            line=dict(color="rgba(34,197,94,0.6)"),
            name="Feasible region", hoverinfo="skip"))

    # Constraint lines.
    for i, con in enumerate(problem.constraints):
        xs, ys = _constraint_line(con, x_name, y_name, x_max, y_max)
        if xs:
            label = _constraint_label(con, x_name, y_name)
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(width=2, dash="solid"),
                name=f"{label}", hovertemplate=label + "<extra></extra>"))

    # Integer lattice points (for ILP).
    if is_ilp and show_integer_lattice:
        pts_x, pts_y, feas_x, feas_y = [], [], [], []
        for xi in range(0, int(x_max) + 1):
            for yi in range(0, int(y_max) + 1):
                if _feasible(problem, x_name, y_name, Fraction(xi), Fraction(yi)):
                    feas_x.append(xi); feas_y.append(yi)
                else:
                    pts_x.append(xi); pts_y.append(yi)
        fig.add_trace(go.Scatter(x=pts_x, y=pts_y, mode="markers",
                                 marker=dict(symbol="x", color="rgba(120,120,120,0.4)"),
                                 name="infeasible integer", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=feas_x, y=feas_y, mode="markers",
                                 marker=dict(symbol="circle", size=8,
                                             color="rgba(22,163,74,0.9)"),
                                 name="feasible integer", hovertemplate="(%{x},%{y})<extra></extra>"))

    # Objective line (iso-profit) through the optimum.
    if solution and solution.get(x_name) is not None and solution.get(y_name) is not None:
        xs = float(solution[x_name])
        ys = float(solution[y_name])
        fig.add_trace(go.Scatter(
            x=[xs], y=[ys], mode="markers",
            marker=dict(symbol="star", size=16, color="red",
                        line=dict(width=1, color="black")),
            name="optimum", hovertemplate=f"({xs}, {ys})<extra>optimum</extra>"))
        # iso line: c1*x + c2*y = Z*
        c1 = float(problem.objective.get(x_name, 0))
        c2 = float(problem.objective.get(y_name, 0))
        z = c1 * xs + c2 * ys
        if c2 != 0:
            lx = [0, x_max]
            ly = [(z - c1 * lx[0]) / c2, (z - c1 * lx[1]) / c2]
        elif c1 != 0:
            ly = [0, y_max]
            lx = [(z - c2 * 0) / c1, (z - c2 * y_max) / c1]
        else:
            lx, ly = [], []
        if lx:
            fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines",
                                     line=dict(width=1.5, dash="dash", color="red"),
                                     name=f"Z = {_fmt_frac(Fraction(z).limit_denominator(1000))}",
                                     hoverinfo="skip"))

    fig.update_layout(
        xaxis_title=x_name, yaxis_title=y_name,
        template="plotly_white", height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10),
    )
    return fig


def _auto_bounds(problem, x_name, y_name):
    x_max = y_max = None
    for con in problem.constraints:
        cx = con.coeffs.get(x_name, 0)
        cy = con.coeffs.get(y_name, 0)
        b = con.rhs
        if con.op == "<=":
            if cx > 0 and cy == 0:
                bound = float(b) / float(cx)
                x_max = min(x_max or bound, bound)
            if cy > 0 and cx == 0:
                bound = float(b) / float(cy)
                y_max = min(y_max or bound, bound)
    return (float(x_max) if x_max else 12.0), (float(y_max) if y_max else 12.0)


def _constraint_line(con, x_name, y_name, x_max, y_max):
    cx = float(con.coeffs.get(x_name, 0))
    cy = float(con.coeffs.get(y_name, 0))
    b = float(con.rhs)
    pts = []
    if cx == 0 and cy == 0:
        return [], []
    if cx != 0 and cy != 0:
        # y = (b - cx*x)/cy
        x0 = b / cx if cx != 0 else 0
        pts = [(0, b / cy), (x_max, (b - cx * x_max) / cy),
               (x0, 0), ((b - cy * y_max) / cx, y_max)]
    elif cx != 0:
        x0 = b / cx
        pts = [(x0, 0), (x0, y_max)]
    else:  # cy != 0
        y0 = b / cy
        pts = [(0, y0), (x_max, y0)]
    # keep points within the box, then return two extreme ones
    inside = [(x, y) for x, y in pts if -1e-6 <= x <= x_max + 1e-6
              and -1e-6 <= y <= y_max + 1e-6]
    if len(inside) < 2:
        # extend to box edges anyway for visual
        inside = pts
    if len(inside) >= 2:
        xs = [inside[0][0], inside[1][0]]
        ys = [inside[0][1], inside[1][1]]
        return xs, ys
    return [], []


def _constraint_label(con, x_name, y_name):
    terms = " + ".join(
        f"{_fmt_frac(v)}{n}" for n, v in con.coeffs.items() if v != 0)
    return f"{terms} {con.op} {con.rhs}"


def _feasible(problem, x_name, y_name, xv, yv):
    point = {x_name: xv, y_name: yv}
    for con in problem.constraints:
        lhs = sum(Fraction(c) * point.get(n, Fraction(0))
                  for n, c in con.coeffs.items())
        rhs = Fraction(con.rhs)
        if con.op == "<=" and not (lhs <= rhs): return False
        if con.op == ">=" and not (lhs >= rhs): return False
        if con.op == "=" and not (lhs == rhs): return False
    return True


def _feasible_polygon(problem, x_name, y_name, x_max, y_max):
    """Compute the feasible region polygon by clipping half-planes."""
    import numpy as np
    # Sutherland-Hodgman polygon clipping against each constraint half-plane.
    poly = [(0, 0), (x_max, 0), (x_max, y_max), (0, y_max)]
    for con in problem.constraints:
        cx = float(con.coeffs.get(x_name, 0))
        cy = float(con.coeffs.get(y_name, 0))
        b = float(con.rhs)
        poly = _clip(poly, cx, cy, b, con.op)
        if not poly:
            return [], []
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return xs, ys


def _clip(poly, cx, cy, b, op):
    """Clip polygon against the half-plane cx*x + cy*y (op) b."""
    if op == "<=":
        # keep cx*x + cy*y <= b
        def inside(p): return cx * p[0] + cy * p[1] <= b + 1e-9
        nx, ny = cx, cy
    elif op == ">=":
        def inside(p): return cx * p[0] + cy * p[1] >= b - 1e-9
        nx, ny = -cx, -cy
        b = -b
    else:  # equality -> thin slab
        def inside(p): return abs(cx * p[0] + cy * p[1] - b) < 1e-6
        # approximate as <= with tiny width (visual only)
        return _clip(_clip(poly, cx, cy, b + 1e-3, "<="), cx, cy, b - 1e-3, ">=") \
            if False else [p for p in poly if inside(p)]
    if not poly:
        return []
    out = []
    n = len(poly)
    for i in range(n):
        cur = poly[i]
        nxt = poly[(i + 1) % n]
        cur_in = inside(cur)
        nxt_in = inside(nxt)
        if cur_in:
            out.append(cur)
        if cur_in != nxt_in:
            inter = _intersect(cur, nxt, nx, ny, b)
            if inter is not None:
                out.append(inter)
    return out


def _intersect(p1, p2, nx, ny, b):
    x1, y1 = p1; x2, y2 = p2
    d = nx * (x2 - x1) + ny * (y2 - y1)
    if abs(d) < 1e-12:
        return None
    t = (b - (nx * x1 + ny * y1)) / d
    return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))


def _fmt_frac(v):
    f = Fraction(v).limit_denominator(1000)
    if f.denominator == 1:
        return str(f.numerator)
    return f"{f.numerator}/{f.denominator}"

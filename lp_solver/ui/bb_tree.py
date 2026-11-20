"""Visualise the branch-and-bound tree with Plotly."""

from __future__ import annotations

import plotly.graph_objects as go


def render_bb_tree(tree: dict | None) -> go.Figure | None:
    if not tree or not tree.get("nodes"):
        return None
    nodes = tree["nodes"]
    # assign levels by parent chain
    level = {0: 0}
    for n in nodes:
        nid = n["id"]
        pid = n["parent"]
        if pid is None:
            level[nid] = 0
        else:
            # compute lazily
            stack = [nid]
            while stack and stack[-1] not in level:
                cur = stack[-1]
                p = next(x["parent"] for x in nodes if x["id"] == cur)
                if p is None:
                    level[cur] = 0
                    stack.pop()
                elif p in level:
                    level[cur] = level[p] + 1
                    stack.pop()
                else:
                    stack.append(p)
    # x positions: spread children within each level
    by_level: dict[int, list[int]] = {}
    for nid, lv in level.items():
        by_level.setdefault(lv, []).append(nid)
    pos: dict[int, tuple[float, float]] = {}
    for lv, ids in sorted(by_level.items()):
        m = len(ids)
        for i, nid in enumerate(ids):
            x = (i - (m - 1) / 2)
            pos[nid] = (x, -lv)

    fig = go.Figure()
    edges_x, edges_y = [], []
    for n in nodes:
        if n["parent"] is not None and n["parent"] in pos:
            x0, y0 = pos[n["parent"]]
            x1, y1 = pos[n["id"]]
            edges_x += [x0, x1, None]
            edges_y += [y0, y1, None]
    fig.add_trace(go.Scatter(x=edges_x, y=edges_y, mode="lines",
                             line=dict(color="#94a3b8", width=1),
                             hoverinfo="skip", showlegend=False))

    node_x, node_y, text, colors = [], [], [], []
    for n in nodes:
        x, y = pos[n["id"]]
        node_x.append(x); node_y.append(y)
        lp = n["lp_value"]
        lp_s = str(lp) if lp else "—"
        sol = ", ".join(f"{k}={v}" for k, v in
                        (n.get("lp_solution") or {}).items())
        text.append(
            f"#{n['id']} {n['branch']}<br>LP Z={lp_s}<br>{sol}<br>"
            f"[{n['status']}] {n.get('fathom_reason','')}")
        if n["is_integer"]:
            colors.append("#16a34a")
        elif n["status"] == "infeasible":
            colors.append("#ef4444")
        elif n["status"] == "fathomed":
            colors.append("#f59e0b")
        else:
            colors.append("#3b82f6")
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                             marker=dict(size=26, color=colors,
                                         line=dict(width=1, color="white")),
                             text=[str(n["id"]) for n in nodes],
                             textposition="middle center",
                             textfont=dict(color="white", size=11),
                             hovertext=text, hoverinfo="text",
                             showlegend=False))
    fig.update_layout(
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        template="plotly_white", height=420,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig

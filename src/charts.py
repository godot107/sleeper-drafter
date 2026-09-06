"""Plotly figures for the browser dashboard.

Each figure is chosen for the job it does, not for variety:

* **Value cliffs** -- magnitude across an ordered pool. Small multiples, one
  panel per position, every panel in a single hue. Identity comes from the panel
  title, which frees the colour channel entirely and sidesteps the
  colour-vision problem that six categorical hues in one scatter would create.
* **Decision scatter** -- the actual draft question, in two dimensions: will he
  last (survival) against what waiting costs (VONA). One series, so no legend is
  needed and no palette gate applies; the top candidates carry direct labels.
* **Standings** -- ranked magnitude with one highlighted member. Two validated
  hues plus direct value labels.
* **Scarcity** -- expected departures per position before your turn. One series.

Hover is on everywhere. A number is never printed on every mark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .theme import layout_defaults, palette

POSITION_ORDER = ("QB", "RB", "WR", "TE", "K", "DEF")


def _empty(theme: str, message: str) -> go.Figure:
    c = palette(theme)
    fig = go.Figure()
    fig.update_layout(**layout_defaults(theme))
    fig.add_annotation(
        text=message, showarrow=False,
        font={"color": c["muted"], "size": 13}, x=0.5, y=0.5, xref="paper", yref="paper",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def value_cliff(board: pd.DataFrame, theme: str, depth: int = 18) -> go.Figure:
    """Projected points by positional rank, one panel per position.

    The steep segments *are* the cliffs -- the thing worth reading off a draft
    board. Tier boundaries are drawn as hairlines so a break you can see is a
    break the model also acted on.
    """
    c = palette(theme)
    positions = [p for p in POSITION_ORDER if (board["pos"] == p).any()]
    if not positions:
        return _empty(theme, "no players available")

    fig = make_subplots(
        rows=2, cols=3, subplot_titles=positions,
        vertical_spacing=0.16, horizontal_spacing=0.07,
    )

    for i, pos in enumerate(positions):
        row, col = divmod(i, 3)
        group = board[board["pos"] == pos].nlargest(depth, "proj_pts").copy()
        group = group.sort_values("proj_pts", ascending=False).reset_index(drop=True)
        group["rank"] = group.index + 1

        fig.add_trace(
            go.Scatter(
                x=group["rank"], y=group["proj_pts"],
                mode="lines+markers",
                line={"color": c["series_1"], "width": 2},
                marker={"size": 8, "color": c["series_1"],
                        "line": {"width": 2, "color": c["surface"]}},
                customdata=np.stack([
                    group["name"], group["adp"], group["dropoff"],
                    group["tier"], group.get("risk", pd.Series([""] * len(group))),
                ], axis=-1),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "%{y:.0f} pts · ADP %{customdata[1]:.1f}<br>"
                    "drop to next: %{customdata[2]:.0f} pts<br>"
                    "tier %{customdata[3]} · %{customdata[4]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row + 1, col=col + 1,
        )

        # Hairlines where the tier changes -- the measured cliffs.
        breaks = group.index[group["tier"].diff().fillna(0) > 0]
        for b in breaks:
            fig.add_vline(
                x=float(group.loc[b, "rank"]) - 0.5,
                line={"color": c["axis"], "width": 1, "dash": "dot"},
                row=row + 1, col=col + 1,
            )

    fig.update_layout(**layout_defaults(theme))
    fig.update_layout(height=430, margin={"l": 48, "r": 16, "t": 32, "b": 36})
    fig.update_xaxes(gridcolor=c["grid"], linecolor=c["axis"], zeroline=False,
                     tickfont={"color": c["muted"]}, title=None)
    fig.update_yaxes(gridcolor=c["grid"], linecolor=c["axis"], zeroline=False,
                     tickfont={"color": c["muted"]}, title=None)
    for annotation in fig.layout.annotations:
        annotation.font.update(color=c["text_secondary"], size=12)
    # Label the axes once on the outer panels rather than on all six.
    for col in (1, 2, 3):
        fig.update_xaxes(title_text="positional rank", row=2, col=col,
                         title_font={"color": c["muted"], "size": 11})
    for row in (1, 2):
        fig.update_yaxes(title_text="projected pts", row=row, col=1,
                         title_font={"color": c["muted"], "size": 11})
    fig.update_layout(margin={"l": 62, "r": 16, "t": 32, "b": 48})
    return fig


def decision_scatter(ranked: pd.DataFrame, theme: str, label_n: int = 8) -> go.Figure:
    """Survival against VONA -- the take-now question in two dimensions.

    Upper-left is the answer: high cost to wait, low chance he lasts. Lower-right
    is the opposite, and is just as useful -- it tells you who you can safely
    pass on this turn.
    """
    c = palette(theme)
    if ranked.empty:
        return _empty(theme, "no players available")

    top = ranked.head(60).copy()
    top["survival_pct"] = top["survival"] * 100.0

    fig = go.Figure()
    fig.update_layout(**layout_defaults(theme))

    # Quadrant guides, recessive: median survival and VONA break-even.
    fig.add_vline(x=50, line={"color": c["axis"], "width": 1, "dash": "dot"})
    fig.add_hline(y=0, line={"color": c["axis"], "width": 1, "dash": "dot"})
    fig.add_annotation(
        x=6, y=top["vona"].max(), text="take now", showarrow=False,
        font={"color": c["muted"], "size": 11}, xanchor="left",
    )
    fig.add_annotation(
        x=97, y=min(0, top["vona"].min()), text="safe to wait", showarrow=False,
        font={"color": c["muted"], "size": 11}, xanchor="right",
    )

    # Label only what is actionable. The crowded lower band is "don't take
    # him", and stacking eight labels there produced overlapping text -- the
    # points worth naming are the ones above break-even, which are sparse.
    labelled = top[top["vona"] > 0].head(label_n)
    if labelled.empty:
        labelled = top.head(3)
    fig.add_trace(go.Scatter(
        x=top["survival_pct"], y=top["vona"],
        mode="markers",
        marker={"size": 10, "color": c["series_1"],
                "line": {"width": 2, "color": c["surface"]}},
        customdata=np.stack([
            top["name"], top["pos"], top["proj_pts"], top["adp"],
            top["tier"], top.get("risk", pd.Series([""] * len(top))),
        ], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
            "%{customdata[2]:.0f} pts · ADP %{customdata[3]:.1f}<br>"
            "survives to your turn: %{x:.0f}%<br>"
            "cost of waiting (VONA): %{y:.0f} pts<br>"
            "tier %{customdata[4]} · %{customdata[5]}<extra></extra>"
        ),
        showlegend=False,
    ))
    # Alternate above/below so near-neighbours cannot collide, and pull labels
    # inward at the edges -- a point at 100% survival sits against the frame and
    # a centred label runs off it.
    order = labelled["vona"].rank(ascending=False, method="first").astype(int)
    positions_alt = []
    for rank_i, survival_x in zip(order, labelled["survival_pct"]):
        vertical = "top" if rank_i % 2 else "bottom"
        if survival_x >= 88:
            horizontal = "left"
        elif survival_x <= 12:
            horizontal = "right"
        else:
            horizontal = "center"
        positions_alt.append(f"{vertical} {horizontal}")
    fig.add_trace(go.Scatter(
        x=labelled["survival_pct"], y=labelled["vona"],
        mode="text",
        text=[f"{n.split()[-1]} ({p})" for n, p in zip(labelled["name"], labelled["pos"])],
        textposition=positions_alt,
        textfont={"color": c["text_secondary"], "size": 11},
        hoverinfo="skip", showlegend=False,
    ))

    fig.update_layout(height=380)
    fig.update_xaxes(title="chance he lasts to your next pick (%)", range=[-4, 104])
    fig.update_yaxes(title="cost of waiting — VONA (pts)")
    return fig


def standings_bar(rated: pd.DataFrame, my_roster: int, theme: str) -> go.Figure:
    """Projected starting-lineup points per team, yours highlighted."""
    c = palette(theme)
    if rated.empty:
        return _empty(theme, "no picks yet")

    frame = rated.sort_values("starters_pts").copy()
    mine = frame["roster_id"] == my_roster
    labels = [f"slot {int(s)}" for s in frame["draft_slot"]]

    fig = go.Figure()
    fig.update_layout(**layout_defaults(theme))
    for is_me, name, colour in ((False, "other teams", c["series_1"]),
                                (True, "your team", c["series_2"])):
        subset = frame[mine == is_me]
        if subset.empty:
            continue
        fig.add_trace(go.Bar(
            y=[f"slot {int(s)}" for s in subset["draft_slot"]],
            x=subset["starters_pts"],
            orientation="h",
            name=name,
            marker={"color": colour, "line": {"width": 2, "color": c["surface"]}},
            text=[f"{v:,.0f}  {g}" for v, g in zip(subset["starters_pts"], subset["grade"])],
            textposition="outside",
            textfont={"color": c["text_secondary"], "size": 11},
            customdata=np.stack([subset["grade"], subset["holes"], subset["picks"]], axis=-1),
            hovertemplate=(
                "%{y}<br>%{x:,.0f} projected starter pts<br>"
                "grade %{customdata[0]} · %{customdata[1]} unfilled · "
                "%{customdata[2]} picks<extra></extra>"
            ),
        ))

    fig.update_layout(height=360, bargap=0.35,
                      # Outside labels carry both the value and the grade, so the
                      # right margin has to hold them or they get clipped mid-word.
                      margin={"l": 68, "r": 128, "t": 44, "b": 40})
    fig.update_xaxes(title="projected starting-lineup points",
                     range=[0, float(frame["starters_pts"].max()) * 1.18])
    # Categories live on the y-axis for a horizontal bar chart. Ordering the
    # x-axis instead silently left the bars unsorted, which is worse than no
    # ranking at all -- it reads as a ranking and is not one.
    fig.update_yaxes(title=None, categoryorder="array", categoryarray=labels)
    return fig


def scarcity_bar(expected: dict[str, float], theme: str) -> go.Figure:
    """Expected players taken per position before your next pick."""
    c = palette(theme)
    items = [(p, expected.get(p, 0.0)) for p in POSITION_ORDER if expected.get(p, 0.0) > 0.01]
    if not items:
        return _empty(theme, "you pick again immediately — nothing comes off the board")

    items.sort(key=lambda kv: kv[1])
    names = [p for p, _ in items]
    values = [v for _, v in items]

    fig = go.Figure()
    fig.update_layout(**layout_defaults(theme))
    fig.add_trace(go.Bar(
        y=names, x=values, orientation="h",
        marker={"color": c["series_1"], "line": {"width": 2, "color": c["surface"]}},
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
        textfont={"color": c["text_secondary"], "size": 11},
        hovertemplate="%{y}: %{x:.1f} expected off the board<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(height=240, bargap=0.35, margin={"l": 56, "r": 48, "t": 24, "b": 36})
    fig.update_xaxes(title="expected picks before your turn")
    fig.update_yaxes(title=None)
    return fig

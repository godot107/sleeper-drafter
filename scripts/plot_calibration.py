#!/usr/bin/env python3
"""Render the survival-calibration figure used in README/blog from measured data.

The other figures in ``docs/assets`` were made ad hoc, which is how the published
one drifted a draft behind the model it describes. This one is a script so the
picture and the coefficient cannot disagree again: refit ``survival_gamma``,
update ``data/calibration_buckets.csv``, re-run this.

Two panels rather than three series. Raw, corrected and actual are three
quantities, but ``src/theme`` deliberately ships two validated categorical hues
and says why -- a third would have to clear colour-vision separation against
both of them, and faceting is the remedy the theme already recommends. So each
panel holds one prediction against the same truth, and the correction is read as
the gap closing between panels.

Usage:
    python scripts/plot_calibration.py [--theme light|dark] [--out PATH]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from plotly.subplots import make_subplots
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.theme import FONT, layout_defaults, palette  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BUCKETS = ROOT / "data" / "calibration_buckets.csv"
DEFAULT_OUT = ROOT / "docs" / "assets" / "survival-calibration.png"


def load_buckets(path: Path) -> list[dict]:
    """Read the committed bucket table, ignoring the provenance header."""
    with path.open() as fh:
        rows = list(csv.DictReader(line for line in fh if not line.startswith("#")))
    if not rows:
        raise SystemExit(f"no rows in {path}")
    for r in rows:
        r["n"] = int(r["n"])
        for k in ("predicted", "actual", "corrected"):
            r[k] = float(r[k])
    return rows


def build(rows: list[dict], theme: str) -> go.Figure:
    c = palette(theme)
    labels = [r["bucket"] for r in rows]
    total = sum(r["n"] for r in rows)

    # n varies by three orders of magnitude across buckets; saying so under each
    # tick is what stops the tiny, badly-calibrated buckets from reading as if
    # they carried the same weight as the 80-100% one.
    ticks = [f"{b}<br><span style='font-size:11px'>n={r['n']:,}</span>"
             for b, r in zip(labels, rows)]

    fig = make_subplots(
        rows=1, cols=2, shared_yaxes=True, horizontal_spacing=0.06,
        subplot_titles=("Raw product form", "Corrected (p<sup>4.11</sup>)"),
    )

    for col, key in ((1, "predicted"), (2, "corrected")):
        pred = [r[key] * 100 for r in rows]
        act = [r["actual"] * 100 for r in rows]
        fig.add_bar(
            x=ticks, y=pred, name="model said", legendgroup="model",
            showlegend=(col == 1), marker_color=c["series_1"],
            text=[f"{v:.0f}%" for v in pred], textposition="outside",
            textfont={"color": c["text_secondary"], "size": 11},
            hovertemplate="model said %{y:.1f}%<extra></extra>",
            row=1, col=col,
        )
        fig.add_bar(
            x=ticks, y=act, name="actually survived", legendgroup="actual",
            showlegend=(col == 1), marker_color=c["series_2"],
            text=[f"{v:.0f}%" for v in act], textposition="outside",
            textfont={"color": c["text_secondary"], "size": 11},
            hovertemplate="actually survived %{y:.1f}%<extra></extra>",
            row=1, col=col,
        )

    fig.update_layout(**layout_defaults(theme))
    fig.update_layout(
        title={
            "text": (
                "Where the survival model is wrong, and what the correction fixes"
                f"<br><span style='font-size:12px'>{total:,} predictions across "
                "four live drafts</span>"
            ),
            "font": {"family": FONT, "color": c["text"], "size": 17},
            "x": 0, "xanchor": "left", "y": 0.96,
        },
        barmode="group", bargap=0.28, bargroupgap=0.04,
        margin={"l": 62, "r": 20, "t": 108, "b": 64},
        legend={"y": 1.06},
        width=1000, height=520,
    )
    # 2px surface gap between adjacent fills, per the mark spec.
    fig.update_traces(marker_line_color=c["surface"], marker_line_width=2,
                      cliponaxis=False)
    # Both series are a share that lasted -- one predicted, one observed -- so the
    # axis must not say "actually". Buckets are cut on the raw prediction in both
    # panels; that is what makes the two panels comparable.
    fig.update_yaxes(range=[0, 112], ticksuffix="%", row=1, col=1,
                     title_text="share that lasted to the next turn")
    fig.update_yaxes(range=[0, 112], row=1, col=2)
    for col in (1, 2):
        fig.update_xaxes(title_text="raw predicted-survival bucket", row=1, col=col)
    for ann in fig.layout.annotations[:2]:
        ann.font.update(color=c["text"], size=13)
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--theme", default="light", choices=("light", "dark"))
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--scale", type=float, default=2.0, help="PNG pixel ratio")
    args = ap.parse_args()

    rows = load_buckets(BUCKETS)
    fig = build(rows, args.theme)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(args.out), scale=args.scale)
    print(f"wrote {args.out} ({sum(r['n'] for r in rows):,} predictions, "
          f"{len(rows)} buckets, {args.theme})")


if __name__ == "__main__":
    main()

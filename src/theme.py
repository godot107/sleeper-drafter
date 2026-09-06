"""Chart palette and chrome, defined once for both themes.

Values come from a validated categorical palette. The two series colours used
together anywhere in this dashboard -- "every other team" and "your team" on the
standings chart -- were checked with the palette validator and clear every gate
in both modes (all-pairs CVD dE 24.7 light / 26.8 dark against an >=8 target;
normal-vision 33.6 / 31.8 against a >=15 floor).

Everywhere else deliberately uses a **single hue**. The obvious design for the
value-cliff chart is one colour per position, but six categorical hues in a
small-multiple layout cannot clear colour-vision-deficiency separation. Faceting
solves it properly: each panel is titled with its position, so identity comes
from position on the page rather than from hue, and the colour channel is left
free. That is the recommended remedy, not a compromise.
"""

from __future__ import annotations

LIGHT = {
    "surface": "#fcfcfb",
    "plane": "#f9f9f7",
    "text": "#0b0b0b",
    "text_secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "border": "rgba(11,11,11,0.10)",
    "series_1": "#2a78d6",
    "series_2": "#eb6834",
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

DARK = {
    "surface": "#1a1a19",
    "plane": "#0d0d0d",
    "text": "#ffffff",
    "text_secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "border": "rgba(255,255,255,0.10)",
    "series_1": "#3987e5",
    "series_2": "#d95926",
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def palette(theme: str) -> dict[str, str]:
    return DARK if theme == "dark" else LIGHT


def layout_defaults(theme: str) -> dict:
    """Recessive chrome: hairline grid, no chart junk, hover always on."""
    c = palette(theme)
    return {
        "paper_bgcolor": c["surface"],
        "plot_bgcolor": c["surface"],
        "font": {"family": FONT, "color": c["text_secondary"], "size": 12},
        "margin": {"l": 56, "r": 16, "t": 40, "b": 44},
        "xaxis": {
            "gridcolor": c["grid"], "linecolor": c["axis"], "zeroline": False,
            "tickfont": {"color": c["muted"]}, "title": {"font": {"color": c["muted"]}},
        },
        "yaxis": {
            "gridcolor": c["grid"], "linecolor": c["axis"], "zeroline": False,
            "tickfont": {"color": c["muted"]}, "title": {"font": {"color": c["muted"]}},
        },
        "hoverlabel": {
            "bgcolor": c["surface"], "bordercolor": c["border"],
            "font": {"family": FONT, "color": c["text"], "size": 12},
        },
        "legend": {
            "orientation": "h", "yanchor": "bottom", "y": 1.02,
            "xanchor": "left", "x": 0,
            "font": {"color": c["text_secondary"]},
            "bgcolor": "rgba(0,0,0,0)",
        },
    }

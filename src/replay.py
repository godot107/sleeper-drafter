"""Replay a completed draft and score the engine against what actually happened.

This is the only honest test of the opponent model. ``--mock`` samples opponents
from the very distribution the model assumes, so it can validate arithmetic but
never the model; a finished draft is real humans, and their picks either matched
the survival predictions or they did not.

Three things are measured:

**Survival calibration.** At each of your picks, every candidate carries a
predicted probability of lasting until your next turn. The draft record says
whether each actually did. Bucketed, plus a Brier score.

**Top-recommendation urgency.** How often the #1 recommendation was gone by the
next turn. High is *good* -- it means "take him now" was right.

**Pick comparison.** The engine's choice against the pick actually made,
compared on **VORP** rather than raw points -- raw points would put a
quarterback's 296 beside a tight end's 101 and call the difference meaningful,
when the two are not on the same scale. Read even the VORP version loosely: it
holds the rest of the draft fixed, and in reality a different pick changes what
everyone else takes. It is a sanity check, not a counterfactual.

Results append to a calibration log, because one draft is roughly fifteen
independent events -- nowhere near enough to fit a correction on. Several drafts
might be.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .optimizer import rank_with_lookahead
from .state import DraftState

logger = logging.getLogger(__name__)

BUCKETS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001]
BUCKET_LABELS = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]


def replay(
    draft: dict,
    picks: list[dict],
    board: pd.DataFrame,
    my_slot: int,
    *,
    candidates: int = 60,
) -> dict:
    """Replay ``picks`` pick by pick, scoring predictions against outcomes."""
    picks = sorted(picks, key=lambda p: p["pick_no"])
    taken_at = {str(p["player_id"]): p["pick_no"] for p in picks}
    my_picks = [p["pick_no"] for p in picks if p.get("draft_slot") == my_slot]
    if not my_picks:
        raise ValueError(f"slot {my_slot} made no picks in this draft")

    actual_by_pick = {p["pick_no"]: str(p["player_id"]) for p in picks}
    indexed = board.set_index("player_id")
    points = indexed["proj_pts"].to_dict()
    vorp = indexed["vorp"].to_dict()
    names = indexed["name"].to_dict()

    observations: list[dict] = []
    comparisons: list[dict] = []

    state = DraftState(draft, board, league=None)
    for pick_no in range(1, len(picks) + 1):
        if pick_no in my_picks:
            ranked, _ = rank_with_lookahead(state, board, my_slot)
            later = [x for x in my_picks if x > pick_no]
            if later and not ranked.empty:
                target = later[0]
                for _, row in ranked.head(candidates).iterrows():
                    gone_at = taken_at.get(row["player_id"])
                    observations.append({
                        "pick_no": pick_no,
                        "predicted": float(row["survival"]),
                        "survived": bool(gone_at is None or gone_at >= target),
                    })

            if not ranked.empty:
                top = ranked.iloc[0]
                actual = actual_by_pick.get(pick_no)
                gone_at = taken_at.get(top["player_id"], 10**9)
                comparisons.append({
                    "pick_no": pick_no,
                    "engine": names.get(top["player_id"], top["player_id"]),
                    "engine_pos": top["pos"],
                    "engine_pts": float(top["proj_pts"]),
                    "engine_vorp": float(top["vorp"]),
                    "engine_survival": float(top["survival"]),
                    "engine_gone_by_next": bool(later and gone_at < later[0]),
                    "actual": names.get(actual, actual),
                    "actual_pts": float(points.get(actual, float("nan"))),
                    "actual_vorp": float(vorp.get(actual, float("nan"))),
                    "agreed": actual == top["player_id"],
                })

        state.ingest([picks[pick_no - 1]])

    obs = pd.DataFrame(observations)
    obs["bucket"] = pd.cut(obs["predicted"], BUCKETS, labels=BUCKET_LABELS)
    calibration = (
        obs.groupby("bucket", observed=True)
        .agg(n=("survived", "size"),
             predicted=("predicted", "mean"),
             actual=("survived", "mean"))
        .reset_index()
    )
    brier = float(((obs["predicted"] - obs["survived"].astype(float)) ** 2).mean())

    return {
        "draft_id": draft.get("draft_id"),
        "teams": state.teams,
        "rounds": state.rounds,
        "scoring": state.scoring,
        "my_slot": my_slot,
        "n_observations": len(obs),
        "predicted_mean": float(obs["predicted"].mean()),
        "actual_mean": float(obs["survived"].mean()),
        "brier": brier,
        "calibration": calibration,
        "comparisons": pd.DataFrame(comparisons),
    }


def append_log(report: dict, path: Path) -> None:
    """Accumulate one line per replayed draft, so evidence builds up."""
    row = {k: report[k] for k in
           ("draft_id", "teams", "rounds", "scoring", "my_slot",
            "n_observations", "predicted_mean", "actual_mean", "brier")}
    row["bias"] = report["predicted_mean"] - report["actual_mean"]
    frame = pd.DataFrame([row])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        prior = pd.read_csv(path)
        prior = prior[prior["draft_id"].astype(str) != str(row["draft_id"])]
        frame = pd.concat([prior, frame], ignore_index=True)
    frame.to_csv(path, index=False)

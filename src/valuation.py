"""Player valuation, grounded in Isaac T. Petersen, *Fantasy Football Analytics*.

Three of the four concepts here come straight from the textbook, and the
definitions matter because the original spec left them vague:

**VORP** (Ch. 6) -- "the difference between (a) a player's projected fantasy
points and (b) the fantasy points that you would be expected to get from a
typical bench player at that position." Note *bench*, not last-starter: the
counterfactual to drafting a player is what you'd get off the waiver/bench pool
at that position, which sits a cohort below the starter cutoff.

**Dropoff** (Ch. 6, Ch. 7) -- "the difference between (a) the player's projected
points and (b) the projected points of the next-best player at that position."
This replaces the spec's arbitrary "tier cliff > 30 pts" constant with a
measured quantity. Petersen's empirical finding (§7.4.1) -- RB drops off
fastest, then TE, then QB after roughly the top 10, while K and DEF drop off
least -- is the reason the optimizer gates K/DEF to the final rounds.

**Tiers** (Ch. 21, Cluster Analysis) -- rather than thresholding, tiers are
found by 1-D clustering on projected points within a position. Implemented as
exact Fisher-Jenks natural breaks (a dynamic program), so tiers are
deterministic run to run -- which matters when you are reading the board under
a pick clock.

**Consistency** (Ch. 6, Eq. 6.1) -- CV = s/x̄ over weekly points. Petersen:
prefer low-uncertainty players for starting slots, high-uncertainty players for
the bench, where the upside is worth more than the downside costs.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------- dropoff

def add_dropoff(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``dropoff``: points lost stepping to the next-best player at the position.

    Petersen's definition. The last player at a position has no next-best, so
    the dropoff there is 0 rather than NaN -- it is the end of the cliff, not an
    unknown.
    """
    out = frame.copy()
    out["dropoff"] = (
        out.sort_values("proj_pts", ascending=False)
        .groupby("pos")["proj_pts"]
        .diff(-1)
        .reindex(out.index)
    )
    out["dropoff"] = out["dropoff"].fillna(0.0)
    return out


# --------------------------------------------------------------- replacement

def replacement_levels(
    frame: pd.DataFrame,
    starters_by_pos: dict[str, float],
    teams: int,
    *,
    cohort: int | None = None,
) -> dict[str, float]:
    """Projected points of a *typical bench player* at each position.

    League-wide, the first ``teams * starters`` players at a position are
    starters. The bench cohort is the ``teams`` players immediately after that;
    we take its median as the replacement level, which is more stable than
    pinning to a single rank.
    """
    cohort = cohort or teams
    levels: dict[str, float] = {}

    for pos, group in frame.groupby("pos"):
        ranked = group["proj_pts"].sort_values(ascending=False).to_numpy()
        starters = int(round(teams * starters_by_pos.get(pos, 0)))
        if ranked.size == 0:
            levels[pos] = 0.0
            continue
        window = ranked[starters : starters + cohort]
        if window.size == 0:
            # Thin position (K, DEF in a shallow pool): fall back to the tail.
            window = ranked[-min(cohort, ranked.size):]
        levels[pos] = float(np.median(window))

    return levels


def add_vorp(
    frame: pd.DataFrame,
    starters_by_pos: dict[str, float],
    teams: int,
) -> pd.DataFrame:
    """Add ``vorp`` and ``replacement_pts`` using Petersen's bench baseline."""
    levels = replacement_levels(frame, starters_by_pos, teams)
    out = frame.copy()
    out["replacement_pts"] = out["pos"].map(levels).fillna(0.0)
    out["vorp"] = out["proj_pts"] - out["replacement_pts"]
    return out


# --------------------------------------------------------------------- tiers

def _jenks_breaks(values: np.ndarray, k: int) -> list[int]:
    """Exact 1-D Fisher-Jenks: indices that start each of ``k`` classes.

    ``values`` must be sorted ascending. Returns class-start indices including 0.
    Dynamic program over within-class sum of squares, O(k*n^2) -- at n~215 and
    k~8 that is well under a millisecond, and unlike k-means it has no random
    initialisation to make tiers jump between refreshes.
    """
    n = values.size
    if k <= 1 or n <= k:
        return list(range(n))

    prefix = np.concatenate([[0.0], np.cumsum(values)])
    prefix_sq = np.concatenate([[0.0], np.cumsum(values**2)])

    def sse(i: int, j: int) -> float:
        """Sum of squared error for values[i:j] (half-open)."""
        count = j - i
        if count <= 1:
            return 0.0
        total = prefix[j] - prefix[i]
        total_sq = prefix_sq[j] - prefix_sq[i]
        return float(total_sq - total * total / count)

    # cost[m][j] = best cost splitting values[0:j] into m+1 classes
    cost = np.full((k, n + 1), np.inf)
    split = np.zeros((k, n + 1), dtype=int)

    for j in range(1, n + 1):
        cost[0][j] = sse(0, j)

    for m in range(1, k):
        for j in range(m + 1, n + 1):
            best, best_i = np.inf, m
            for i in range(m, j):
                candidate = cost[m - 1][i] + sse(i, j)
                if candidate < best:
                    best, best_i = candidate, i
            cost[m][j] = best
            split[m][j] = best_i

    starts, j = [], n
    for m in range(k - 1, 0, -1):
        starts.append(split[m][j])
        j = split[m][j]
    # sorted(set(...)): a degenerate split can repeat an index, which would
    # silently collapse two tiers into one.
    return sorted({0, *starts})


def assign_tiers(
    frame: pd.DataFrame,
    *,
    max_tiers: int = 8,
    min_per_position: int = 6,
    draftable_depth: int = 40,
) -> pd.DataFrame:
    """Add a 1-indexed ``tier`` per position (tier 1 = best).

    Only the *draftable* pool is clustered. Clustering all 45 projected kickers
    or all 215 receivers wastes the tier budget on players nobody will take:
    the interesting structure is at the top of the board. Everyone past the
    draftable depth lands in one trailing "deep" tier.
    """
    out = frame.copy()
    out["tier"] = 0

    for pos, group in out.groupby("pos"):
        if len(group) < min_per_position:
            out.loc[group.index, "tier"] = 1
            continue

        ordered = group.sort_values("proj_pts", ascending=False)
        depth = min(len(ordered), draftable_depth)
        head, tail = ordered.iloc[:depth], ordered.iloc[depth:]

        ascending = head["proj_pts"].to_numpy()[::-1]  # Jenks wants ascending
        k = max(3, min(max_tiers, depth // 5))
        starts = _jenks_breaks(ascending, k)

        # Map ascending class starts back onto descending rank order.
        tier_asc = np.zeros(ascending.size, dtype=int)
        for tier_index, start in enumerate(starts):
            tier_asc[start:] = tier_index
        # Ascending index 0 is the worst player, so invert to make tier 1 best.
        tier_desc = (len(starts) - tier_asc)[::-1]
        out.loc[head.index, "tier"] = tier_desc
        if len(tail):
            out.loc[tail.index, "tier"] = len(starts) + 1

    return out


# --------------------------------------------------------------- consistency

def add_consistency(
    frame: pd.DataFrame,
    consistency: pd.DataFrame | None,
    *,
    volatile_quantile: float = 0.67,
    steady_quantile: float = 0.33,
) -> pd.DataFrame:
    """Merge weekly CV and label risk within position.

    Petersen (Ch. 6): uncertainty is a risk profile, not a defect. ``steady``
    players are what you want in locked starting slots; ``volatile`` players are
    the sleeper profile you want late, on the bench, where upside is cheap.
    Players with no prior-season history (rookies) are ``unknown`` -- absence of
    a variance estimate is not the same as low variance.
    """
    out = frame.copy()
    if consistency is None or consistency.empty:
        out["wk_cv"] = np.nan
        out["games_played"] = np.nan
        out["risk"] = "unknown"
        return out

    cons = consistency.copy()
    cons["player_id"] = cons["player_id"].astype(str)
    out["player_id"] = out["player_id"].astype(str)
    out = out.merge(
        cons[["player_id", "wk_cv", "wk_sd", "wk_mean", "games_played"]],
        on="player_id", how="left",
    )

    out["risk"] = "unknown"
    for pos, group in out.groupby("pos"):
        cv = group["wk_cv"].dropna()
        if len(cv) < 5:
            continue
        low, high = cv.quantile(steady_quantile), cv.quantile(volatile_quantile)
        idx = group.index
        out.loc[idx[group["wk_cv"] <= low], "risk"] = "steady"
        out.loc[idx[(group["wk_cv"] > low) & (group["wk_cv"] < high)], "risk"] = "neutral"
        out.loc[idx[group["wk_cv"] >= high], "risk"] = "volatile"

    return out


def build_board(
    projections: pd.DataFrame,
    starters_by_pos: dict[str, float],
    teams: int,
    *,
    consistency: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Full valuation pass: VORP, dropoff, tiers, consistency.

    Positions the league cannot roster are dropped first. Sleeper's RB query
    returns fullbacks, and an FB has no replacement level worth computing --
    left in, one can surface in the recommendations.
    """
    rosterable = {p for p, n in starters_by_pos.items() if n}
    unknown = set(projections["pos"].unique()) - rosterable
    if unknown:
        logger.info("dropping %d non-rosterable positions: %s",
                    len(unknown), ", ".join(sorted(unknown)))
        projections = projections[projections["pos"].isin(rosterable)]

    board = add_vorp(projections, starters_by_pos, teams)
    board = add_dropoff(board)
    board = assign_tiers(board)
    board = add_consistency(board, consistency)
    board = add_opportunity(board)
    board = add_ceiling(board, consistency)
    return board.sort_values("vorp", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------- opportunity

def add_opportunity(frame: pd.DataFrame) -> pd.DataFrame:
    """Add ``opportunity`` — how much of his position's volume a player commands.

    Grew out of a live draft. The consistency label (CV) turned out to measure
    the wrong thing: tested against 231 players with 10+ games, CV correlated
    **−0.57** with points per game and **−0.20** with upside above a player's own
    average. Because CV is ``sd / mean``, the denominator dominates, so it mostly
    flags low-volume players whose scores bounce around near zero rather than
    genuine boom-or-bust starters.

    Volume is what CV was reaching for and missing. Two signals, both already in
    the projections payload:

    * ``depth_chart_order`` — 1 means he is his own team's starter. Fully
      populated (96 players at each of orders 1, 2 and 3).
    * projected touches (``rec + rush_att``), scaled within position.

    Caveat on receptions: Sleeper projects no target share, so receptions stand
    in. That bakes in catch rate, and therefore understates high-target
    receivers in poor offences. Targets would be the better stat; they are not
    on offer.
    """
    out = frame.copy()
    if "proj_rec" not in out.columns:
        out["opportunity"] = np.nan
        out["role"] = "unknown"
        return out

    touches = out["proj_rec"].fillna(0) + out["proj_rush_att"].fillna(0)
    out["touches"] = touches
    # Percentile within position, so a tight end is not judged against a back.
    out["opportunity"] = out.groupby("pos")["touches"].rank(pct=True)

    depth = out.get("depth_chart_order")
    starter = depth.fillna(9) <= 1 if depth is not None else pd.Series(False, index=out.index)
    out["role"] = np.where(
        starter & (out["opportunity"] >= 0.5), "workhorse",
        np.where(starter, "starter",
                 np.where(out["opportunity"] >= 0.5, "committee", "backup")),
    )
    return out


def add_ceiling(frame: pd.DataFrame, consistency: pd.DataFrame | None) -> pd.DataFrame:
    """Add ``ceiling`` — a player's 90th-percentile week last season.

    The late-round question is not "how steady is he" but "how high can he go".
    Because a bench player you can drop is effectively a call option — bounded
    downside, real upside — variance is genuinely worth paying for late. But CV
    does not find it: the volatile third of the league had a *lower* absolute
    ceiling (9.4 pts) than the steady third (15.4). Ceiling measures it directly.
    """
    out = frame.copy()
    if consistency is None or consistency.empty or "wk_ceiling" not in consistency.columns:
        out["ceiling"] = np.nan
        return out
    cons = consistency[["player_id", "wk_ceiling"]].copy()
    cons["player_id"] = cons["player_id"].astype(str)
    out["player_id"] = out["player_id"].astype(str)
    out = out.merge(cons, on="player_id", how="left")
    out["ceiling"] = out["wk_ceiling"]
    # Raw ceiling is not comparable across positions -- quarterbacks simply
    # score more, so an unranked list is just a list of quarterbacks. Rank it
    # within position, the same way opportunity is scaled.
    out["ceiling_pct"] = out.groupby("pos")["ceiling"].rank(pct=True)
    return out

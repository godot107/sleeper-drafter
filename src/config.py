"""Single place to read configuration, so the CLI, tests, and any future
scheduler all behave identically.

Mirrors the orchestrator-agnostic ``Settings`` pattern from
``energy-batch-trader`` and ``financial-forecasting-engine``.

No API keys are required to run: every Sleeper endpoint this project touches is
public and unauthenticated. The env vars below only override defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - optional dependency
    pass

_ROOT = Path(__file__).resolve().parent.parent


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    return float(raw) if raw is not None else default


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    return int(raw) if raw is not None else default


@dataclass
class Settings:
    """Paths, API behaviour, and every tunable coefficient in the model."""

    # --- Paths (repo-relative, so runs work from any cwd) ---
    root: Path = _ROOT
    data_dir: Path = field(default_factory=lambda: _ROOT / "data")
    cache_dir: Path = field(default_factory=lambda: _ROOT / "data" / "cache")
    players_cache: Path = field(default_factory=lambda: _ROOT / "data" / "cache" / "players.json")
    projections_csv: Path = field(default_factory=lambda: _ROOT / "data" / "projections.csv")

    # --- Season ---
    season: str = field(default_factory=lambda: _env("SLEEPER_SEASON", "2026"))

    # --- API behaviour ---
    # api.sleeper.com returns 403 to the default "Python-urllib/3.x" UA, so an
    # explicit User-Agent is mandatory, not cosmetic.
    user_agent: str = "sleeper-drafter/0.1 (+https://github.com/godot107/sleeper-drafter)"
    # Adaptive polling. Sleeper's documented ceiling is 1000 calls/min and
    # picks are fetched conditionally (304, zero bytes) when nothing has moved,
    # so even the fast tier is ~60/min of near-empty requests. Polling slowly
    # is the real risk: a pick clock is often 30-60s, so a 15s interval can
    # leave you looking at a board that is half a clock out of date.
    poll_interval_s: float = field(default_factory=lambda: _env_float("POLL_INTERVAL_S", 2.0))
    poll_interval_near_s: float = 1.0    # your turn is imminent
    poll_interval_far_s: float = 4.0     # many picks away
    poll_near_threshold: int = 3         # picks-until-turn considered "near"
    # The picks endpoint is edge-cached (s-maxage=300). If the CDN keeps
    # serving us an aged response we are drafting off stale data, so surface it.
    stale_age_warn_s: float = 45.0
    request_timeout_s: float = field(default_factory=lambda: _env_float("REQUEST_TIMEOUT_S", 10.0))
    max_retries: int = field(default_factory=lambda: _env_int("MAX_RETRIES", 4))
    players_max_age_h: int = 24  # docs: call /players/nfl "not more than once per day"

    # --- Opponent model ---
    # σ grows with pick number: round 1 is predictable, round 10 is not.
    adp_sigma: float = field(default_factory=lambda: _env_float("ADP_SIGMA", 8.0))
    adp_sigma_frac: float = field(default_factory=lambda: _env_float("ADP_SIGMA_FRAC", 0.35))
    adp_sigma_min: float = field(default_factory=lambda: _env_float("ADP_SIGMA_MIN", 2.5))
    consideration_set: int = 24    # how many players an opponent realistically weighs
    urgency_min: float = 0.10          # floor for any unfilled position
    urgency_filled: float = 0.05       # starters already filled -> bench-only interest
    # Survival calibration. The product form assumes intermediate picks are
    # independent; real drafts run, so raw survival is optimistic exactly in
    # the contested middle. Fitted on four completed drafts
    # (data/calibration.csv, 3,360 predictions): observation-weighted mean
    # absolute calibration error 0.066 -> 0.021, improving every bucket.
    # 1.0 disables. Refit as `--replay` evidence accumulates.
    survival_gamma: float = field(
        default_factory=lambda: _env_float("SURVIVAL_GAMMA", 4.11))

    # --- Optimizer: roster fit (projected-point units, so they are
    # commensurate with VORP) ---
    starter_bonus: float = 12.0    # fills an unfilled dedicated starting slot
    flex_bonus: float = 6.0        # fills FLEX / SUPER_FLEX
    depth_penalty: float = 18.0    # per body past what you can start, scaled by round
    risk_weight: float = 4.0       # Petersen Ch.6 steady/volatile nudge
    depth_penalty_floor: float = 0.35  # keep depth costly even in the last round

    # --- Optimizer ---
    denial_lambda: float = field(default_factory=lambda: _env_float("DENIAL_LAMBDA", 0.25))
    tier_cliff_pts: float = field(default_factory=lambda: _env_float("TIER_CLIFF_PTS", 30.0))
    run_warning_opponents: int = 2     # >=N high-urgency opponents at one position
    kdef_gate_from_end: int = 2        # block K/DEF until the last N rounds

    # --- Roster caps: how many of a position you will ever roster ---
    # Willie's rule after three mock drafts: finish with one QB, one TE, one K
    # and one DEF, and spend every other pick on RB/WR. All four are streamable
    # off waivers, and the engine kept wanting a backup at each because a
    # shallow position has a steep VONA even when the slot is already filled.
    # Once a cap is hit the position is blocked outright.
    roster_caps: dict = field(default_factory=lambda: {
        "QB": 1, "TE": 1, "K": 1, "DEF": 1,
    })

    # --- Personal exclusions ---
    do_not_draft_path: Path = field(
        default_factory=lambda: _ROOT / "data" / "do_not_draft.txt")

    # --- Undrafted sentinel used by Sleeper's projections payload ---
    undrafted_adp: float = 999.0


settings = Settings()

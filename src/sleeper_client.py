"""Sleeper API connector.

Read-only by design. Sleeper's public API has no write endpoints, so this
project is a *co-pilot*: it observes the draft and ranks options, and Willie
clicks the pick himself in the Sleeper UI. There is no code path here that
attempts to draft, trade, or modify a roster.

Two behaviours here are not optional, both found by probing the live API:

1. ``api.sleeper.com`` returns **403** to the default ``Python-urllib/3.x``
   User-Agent. Any explicit UA works. We always send one.
2. Error shapes are inconsistent. A bad username returns **HTTP 200 with a
   JSON body of ``null``**; a bad league's drafts return ``200 []``; a bad
   draft id returns ``404 null``. So ``response.ok`` is never a sufficient
   check -- every payload is null-validated in :func:`_get_json`.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import requests

from .config import settings

logger = logging.getLogger(__name__)

# The v1 API (drafts, leagues, users, players) and the projections host are
# different origins with different behaviour; keep both explicit.
API_V1 = "https://api.sleeper.app/v1"
API_PROJECTIONS = "https://api.sleeper.com"

FANTASY_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")


class SleeperError(RuntimeError):
    """Any unrecoverable problem talking to Sleeper."""


class NotFound(SleeperError):
    """The resource does not exist -- including the 200-with-null-body case."""


class SleeperClient:
    """Thin, retrying, single-session client."""

    def __init__(self, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.update(
            {"User-Agent": settings.user_agent, "Accept": "application/json"}
        )

    # ------------------------------------------------------------------ core

    def _get_json(self, url: str, *, params: dict | None = None) -> Any:
        """GET ``url`` and return parsed JSON, retrying transient failures.

        Raises :class:`NotFound` on 404 *and* on a ``null`` body, which Sleeper
        uses for "no such user" while still returning HTTP 200.
        """
        last_exc: Exception | None = None

        for attempt in range(settings.max_retries):
            try:
                resp = self.session.get(
                    url, params=params, timeout=settings.request_timeout_s
                )
            except requests.RequestException as exc:  # network blip, DNS, timeout
                last_exc = exc
                logger.warning("request failed (%s), retrying: %s", attempt + 1, exc)
                self._backoff(attempt)
                continue

            if resp.status_code == 404:
                raise NotFound(f"404 from {url}")

            # 429 = rate limited (docs: stay under 1000 calls/min), 5xx = transient.
            if resp.status_code == 429 or resp.status_code >= 500:
                last_exc = SleeperError(f"HTTP {resp.status_code} from {url}")
                logger.warning("HTTP %s from %s, retrying", resp.status_code, url)
                self._backoff(attempt)
                continue

            if not resp.ok:
                raise SleeperError(f"HTTP {resp.status_code} from {url}")

            try:
                payload = resp.json()
            except json.JSONDecodeError as exc:
                raise SleeperError(f"non-JSON body from {url}") from exc

            # The gotcha: 200 OK with a literal null body.
            if payload is None:
                raise NotFound(f"empty (null) payload from {url}")

            return payload

        raise SleeperError(f"giving up on {url} after {settings.max_retries} attempts") from last_exc

    @staticmethod
    def _backoff(attempt: int) -> None:
        delay = min(2.0**attempt, 8.0) + random.uniform(0, 0.4)
        time.sleep(delay)

    # --------------------------------------------------------------- players

    def players(self, *, refresh: bool = False) -> dict[str, dict]:
        """Return the full player directory, cached on disk.

        The payload is ~14.65 MB and the docs ask that it be fetched "not more
        than once per day", so this is snapshot-cached and never polled.
        """
        path: Path = settings.players_cache

        if path.exists() and not refresh:
            age_h = (time.time() - path.stat().st_mtime) / 3600
            if age_h < settings.players_max_age_h:
                logger.info("players cache hit (%.1fh old)", age_h)
                return json.loads(path.read_text())
            logger.info("players cache stale (%.1fh old), refetching", age_h)

        logger.info("fetching player directory (~15 MB)")
        data = self._get_json(f"{API_V1}/players/nfl")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data))
        return data

    def projections(self, position: str, season: str | None = None) -> list[dict]:
        """Season-long projections *and* ADP for one position.

        This is the undocumented-but-live endpoint that makes the whole project
        work offline-reproducibly: it returns ``pts_ppr``/``pts_half_ppr``/
        ``pts_std`` alongside ``adp_ppr``/``adp_half_ppr``/``adp_std``/``adp_2qb``,
        keyed by the same ``player_id`` as the directory above.
        """
        return self._get_json(
            f"{API_PROJECTIONS}/projections/nfl/{season or settings.season}",
            params={
                "season_type": "regular",
                "position[]": position,
                "order_by": "pts_ppr",
            },
        )

    # ---------------------------------------------------------------- drafts

    def draft(self, draft_id: str) -> dict:
        """Draft metadata: settings, draft_order, slot_to_roster_id, metadata."""
        return self._get_json(f"{API_V1}/draft/{draft_id}")

    def picks(self, draft_id: str) -> list[dict]:
        """Every pick made so far, oldest first."""
        return self._get_json(f"{API_V1}/draft/{draft_id}/picks")

    def traded_picks(self, draft_id: str) -> list[dict]:
        """Traded picks, so urgency is attributed to the roster actually picking.

        Without this, a traded pick makes the opponent model reason about the
        wrong team's roster needs.
        """
        try:
            return self._get_json(f"{API_V1}/draft/{draft_id}/traded_picks")
        except NotFound:
            return []

    def league(self, league_id: str) -> dict:
        """League object -- ``roster_positions`` here is the authoritative roster
        schema, including SUPER_FLEX, which the draft ``settings.slots_*`` keys
        do not expose."""
        return self._get_json(f"{API_V1}/league/{league_id}")

    # ------------------------------------------------------------- discovery

    def user(self, username_or_id: str) -> dict:
        """Look up a user. A bad username raises :class:`NotFound` (200 + null)."""
        return self._get_json(f"{API_V1}/user/{username_or_id}")

    def user_drafts(self, user_id: str, season: str | None = None) -> list[dict]:
        return self._get_json(
            f"{API_V1}/user/{user_id}/drafts/nfl/{season or settings.season}"
        )

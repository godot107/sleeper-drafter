"""Draft state: who has picked, who is left, and who picks next.

Three things in here are easy to get wrong and expensive to get wrong on draft
night, so each is isolated as a pure function with tests:

1. **Snake arithmetic.** Which draft slot owns global pick N, including linear
   drafts and third-round-reversal boards.
2. **The roster schema.** Taken from the league's ``roster_positions`` array,
   not the draft's ``settings.slots_*``. The documented slot keys are
   ``slots_qb/rb/wr/te/flex/def/k/bn`` -- there is no ``slots_super_flex``, so a
   superflex league is invisible from the draft object alone.
3. **Traded picks.** ``slot_to_roster_id`` says which roster *started* with a
   slot. If picks have been traded, the roster actually picking at (round, slot)
   differs, and attributing urgency to the wrong team's needs corrupts the whole
   opponent model.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

# Which positions may fill each flex-style slot Sleeper supports.
FLEX_ELIGIBILITY: dict[str, frozenset[str]] = {
    "FLEX": frozenset({"RB", "WR", "TE"}),
    "WRRB_FLEX": frozenset({"RB", "WR"}),
    "WRRB-FLEX": frozenset({"RB", "WR"}),
    "REC_FLEX": frozenset({"WR", "TE"}),
    "SUPER_FLEX": frozenset({"QB", "RB", "WR", "TE"}),
}

DEDICATED_POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")


# --------------------------------------------------------------- pick arithmetic

def round_for_pick(pick_no: int, teams: int) -> int:
    """1-indexed round containing 1-indexed global ``pick_no``."""
    if pick_no < 1:
        raise ValueError(f"pick_no must be >= 1, got {pick_no}")
    return (pick_no - 1) // teams + 1


def slot_for_pick(
    pick_no: int,
    teams: int,
    *,
    draft_type: str = "snake",
    reversal_round: int | None = None,
) -> int:
    """Return the 1-indexed draft slot that owns global ``pick_no``.

    ``linear`` drafts run 1..N every round. ``snake`` alternates direction.
    ``reversal_round`` (third-round reversal) inverts the direction from that
    round onward, so a 3RR board runs forward, reverse, **reverse**, forward...
    """
    if pick_no < 1:
        raise ValueError(f"pick_no must be >= 1, got {pick_no}")

    rnd = round_for_pick(pick_no, teams)
    idx = (pick_no - 1) % teams  # 0-indexed position within the round

    if draft_type == "linear":
        return idx + 1

    forward = rnd % 2 == 1
    if reversal_round and rnd >= reversal_round:
        forward = not forward

    return idx + 1 if forward else teams - idx


def picks_until_next_turn(
    my_slot: int,
    current_pick: int,
    teams: int,
    *,
    total_picks: int | None = None,
    draft_type: str = "snake",
    reversal_round: int | None = None,
) -> list[int]:
    """Global pick numbers strictly between ``current_pick`` and my next pick.

    This is the window the opponent model has to survive. On a turn boundary it
    is empty (back-to-back picks); mid-round in a 12-team league it is ~22 long.
    """
    picks: list[int] = []
    pick = current_pick + 1
    while total_picks is None or pick <= total_picks:
        slot = slot_for_pick(
            pick, teams, draft_type=draft_type, reversal_round=reversal_round
        )
        if slot == my_slot:
            return picks
        picks.append(pick)
        pick += 1
        if len(picks) > teams * 2 + 2:  # cannot happen in a well-formed board
            raise RuntimeError(f"no turn for slot {my_slot} within two rounds of {current_pick}")
    return picks


# ------------------------------------------------------------- roster schema

@dataclass
class RosterSchema:
    """Starting requirements for one team, derived from ``roster_positions``."""

    teams: int
    rounds: int
    starters: dict[str, int] = field(default_factory=dict)   # dedicated slots by position
    flex: dict[str, int] = field(default_factory=dict)       # flex slot name -> count
    bench: int = 0

    @property
    def is_superflex(self) -> bool:
        return self.flex.get("SUPER_FLEX", 0) > 0

    def flex_capacity(self, pos: str) -> int:
        """How many flex slots this position is eligible to fill."""
        return sum(
            n for name, n in self.flex.items()
            if pos in FLEX_ELIGIBILITY.get(name, frozenset())
        )

    def total_starters(self, pos: str) -> int:
        """Dedicated + flex-eligible starting slots. The denominator in urgency."""
        return self.starters.get(pos, 0) + self.flex_capacity(pos)

    @classmethod
    def from_league(cls, draft: dict, league: dict | None) -> "RosterSchema":
        settings = draft.get("settings") or {}
        teams = int(settings.get("teams") or 12)
        rounds = int(settings.get("rounds") or 15)

        positions = (league or {}).get("roster_positions")
        if positions:
            starters = {p: 0 for p in DEDICATED_POSITIONS}
            flex: dict[str, int] = {}
            bench = 0
            for slot in positions:
                if slot in ("BN", "IR", "TAXI"):
                    bench += 1
                elif slot in FLEX_ELIGIBILITY:
                    flex[slot] = flex.get(slot, 0) + 1
                elif slot in starters:
                    starters[slot] += 1
                else:  # IDP or anything unrecognised -- count it, never crash
                    logger.debug("unrecognised roster slot %r, treating as bench", slot)
                    bench += 1
            return cls(teams=teams, rounds=rounds, starters=starters, flex=flex, bench=bench)

        # Fallback: the draft object's slots_* keys. Parsed generically so an
        # unknown slots_<x> key cannot crash us -- but note this path cannot see
        # SUPER_FLEX at all, which is exactly why roster_positions is preferred.
        logger.warning("no roster_positions available; falling back to draft settings.slots_*")
        starters = {}
        flex = {}
        for key, value in settings.items():
            if not key.startswith("slots_") or not value:
                continue
            name = key[len("slots_"):].upper()
            if name == "BN":
                continue
            if name in FLEX_ELIGIBILITY:
                flex[name] = int(value)
            else:
                starters[name] = int(value)

        # Standalone mock drafts carry no `slots_bn`, which used to leave bench
        # at 0. That is not cosmetic: with no bench every round looks like a
        # starting slot, so the mandatory-slot constraint fires several rounds
        # early and starts forcing K/DEF around round 10 instead of 13. Infer
        # the bench from the rounds the starting slots cannot account for.
        declared_bench = settings.get("slots_bn")
        if declared_bench:
            bench = int(declared_bench)
        else:
            starting_slots = sum(starters.values()) + sum(flex.values())
            bench = max(0, rounds - starting_slots)
            logger.info("no slots_bn; inferred %d bench spots (%d rounds - %d starters)",
                        bench, rounds, starting_slots)

        return cls(teams=teams, rounds=rounds, starters=starters, flex=flex, bench=bench)


def scoring_format(draft: dict, league: dict | None = None) -> str:
    """Return ``ppr`` / ``half_ppr`` / ``std`` / ``2qb`` for this league.

    ``draft.metadata.scoring_type`` gives it directly. Falls back to the
    league's ``scoring_settings.rec`` (1.0 -> ppr, 0.5 -> half, 0 -> std).
    Superflex boards price QBs so differently that they get their own ADP
    column, so that wins over the receiving-points format.
    """
    schema = RosterSchema.from_league(draft, league)
    if schema.is_superflex:
        return "2qb"

    declared = ((draft.get("metadata") or {}).get("scoring_type") or "").lower()
    if declared in ("ppr", "half_ppr", "std"):
        return declared
    if "half" in declared:
        return "half_ppr"
    if declared:
        logger.debug("unrecognised scoring_type %r, inferring from scoring_settings", declared)

    rec = ((league or {}).get("scoring_settings") or {}).get("rec")
    if rec is None:
        return "ppr"
    rec = float(rec)
    if rec >= 0.75:
        return "ppr"
    return "half_ppr" if rec >= 0.25 else "std"


# ---------------------------------------------------------------- draft state

@dataclass
class Team:
    roster_id: int
    draft_slot: int
    roster: list[str] = field(default_factory=list)          # player_ids, draft order
    slot_counts: dict[str, int] = field(default_factory=dict)  # position -> count owned

    def unfilled_starters(self, schema: RosterSchema, pos: str) -> int:
        """Starting slots at ``pos`` this team still needs to fill.

        Dedicated slots first; players beyond them spill into flex capacity.
        """
        owned = self.slot_counts.get(pos, 0)
        return max(0, schema.total_starters(pos) - owned)


class DraftState:
    """Live view of one draft: the board, the rosters, and whose turn it is."""

    def __init__(
        self,
        draft: dict,
        projections: pd.DataFrame,
        *,
        league: dict | None = None,
        traded_picks: list[dict] | None = None,
    ) -> None:
        self.draft = draft
        self.draft_id = draft.get("draft_id")
        self.draft_type = (draft.get("type") or "snake").lower()
        self.schema = RosterSchema.from_league(draft, league)
        self.scoring = scoring_format(draft, league)

        settings = draft.get("settings") or {}
        self.teams = self.schema.teams
        self.rounds = self.schema.rounds
        self.total_picks = self.teams * self.rounds
        # Not documented for drafts, but present on boards that use it.
        self.reversal_round = settings.get("reversal_round") or None

        # slot -> roster_id, as the board started.
        self.slot_to_roster: dict[int, int] = {
            int(k): int(v) for k, v in (draft.get("slot_to_roster_id") or {}).items()
        }
        if not self.slot_to_roster:  # mock drafts often omit it
            self.slot_to_roster = {s: s for s in range(1, self.teams + 1)}

        # (round, original_roster_id) -> roster that now owns the pick.
        self._traded: dict[tuple[int, int], int] = {}
        for tp in traded_picks or []:
            try:
                self._traded[(int(tp["round"]), int(tp["roster_id"]))] = int(tp["owner_id"])
            except (KeyError, TypeError, ValueError):
                logger.warning("skipping malformed traded pick: %r", tp)

        self.all_players = projections.copy()
        self.all_players["player_id"] = self.all_players["player_id"].astype(str)
        self._pos_by_id: dict[str, str] = dict(
            zip(self.all_players["player_id"], self.all_players["pos"])
        )

        self.teams_by_roster: dict[int, Team] = {
            rid: Team(roster_id=rid, draft_slot=slot)
            for slot, rid in self.slot_to_roster.items()
        }

        self.drafted: set[str] = set()
        self.processed_pick_ids: set[str] = set()
        self.picks: list[dict] = []

    # ------------------------------------------------------------- ingestion

    def ingest(self, picks: list[dict]) -> list[dict]:
        """Apply any picks not already seen. Returns just the new ones.

        Keyed off ``roster_id`` rather than ``picked_by``, which is an empty
        string on autopicks and commissioner picks. Keepers arrive here as
        ordinary picks and are applied the same way.
        """
        new: list[dict] = []
        for pick in picks:
            key = str(pick.get("pick_no") or pick.get("player_id"))
            if key in self.processed_pick_ids:
                continue
            self.processed_pick_ids.add(key)
            self._apply(pick)
            new.append(pick)
            self.picks.append(pick)
        if new:
            logger.debug("ingested %d new picks (now at %d)", len(new), self.current_pick_no)
        return new

    def _apply(self, pick: dict) -> None:
        player_id = str(pick.get("player_id") or "")
        if not player_id:
            return
        self.drafted.add(player_id)

        roster_id = pick.get("roster_id")
        if roster_id is None:
            slot = pick.get("draft_slot")
            roster_id = self.slot_to_roster.get(int(slot)) if slot else None
        if roster_id is None:
            return

        team = self.teams_by_roster.setdefault(
            int(roster_id), Team(roster_id=int(roster_id), draft_slot=int(pick.get("draft_slot") or 0))
        )
        team.roster.append(player_id)

        pos = self._pos_by_id.get(player_id) or (pick.get("metadata") or {}).get("position")
        if pos:
            team.slot_counts[pos] = team.slot_counts.get(pos, 0) + 1

    # ---------------------------------------------------------------- views

    @property
    def current_pick_no(self) -> int:
        """1-indexed pick that is on the clock right now."""
        return len(self.drafted) + 1

    @property
    def available_players(self) -> pd.DataFrame:
        return self.all_players[~self.all_players["player_id"].isin(self.drafted)]

    def slot_at(self, pick_no: int) -> int:
        return slot_for_pick(
            pick_no, self.teams,
            draft_type=self.draft_type, reversal_round=self.reversal_round,
        )

    def roster_at(self, pick_no: int) -> int:
        """Roster id actually making ``pick_no``, honouring traded picks."""
        slot = self.slot_at(pick_no)
        original = self.slot_to_roster.get(slot, slot)
        rnd = round_for_pick(pick_no, self.teams)
        return self._traded.get((rnd, original), original)

    def team_at(self, pick_no: int) -> Team:
        rid = self.roster_at(pick_no)
        return self.teams_by_roster.setdefault(
            rid, Team(roster_id=rid, draft_slot=self.slot_at(pick_no))
        )

    def picks_until_my_turn(self, my_slot: int, current_pick: int | None = None) -> list[int]:
        return picks_until_next_turn(
            my_slot,
            current_pick if current_pick is not None else self.current_pick_no,
            self.teams,
            total_picks=self.total_picks,
            draft_type=self.draft_type,
            reversal_round=self.reversal_round,
        )

    def is_my_turn(self, my_slot: int) -> bool:
        return self.slot_at(self.current_pick_no) == my_slot

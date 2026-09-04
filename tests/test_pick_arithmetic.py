"""Guards the snake/linear/reversal pick arithmetic.

Every downstream number depends on knowing exactly which picks happen between
now and the user's next turn. If this is off by one, the survival window is
wrong, VONA is wrong, and the recommendations are wrong -- silently, and only
on draft night. Hence the exhaustive table-driven cases.
"""

import pytest

from src.state import picks_until_next_turn, round_for_pick, slot_for_pick


class TestRoundForPick:
    @pytest.mark.parametrize(
        "pick,teams,expected",
        [(1, 12, 1), (12, 12, 1), (13, 12, 2), (24, 12, 2), (25, 12, 3),
         (1, 10, 1), (10, 10, 1), (11, 10, 2), (100, 10, 10)],
    )
    def test_rounds(self, pick, teams, expected):
        assert round_for_pick(pick, teams) == expected

    def test_rejects_zero(self):
        with pytest.raises(ValueError):
            round_for_pick(0, 12)


class TestSnakeSlots:
    @pytest.mark.parametrize(
        "pick,expected",
        [
            # Round 1 runs forward 1..12
            (1, 1), (2, 2), (11, 11), (12, 12),
            # Round 2 runs back 12..1
            (13, 12), (14, 11), (23, 2), (24, 1),
            # Round 3 forward again
            (25, 1), (26, 2), (36, 12),
            # Round 4 back
            (37, 12), (48, 1),
        ],
    )
    def test_twelve_team(self, pick, expected):
        assert slot_for_pick(pick, 12) == expected

    @pytest.mark.parametrize(
        "pick,expected",
        [(1, 1), (10, 10), (11, 10), (20, 1), (21, 1), (30, 10)],
    )
    def test_ten_team(self, pick, expected):
        assert slot_for_pick(pick, 10) == expected

    def test_every_slot_used_once_per_round(self):
        for rnd in range(1, 16):
            slots = {slot_for_pick((rnd - 1) * 12 + i, 12) for i in range(1, 13)}
            assert slots == set(range(1, 13)), f"round {rnd} does not cover every slot"


class TestLinearSlots:
    @pytest.mark.parametrize("pick,expected", [(1, 1), (12, 12), (13, 1), (24, 12), (25, 1)])
    def test_linear_never_reverses(self, pick, expected):
        assert slot_for_pick(pick, 12, draft_type="linear") == expected


class TestThirdRoundReversal:
    """3RR: forward, reverse, **reverse**, forward, reverse..."""

    @pytest.mark.parametrize(
        "pick,expected",
        [
            (1, 1), (12, 12),        # R1 forward
            (13, 12), (24, 1),       # R2 reverse
            (25, 12), (36, 1),       # R3 reverse again -- the reversal
            (37, 1), (48, 12),       # R4 forward
            (49, 12), (60, 1),       # R5 reverse
        ],
    )
    def test_reversal_round_three(self, pick, expected):
        assert slot_for_pick(pick, 12, reversal_round=3) == expected

    def test_slot_one_gets_back_to_back_at_the_turn(self):
        # Without 3RR slot 1 pairs picks 24+25; with it, slot 12 does.
        assert slot_for_pick(24, 12, reversal_round=3) == 1
        assert slot_for_pick(25, 12, reversal_round=3) == 12


class TestPicksUntilNextTurn:
    def test_turn_boundary_is_back_to_back(self):
        # Slot 1 picks 24 then 25 -- nothing in between to survive.
        assert picks_until_next_turn(1, 24, 12) == []

    def test_slot_twelve_turn(self):
        assert picks_until_next_turn(12, 12, 12) == []

    def test_slot_one_full_wait(self):
        # From pick 1, slot 1 waits until 24: picks 2..23 intervene.
        assert picks_until_next_turn(1, 1, 12) == list(range(2, 24))

    def test_middle_slot_wait(self):
        # Slot 6 in round 1 picks 6; round 2 reverses so it picks 19.
        assert picks_until_next_turn(6, 6, 12) == list(range(7, 19))

    def test_window_never_exceeds_two_rounds(self):
        for slot in range(1, 13):
            for pick in range(1, 60):
                window = picks_until_next_turn(slot, pick, 12)
                assert len(window) <= 2 * 12 - 1

    def test_none_of_the_intervening_picks_are_mine(self):
        for slot in range(1, 13):
            window = picks_until_next_turn(slot, slot, 12)
            assert all(slot_for_pick(p, 12) != slot for p in window)

    def test_stops_at_end_of_draft(self):
        # 12x2 board: slot 1's last pick is 24, so from there the window is empty.
        assert picks_until_next_turn(1, 24, 12, total_picks=24) == []

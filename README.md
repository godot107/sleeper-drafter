# sleeper-drafter

A real-time draft co-pilot for Sleeper fantasy football. It watches a live draft, models
which players your opponents will take before your next turn, and tells you whether to
take a player **now** or whether he'll still be there.

That last distinction is the whole point. Static value rankings tell you who is good.
They don't tell you that the tight end you want will last another two rounds while the
running back one slot below him will not.

```
┏━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━┳━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━┳━━━━━━┳━━━━━━━┳━━━━━━━━━┓
┃    # ┃ Player                ┃ Pos ┃ Proj ┃  ADP ┃ Tier ┃ Surv% ┃ VORP ┃ VONA ┃ Score ┃ Risk    ┃
┡━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━╇━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━╇━━━━━━╇━━━━━━━╇━━━━━━━━━┩
│    1 │ Jahmyr Gibbs          │ RB  │  331 │  1.0 │    1 │    23 │  210 │   64 │   149 │ neutral │
│    2 │ Christian McCaffrey   │ RB  │  291 │  5.1 │    2 │    23 │  170 │   15 │    89 │ steady  │
│    4 │ Jonathan Taylor       │ RB  │  272 │  7.4 │    3 │    24 │  151 │   -7 │    59 │ steady  │
│   10 │ Nico Collins          │ WR  │  262 │ 25.0 │    3 │    89 │  115 │   -1 │    18 │ steady  │
└──────┴───────────────────────┴─────┴──────┴──────┴──────┴───────┴──────┴──────┴───────┴─────────┘
```

Gibbs at **+64** is the last elite back before a cliff — take him. Jonathan Taylor at
**−7** has more raw value than most of the board, but comparable backs will still be
there next turn. Nico Collins at 89% survival can wait.

## It does not draft for you

Sleeper's public API is read-only. There are no documented endpoints for submitting a
pick, and this project does not try to work around that — no session scraping, no
headless browser. You keep the veto, which matters when injury news breaks mid-draft and
the model has no idea.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/fetch_projections.py       # real projections + ADP
python scripts/fetch_consistency.py       # weekly-variance risk labels

python main.py --mock --slot 5            # try it offline
python main.py --find-draft <username>    # find your draft_id
python main.py --draft-id <id> --slot 5   # draft night, terminal

python main.py --web --mock --slot 12     # browser dashboard on a simulated draft
```

## Two front ends

The **terminal UI** (`rich`) is the reliable one: a single process, no browser, works
over SSH.

The **browser dashboard** (`--web`) adds interactive filters and four charts — a
take-now/wait scatter, value cliffs faceted by position, expected departures before your
turn, and live league grades. It states when data last changed and turns amber then red
as that ages, because a draft dashboard that has quietly stopped updating is worse than
no dashboard.

Both run at once, which is the intended multi-monitor setup: dashboard on one screen,
terminal on another, the Sleeper app on a third. The browser never calls Sleeper — one
background thread polls and publishes a snapshot that every view reads.

See [RUNBOOK.md](RUNBOOK.md) for the draft-night procedure.

## How it works

1. **Board.** Real season projections and ADP from Sleeper, joined to the player
   directory. 631 players with projections, 422 with a real ADP.
2. **Survival.** For every pick between now and your next turn, a normalised probability
   that each available player is taken, from opponent roster need × ADP proximity.
   Multiplied across the window.
3. **VONA.** Your points minus the expected best player at that position when you pick
   again — *excluding the candidate himself*, since drafting him is exactly what makes
   him unavailable.
4. **Denial + roster fit.** A small penalty to opponents for taking a player they need,
   and constraints that keep you from drafting a fourth running back or finishing without
   a kicker.

## Methodology

Grounded in [Isaac T. Petersen, *Fantasy Football
Analytics*](https://isaactpetersen.github.io/Fantasy-Football-Analytics-Textbook/):

| Concept | Source | Implementation |
|---|---|---|
| **VORP** | Ch. 6 | Measured against "a typical **bench** player at that position" — not the last starter. The median of the bench cohort just past the league-wide starter cutoff. |
| **Dropoff** | Ch. 6, 7 | Points minus the next-best player at the same position. Measured, replacing an arbitrary points threshold. |
| **Tiers** | Ch. 21 | Exact Fisher-Jenks natural breaks, chosen over k-means because it has no random initialisation — tiers must not reshuffle between refreshes while you read the board on a clock. |
| **Uncertainty** | Ch. 6, Eq. 6.1 | CV = s/x̄ over prior-season weekly points. Retained, but see the caveat below — it did not survive contact with the data. |
| **Opportunity** | — | `depth_chart_order` plus projected touches, ranked within position. Correlates **+0.64** with projected points where CV correlates **−0.47**. |
| **Ceiling** | — | 90th-percentile week from last season, ranked within position. The honest late-round upside measure. |
| **K/DEF late** | §7.4.1 | Kickers and defenses have the lowest measured dropoff, so waiting costs nothing. |
| **Runs** | Ch. 7 | "Avoid joining a run mid-stream" — why the denial coefficient stays low. |

## Known limitations

Stated plainly, because they affect how much to trust a given number.

- **Survival assumes picks are independent.** Real drafts have runs — three receivers in
  four picks. The model is therefore mildly *optimistic* about survival mid-run. The
  position-run panel is the mitigation, not a fix.
- **One projection source.** Petersen recommends aggregating across sources, and measures
  uncertainty as the spread between them. We have Sleeper only, so week-to-week variance
  from last season stands in. It measures a real thing, but not the same thing.
- **Rookies have no risk label.** No prior-season weeks means no variance estimate. They
  are marked `unknown` rather than `steady`; absence of evidence is not low risk.
- **Opponents are modelled as ADP-followers with roster needs.** Managers who reach, or
  who draft their own team's players, will not be predicted well.
- **Offline `--mock` cannot validate the opponent model.** It samples opponents from the
  very distribution the model assumes. It tests the arithmetic. Only a live Sleeper mock
  draft tests the model.
- **Projections are a point estimate of a noisy quantity.** A 12-point VONA edge is inside
  the noise. Treat the tiers and the big cliffs as signal; treat small gaps as ties and
  use your own read.
- **Survival runs ~5 points optimistic.** Measured by replaying a completed 10-team draft:
  predicted 91.0% mean survival against 85.8% actual, Brier 0.079. The mid-range is worst
  — players the model calls 40-60% survived only 19% of the time. That is the independence
  assumption showing up exactly where predicted. A single-parameter correction would fit
  the overall mean but overcorrects the low bucket, and one draft is ~15 independent
  events, so nothing is baked in yet; `--replay` exists to accumulate the evidence.
- **The `risk` label is weaker than it looks.** CV measures week-to-week bounce, not
  upside. Prefer `role` (depth chart + volume) for quality and `ceiling` for late-round
  upside — see below.

### Where the textbook's metric failed

Petersen's uncertainty measure (Ch. 6) is the spread of a player's projections *across
sources*. We have one source, so week-to-week scoring variance stood in. Tested on 231
players with 10+ games in 2025, that substitution does not measure what the draft guidance
is about:

| Tercile | Pts/wk | CV | Ceiling (p90) |
|---|---|---|---|
| steady | 9.00 | 0.60 | **15.36** |
| volatile | 4.09 | 1.09 | **9.37** |

The volatile third has a *lower* ceiling, and `corr(CV, points) = −0.57`. Because CV is
`sd / mean`, the denominator dominates: it flags low-volume players whose scores bounce
near zero, not boom-or-bust starters. So `role` and `ceiling` were added to measure
opportunity and upside directly, and `risk` is kept only as what it actually is —
week-to-week bounce.

## Development

```bash
pytest -q     # 99 tests
```

The pick arithmetic, survival normalisation, VONA exclusion, and roster-legality
invariants are all pinned. Several were written after a bug: the first mock draft
finished with no quarterback, no kicker and no defense, and nothing in the ranking maths
was wrong.

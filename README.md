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

python scripts/fetch_projections.py --scoring half_ppr   # projections + ADP
python scripts/fetch_consistency.py       # weekly-variance risk labels

python main.py --mock --slot 5            # try it offline
python main.py --find-draft <username>    # find your draft_id
python main.py --draft-id <id> --slot 5   # draft night, terminal

python main.py --web --mock --slot 12     # browser dashboard on a simulated draft
python main.py --replay <draft_id>        # score the model against a finished draft

python scripts/experiment_cv.py           # why the CV risk label was replaced
python scripts/experiment_first_two.py    # first-two-picks strategy, by draft slot
```

For the write-up — the theory, the credits, and the things that turned out to be wrong —
see **[blog.md](blog.md)**. For the raw measured findings, **[docs/FINDINGS.md](docs/FINDINGS.md)**.

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
| **Opportunity** | — | `depth_chart_order` plus projected touches, ranked within position. Correlates **+0.80** with projected points where CV correlates **−0.50** (312 players, 10+ games). |
| **Ceiling** | — | 90th-percentile week from last season, ranked within position. The honest late-round upside measure. |
| **K/DEF late** | §7.4.1 | Kickers and defenses have the lowest measured dropoff, so waiting costs nothing. |
| **Runs** | Ch. 7 | "Avoid joining a run mid-stream" — why the denial coefficient stays low. |

## Tuning it to how you draft

Two things the projections cannot know:

**`data/do_not_draft.txt`** — one name per line, `#` comments allowed. Injury history, a
holdout, a player you have simply seen enough of. Excluded players are blocked from
recommendations but stay on the board, so the pick feed still names them when someone else
takes them.

**Scoring format.** `--scoring` picks which of Sleeper's precomputed columns the board is
built from: `ppr`, `half_ppr`, `std`, or `2qb` for superflex. The format is stamped into
`data/projections.csv`, and a live draft against a league in a different format is refused
rather than silently mispriced — override with `--ignore-scoring-mismatch` if you must.

Everything below the reception value is already baked into Sleeper's projection. Rescoring
all 554 skill players from raw component stats (`pass_yd`, `pass_td`, `pass_int`, `rush_td`,
`rec_yd`, `fum_lost`, the 2-pointers) with a typical league's coefficients reproduces
`pts_half_ppr` exactly, to the cent, for every one of them but Travis Hunter — whose
projection carries IDP credit most leagues do not award. So the reception value is the only
setting worth configuring. Kicker FG-distance tiers and defensive points-allowed tiers
cannot be checked at all (Sleeper does not publish those splits), and do not need to be:
they are the flattest positions on the board and go in the last two rounds.

**`roster_caps` in `src/config.py`** — how many of a position you will ever roster.
Defaults to one each of QB, TE, K and DEF, on the reasoning that all four stream off
waivers and every other pick is better spent on RB/WR. Raise or remove them for a
superflex or TE-premium league.

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
- **Survival is calibrated, not raw.** The product form assumes intermediate picks are
  independent; real drafts run. Measured over four completed drafts (3,360 predictions in
  `data/calibration.csv`), raw survival ran **+6.7 points optimistic**, concentrated in the
  contested middle: players called 40-60% survived 14% of the time, 60-80% survived 32%.
  A single exponent, `config.survival_gamma = 4.11`, cuts observation-weighted calibration
  error from 0.066 to 0.021 and improves every bucket; across the four drafts mean bias
  goes +0.066 → +0.005 and mean Brier 0.061 → 0.044. An exponent (not a linear shrink) is
  used because `p**g` fixes 0 and 1, so a turn boundary stays exactly certain. It is
  deliberately **not** A/B tested in `--mock` — mock opponents are drawn from
  `selection_probs` itself, so a mock has none of the herding this corrects. Set
  `SURVIVAL_GAMMA=1.0` to disable. It still slightly worsens the one 10-team standard
  draft while clearly helping the three 12-team half-PPR ones, so gamma may need to vary
  with league size; four drafts is too few to fit that. Keep running `--replay`.
- **The `risk` label is weaker than it looks.** CV measures week-to-week bounce, not
  upside. Prefer `role` (depth chart + volume) for quality and `ceiling` for late-round
  upside — see below.

### Where the textbook's metric failed

Petersen's uncertainty measure (Ch. 6) is the spread of a player's projections *across
sources*. We have one source, so week-to-week scoring variance stood in. Tested on the 314
players with 10+ games in 2025 (`python scripts/experiment_cv.py`), that substitution does
not measure what the draft guidance is about:

| Tercile | Pts/wk | CV | Ceiling (p90) |
|---|---|---|---|
| steady | 12.69 | 0.46 | **19.35** |
| neutral | 9.29 | 0.63 | 16.36 |
| volatile | 6.69 | 0.90 | **13.83** |

The volatile third has a *lower* ceiling, and `corr(CV, points/wk) = −0.58`. Because CV is
`sd / mean`, the denominator dominates: it flags low-volume players whose scores bounce
near zero, not boom-or-bust starters. So `role` and `ceiling` were added to measure
opportunity and upside directly, and `risk` is kept only as what it actually is —
week-to-week bounce.

## Development

```bash
pytest -q     # 163 tests
```

The pick arithmetic, survival normalisation, VONA exclusion, and roster-legality
invariants are all pinned. Several were written after a bug: the first mock draft
finished with no quarterback, no kicker and no defense, and nothing in the ranking maths
was wrong.

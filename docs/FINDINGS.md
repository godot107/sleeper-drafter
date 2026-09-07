# Measured findings

Everything here was measured against real drafts, not reasoned about. Each entry
records what was believed, what the data said, and what changed as a result.
Re-verify before assuming any of it still holds — most of it is a property of one
season's player pool.

Draft identifiers are hashed (`replay.draft_key`). The leagues contain real
people who did not agree to appear in a public repository.

---

## Survival was optimistic, and is now calibrated

The product form assumes intermediate picks are independent. Real drafts run —
managers watch each other. Over four completed drafts (3,360 predictions,
`data/calibration.csv`) raw survival ran **+6.7 points optimistic**, concentrated
exactly where independence is weakest:

| bucket | n | predicted | actual | corrected |
|---|---:|---:|---:|---:|
| 0–20% | 19 | 0.159 | 0.053 | 0.001 |
| 20–40% | 74 | 0.310 | 0.081 | 0.008 |
| 40–60% | 113 | 0.520 | 0.142 | 0.068 |
| 60–80% | 192 | 0.685 | 0.318 | 0.212 |
| 80–100% | 2962 | 0.994 | 0.964 | 0.975 |

`config.survival_gamma = 4.11`, applied as `p**gamma`. Observation-weighted mean
absolute calibration error **0.066 → 0.021**, improving every bucket; across the
four drafts mean bias **+0.066 → +0.005** and mean Brier **0.061 → 0.044**.

An exponent rather than a linear shrink because `p**g` fixes 0 and 1 — a turn
boundary ("empty window, he is certainly there") must stay exactly 1.0.

With three drafts this was **not** fittable; a single exponent overshot the
middle. It waited for a fourth.

**Never tune this against `--mock`.** Mock opponents are sampled from
`selection_probs` itself, so a simulated draft contains none of the herding the
correction exists for and will score it as a regression. Live drafts are the only
valid evidence.

**Open caveat:** it slightly *worsens* the one 10-team standard-scoring draft
(Brier 0.071 → 0.074) while clearly helping the three 12-team half-PPR ones.
Gamma likely wants to vary with league size; four drafts is too few to fit that.

## "Real drafts run more than mock rooms" — tested, not supported

A plausible hypothesis after a real-league draft went badly at running back.
Measured `p(pick matches the previous pick's position)` across all four drafts:

| draft | kind | p(same as prev) | RB in rd 1 | RB in rds 1–3 |
|---|---|---:|---:|---:|
| 3f0d0455 | mock room | 0.356 | 7 | 17 |
| b78910a2 | mock room | 0.240 | 7 | 17 |
| efae8aa2 | mock room | 0.263 | 6 | 16 |
| fcc27074 | **real league** | **0.218** | 7 | **20** |

The real league ran *least*. Round-1 RB count was identical. What actually
differed: **RB taken through three rounds — 20 of 36 against 16–17.** The run did
not come faster, it ran deeper. So the fix was a calibrated survival model, not a
separate "real humans" opponent model.

## CV was the wrong uncertainty metric

Petersen's Eq. 6.1 measures spread *across projection sources*. With one source,
week-to-week variance was substituted. It does not measure the same thing
(`scripts/experiment_cv.py`, 314 players with 10+ games):

| tercile | pts/wk | CV | ceiling (p90) |
|---|---:|---:|---:|
| steady | 12.69 | 0.46 | **19.35** |
| neutral | 9.29 | 0.63 | 16.36 |
| volatile | 6.69 | 0.90 | **13.83** |

The volatile third has the *lower* ceiling. `corr(CV, points/wk) = −0.58`.
Because CV is sd/mean, the denominator dominates: it finds low-volume players
whose scores bounce near zero, not boom-or-bust starters.

**Replaced by** `role` (depth-chart order + projected touches) and `ceiling`
(90th-percentile week). On the 312 players with 10+ games, correlation with
projected points: **CV −0.50, role +0.80, ceiling +0.80.** Both metrics were
already in payloads being fetched and went unused for weeks.

## Flex accounting caused two separate bugs, both favouring TE

1. `flex_capacity` counted a single FLEX slot in full for RB, WR *and* TE, so
   every VORP baseline was drawn too deep — worst at TE, whose pool is shallow.
   Result: 3.5 tight ends drafted per draft for one starting slot.
2. `best_lineup` credited an empty *dedicated* slot at replacement but an empty
   *flex* slot at **zero**, so filling flex appeared worth a player's entire
   projection. A 158-point TE outranked a 190-point RB.

**If a TE recommendation ever looks too good again, suspect flex accounting
first.** That is where this class of error lives.

## Bench picks need a different ranking than starter picks

Once every slot is filled, a bench body adds nothing to the lineup, so `score`
falls back on VONA — and VONA is largest where the pool is shallowest, which is
why the engine kept demanding a third TE and a backup QB. Shallow is the reason
*not* to spend the pick. Fixed with `roster_caps`, but the general lesson holds:
**when marginal lineup value is zero for every candidate, rank by ceiling and
role, not by VONA.**

## Rank the positional outlook by expected loss, not by cost

`cost_of_waiting × P(gone)`. Ranking by cost alone puts QB and TE on top because
their cliffs are steep — but those are exactly the positions that survive, so the
loss rarely comes due.

## Where the draft is actually won

Decomposing every team's final starting lineup in one 12-team draft by the round
each starter was taken:

| bucket | correlation with team total | spread between teams (sd) |
|---|---:|---:|
| R1–3 | +0.63 | 28.2 |
| R4–7 | +0.31 | **49.7** |
| R8–10 | +0.07 | 15.0 |

Rounds 1–3 correlate most with a good team but **differentiate least** —
everyone picks from the same steep part of the curve. Rounds 4–7 have nearly
double the spread: the board is flat enough there that ADP and true value
diverge, which is where a model beats a consensus list.

Rounds 8+ contribute ~nothing to a *projected* lineup, and that is the point:
their value is option value. A bench player is a call option — bounded downside
(you drop him), real upside — which is why variance is worth paying for late and
why `ceiling` exists.

## Known limitation: back-to-back turns are ranked greedily

At a turn boundary the engine ranks each pick independently rather than
optimising the *pair*, so greedily taking its top two is not guaranteed to be the
best two available. Unfixed.

## Two rejected changes, kept here so they are not retried

- **Extended turn horizon.** Theoretically appealing; A/B'd over 20 drafts it
  cost −6.9 (slot 1) and −3.3 (slot 12). You get both players at a turn
  regardless of order.
- **Ranking on marginal lineup value.** Per-pick correlation with VONA 0.94 and
  the top pick agrees everywhere, but A/B'd across VONA weights 0.5–3.0 at slots
  1/6/12 it lost at *every* setting (−7.2 best, −11.2 worst). Kept as a
  display-only column, because it is still the best cross-check for flex bugs.

# sleeper-drafter

Real-time draft co-pilot for Sleeper fantasy football. Polls a live draft, models
which players opponents will take before your next turn, and ranks your options by
forward-looking **VONA** (Value Over Next Available) rather than static VORP.

**Read-only by design.** Sleeper's public API has no write endpoints. The engine
ranks; you click the pick yourself in the Sleeper UI. Do NOT build session auth,
headless-browser pick submission, or anything that writes to Sleeper.

## Run

```bash
source .venv/bin/activate

python scripts/fetch_projections.py            # build data/projections.csv (real pts + ADP)
python scripts/fetch_consistency.py            # build data/consistency.csv (CV from last season)

python main.py --find-draft <username>         # discover your draft_id and slot
python main.py --mock --slot 5                 # offline simulated draft
python main.py --mock --slot 5 --step          # pause at each of your picks
python main.py --export-cheatsheet             # static fallback board (print this)
python main.py --replay <draft_id>             # score the model against a finished draft
python main.py --draft-id <id> --slot 5        # draft night (terminal UI)

python main.py --web --mock --slot 12          # browser dashboard, simulated draft
python main.py --web --draft-id <id> --slot 12 # browser dashboard, live
# -> http://localhost:8050  (--port to change; --host 0.0.0.0 to reach another device)

pytest -q                                      # 124 tests
```

## Layout

- `main.py` — argparse entrypoint; `--mock`, `--find-draft`, `--export-cheatsheet`, live loop.
- `src/config.py` — `Settings` dataclass: paths, API behaviour, every model coefficient.
- `src/sleeper_client.py` — HTTP. Explicit User-Agent, null-validation, retry/backoff.
- `src/state.py` — `DraftState`, snake/linear/reversal pick arithmetic, traded-pick remap.
- `src/valuation.py` — VORP, dropoff, tiers, consistency. The Petersen layer.
- `src/opponent_model.py` — urgency, ADP kernel, normalised survival.
- `src/optimizer.py` — VONA, denial, roster fit, final score.
- `src/cli.py` — `rich` terminal dashboard.
- `src/web.py` — Dash browser dashboard: one poller thread publishes a snapshot, callbacks read it.
- `src/charts.py` — Plotly figures.
- `src/theme.py` — validated palette and chart chrome for both themes.
- `src/rating.py` — best-lineup fill and league draft grades.
- `src/mock.py` — offline draft simulation.
- `src/replay.py` — scores survival predictions against a completed draft; appends to a
  calibration log so evidence accumulates across drafts.
- `scripts/` — the two data builders. Both write frozen CSV snapshots.
- `data/` — generated artifacts; all gitignored, all rebuildable.

## Key decisions / constraints

- **No API keys.** Every endpoint used is public and unauthenticated.
- **`data/projections.csv` is generated, not hand-written.** The original spec called for
  a mocked file; `api.sleeper.com/projections/nfl/<season>` returns real projections *and*
  ADP keyed by the same `player_id`, which is what makes draft-day use possible. Mock runs
  and live runs share the one frozen snapshot.
- **`api.sleeper.com` 403s the default `Python-urllib` User-Agent.** Always send one.
- **Sleeper's error shapes are inconsistent** — bad username returns HTTP 200 with a `null`
  body, bad league `200 []`, bad draft `404 null`. `response.ok` is never sufficient;
  `_get_json` null-validates everything.
- **`/v1/players/nfl` is 14.65 MB.** Docs say fetch it at most once a day. It is disk-cached
  with a 24h TTL and never polled.
- **Roster schema comes from the league's `roster_positions`**, not the draft's `slots_*`.
  There is no `slots_super_flex`, so superflex is invisible from the draft object alone.
- **Traded picks are fetched and remapped.** Without it, urgency is attributed to the roster
  that *started* with a slot rather than the one actually picking.
- **Methodology is grounded in Isaac T. Petersen, _Fantasy Football Analytics_** — see
  README for the chapter-by-chapter mapping. VORP measures against a typical *bench* player;
  dropoff is measured, not thresholded; tiers use Fisher-Jenks (not k-means, which would
  reshuffle tiers between refreshes under a pick clock).
- **Fielding a legal lineup is a hard constraint, not a bonus.** VORP correctly implies you
  should stream QB rather than draft QB25 — but a roster that cannot fill its QB slot scores
  zero there. When picks remaining ≤ mandatory slots unfilled, the pool is restricted.
  Opponents obey the same rule, because real managers do.
- **Denial stays weak (λ=0.25, window-scoped).** Petersen Ch.7 warns against joining a run
  mid-stream, which aggressive denial encourages.
- **Recompute budget is 200ms.** Currently 13ms mean / 23ms max.
- **The browser never calls Sleeper.** One background thread polls and publishes an
  immutable snapshot; every Dash callback reads it. Extra tabs cost nothing, and a slow
  network cannot wedge the UI.
- **Chart colour is validated, not eyeballed.** The only frame where two series share
  space is the standings chart, and that pair clears every colour-vision gate in both
  themes. The value-cliff chart is faceted per position precisely so it does *not* need
  six categorical hues, which could not pass.

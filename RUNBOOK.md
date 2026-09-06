# Draft-night runbook

The tool is worthless the day after the draft. Work through this in order.

## The day before

```bash
cd ~/workspace/projects/sleeper-drafter
source .venv/bin/activate

python scripts/fetch_projections.py       # refresh projections + ADP
python scripts/fetch_consistency.py       # refresh weekly-variance risk labels
python main.py --find-draft <username>    # note your draft_id and slot
python main.py --export-cheatsheet        # then PRINT data/cheatsheet.csv
```

Then **dry-run against a live Sleeper mock draft.** Join one in the app, grab its
`draft_id` from the URL, and run the real live path:

```bash
python main.py --draft-id <mock_draft_id> --slot <your_slot>
```

This step is not optional and `--mock` does not replace it. Offline mock mode samples
opponents from the very distribution the opponent model assumes, so it validates the
arithmetic but can never disconfirm the model. Only real humans test whether the model
holds — and only a real draft exercises polling, slot mapping, and roster-schema parsing.

## The morning of

```bash
python scripts/fetch_projections.py       # picks up late injury / depth-chart moves
python main.py --export-cheatsheet        # reprint if anything moved
```

## On the clock

```bash
source .venv/bin/activate
python main.py --draft-id <draft_id> --slot <your_slot>
```

- Terminal at **≥120 columns** or the table wraps.
- Second monitor, or split screen beside the Sleeper app.
- The tool ranks. **You** click the pick. It has no write path.

Reading the board:

| Column | Means |
|---|---|
| **Surv%** | Chance he is still there at your next turn. High = you can wait. |
| **VONA** | Points gained by taking him now vs. the best you'd get at that position next turn. **This is the decision number.** Negative means wait. |
| **VORP** | Value over a typical bench player. Sanity check on absolute quality. |
| **Tier** | Fisher-Jenks cluster. A tier break is a real cliff. |
| **Risk** | `steady` / `volatile` from last season's weekly variance. Steady into starting slots, volatile onto the bench. |
| **Score** | VONA + denial + roster fit. The ranking. |

The **position-run panel** fires when a position is expected to be picked clean before
your turn and a tier cliff is live. Petersen's advice is to get *ahead* of a run, not to
join one mid-stream.

## When something breaks

### A note on polling

The app polls adaptively: **1s** when your pick is imminent, **2s** in the middle
distance, **4s** when you are 15+ picks away. Even the fast tier is only 6% of Sleeper's
documented 1000 calls/min ceiling, and picks are fetched with `If-None-Match`, so a draft
that has not moved returns `304` with **zero bytes**.

Do not slow this down to 15s. A pick clock is often 30-60 seconds, so a 15s interval can
leave you looking at a board half a clock out of date — and the picks you most need to see
are the ones immediately before your turn.

**Watch for this on draft night:** Sleeper serves the picks endpoint with
`cache-control: s-maxage=300`, meaning Cloudflare may cache it for up to five minutes. If
the CDN serves an aged response, the app shows the `⚠ STALE` banner (it checks the `Age`
header, warning past 45s). If that banner sticks while picks are visibly happening in the
Sleeper app, the edge cache is the cause — report it and fall back to the cheatsheet.
This is the single most important thing to verify in the mock-draft dry run.

| Symptom | What to do |
|---|---|
| `⚠ STALE` banner, briefly | A poll failed or the CDN served an aged response; it retries. Keep drafting off the last good board. |
| `⚠ STALE` banner that will not clear | Edge cache is serving old picks. Cross-check against the Sleeper app; if it is genuinely behind, use the cheatsheet. |
| Wrong roster counts | League uses a slot schema we misread. Fall back to the cheatsheet. |
| Recommendations look absurd at 1.01 | Scoring-format detection is wrong. Check `draft.metadata.scoring_type`. |
| Total failure | The printed cheatsheet. This is why you print it. |

The loop never crashes out mid-draft by design: a failed poll shows the last known board
with a stale banner rather than exiting.

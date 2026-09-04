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

| Symptom | What to do |
|---|---|
| `⚠ STALE` banner | An API poll failed; it retries automatically. Keep drafting off the last good board. |
| Wrong roster counts | League uses a slot schema we misread. Fall back to the cheatsheet. |
| Recommendations look absurd at 1.01 | Scoring-format detection is wrong. Check `draft.metadata.scoring_type`. |
| Total failure | The printed cheatsheet. This is why you print it. |

The loop never crashes out mid-draft by design: a failed poll shows the last known board
with a stale banner rather than exiting.

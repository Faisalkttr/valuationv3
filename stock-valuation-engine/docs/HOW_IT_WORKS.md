# How the valuation engine works

This document explains the logic behind the numbers and how data moves
through the project, end to end.

## 1. What the engine is actually answering

For every ticker, the core question is:

> "Is the market's current price already backed by realistic growth,
> or does it require growth that hasn't happened yet?"

It answers that by comparing three things:
1. Where the stock trades **today** relative to its **own trading history**
2. Where it would trade if next year's **analyst revenue estimate** comes true
3. How much growth is **actually required**, at today's price, to look
   "normal" again — compared with how much growth is **forecast**

The gap between (3)'s two numbers is the single most useful output: it tells
you whether the market is asking for more than analysts expect, or less.

## 2. Project structure

```
stock-valuation-engine/
├── config/
│   └── tickers.yaml        # the watchlist: sections -> layers -> tickers
├── engine/
│   ├── fetch.py             # pulls raw data from Yahoo Finance
│   ├── metrics.py           # pure computation, no network calls
│   └── run.py                # orchestrates fetch -> compute -> store, for the whole watchlist
├── data/
│   ├── latest.csv            # one row per ticker, most recent snapshot
│   └── history.db            # SQLite, every snapshot ever computed
├── .github/workflows/
│   └── refresh.yml           # runs the engine on a schedule, commits data/
├── app.py                    # Streamlit dashboard, reads only from data/
└── .streamlit/config.toml    # dashboard theme
```

Each piece has exactly one job:

| File | Job | Talks to the network? |
|---|---|---|
| `config/tickers.yaml` | defines what to track and how it's weighted | no |
| `engine/fetch.py` | gets raw numbers into a consistent shape | yes (Yahoo Finance) |
| `engine/metrics.py` | turns raw numbers into the valuation model | no — pure functions |
| `engine/run.py` | wires the above together across the whole watchlist | no (delegates to fetch.py) |
| `app.py` | displays what's already in `data/` | no |

That separation matters for one practical reason: `metrics.py` can be unit
tested with fake numbers (no API calls, no flakiness), and `app.py` never
blocks on a live fetch — it just reads a file, so the dashboard loads
instantly even if Yahoo Finance is slow or down.

## 3. The watchlist config

`config/tickers.yaml` mirrors an allocation grid: **sections** (e.g. INFRA,
14% target) contain **layers** (e.g. "Grid & Utilities", 40% of that
section) which contain **tickers**.

```yaml
sections:
  - name: INFRA
    target_pct: 0.14
    layers:
      - name: "Layer 2: Grid & Utilities"
        weight: 0.40
        tickers:
          - LIN
          - ETN
          - ...
```

Each ticker's portfolio weight is derived, not hand-entered:

```
portfolio_weight = section.target_pct × layer.weight × (ticker's share within the layer)
```

If a layer just lists tickers, the layer's weight is split evenly across
them. If it needs an uneven split (e.g. the Business & Futuristic Overlay,
where NVO/AZN/ISRG/TMO aren't equal-weighted), tickers can instead be listed
as `{ticker: NVO, weight: 0.30}` pairs. `engine/run.py`'s `flatten_watchlist()`
function does this math once per run.

## 4. The fetch layer — turning tickers into raw numbers

For each unique ticker, `engine/fetch.py` pulls:

- **Quarterly revenue** (income statement) → summed into trailing-twelve-month (TTM) revenue
- **Market cap** (current)
- **A historical P/S series** — reconstructed by combining rolling TTM
  revenue with the historical share price at each quarter-end, over the
  configured lookback window (default 5 years)
- **The +1 year forward revenue estimate**, if analysts cover the stock
- **Fundamentals overlay**: margins (gross/operating/net), leverage (net
  debt/EBITDA, interest coverage), and cash flow (free cash flow, FCF
  margin, cash conversion)

Every one of these is wrapped so a single missing field — common for
thinly-covered or foreign-listed tickers — doesn't fail the whole ticker.
Fields that can't be computed are simply left blank (`None`) and show as
"n/a" downstream, rather than the engine guessing or crashing.

**Known approximation:** the historical P/S series treats shares
outstanding as constant across the lookback window, since Yahoo Finance
doesn't expose a clean historical share-count series for most tickers.
Fine for large caps over a 5-year window; less accurate for names with
heavy buybacks or dilution.

## 5. The computation — how each metric is derived

All of this happens in `engine/metrics.py`, with no network dependency,
starting from `current_ps = market_cap / revenue_ttm`.

**Historical distribution** — the median, 75th, and 90th percentile of the
historical P/S series. This is what "expensive relative to its own
history" or "in-line with history" means in the model.

**Target multiple** — the historical median is used as the "normal"
valuation anchor.

**Required revenue / required growth** — how much revenue the company
would need, right now, for its P/S to fall back to that anchor at the
*current* price:
```
required_revenue = market_cap / historical_median_ps
required_growth  = (required_revenue / revenue_ttm) − 1
```

**Forward P/S and forward growth** — using the analyst estimate instead of
current revenue:
```
forward_growth = (forward_revenue_estimate / revenue_ttm) − 1
forward_ps     = market_cap / forward_revenue_estimate
```

**Growth gap** — the headline number. Required growth minus forecast
growth:
```
growth_gap = required_growth − forward_growth
```
Negative means analysts expect *more* growth than the price needs to be
justified (favorable). Positive means the price is demanding more growth
than is currently forecast (a stretch).

**Expectations burden score** (0–100) and classification — how much of the
required growth is *not* already covered by the forward estimate:
```
if required_growth <= 0:
    burden_score = 0   # already fairly valued at zero growth -- no debt to cover
else:
    coverage     = forward_growth / required_growth
    burden_score = 100 × (1 − min(coverage, 1))
```
`< 25` → "Forward Expectations Manageable", `< 60` → "Elevated", else
"Stretched". The `required_growth <= 0` branch matters: a stock already
trading at or below its own historical median needs no growth to look
"normal," so it's automatically read as favorable regardless of the
forecast — earlier versions of this formula could flip sign in that case
and misclassify the most favorable setup as "Stretched."

**Plain-English explanation** — a one-line, non-technical readout
(`plain_explanation`) stating what the classification actually means for
that ticker's numbers, e.g. "Needs 17% growth to look fairly valued;
analysts expect 77% -- comfortably cleared." Shown directly in the
dashboard's ticker detail tab, ahead of any currency warning (below) if
one applies.

**Currency normalization** — before any of the above, `engine/fetch.py`
checks whether a ticker's financial statements are reported in a
different currency than its market cap/share price (common for ADRs,
e.g. a company reporting revenue in its home currency while trading in
USD). If so, revenue is converted using a live FX rate before any ratio
is computed; if no FX rate can be resolved, the ticker is flagged with a
visible warning rather than silently producing a P/S that's off by the
FX ratio.

**Quality flag** — a separate, conservative read on the fundamentals
overlay: flags "unprofitable" if operating margin is negative, "high
leverage" if net debt/EBITDA exceeds 3x, and "cash burning" if free cash
flow margin is negative. This exists specifically so a favorable growth
gap doesn't get treated as a green light without checking whether the
underlying business can actually execute on it.

## 6. Orchestration — from watchlist to output rows

`engine/run.py`'s `main()`:

1. Loads and flattens the watchlist config
2. Fetches each **unique** symbol once (a ticker appearing in multiple
   layers is only pulled once)
3. Runs `compute_valuation()` for every watchlist entry, attaching that
   entry's `section`, `layer`, and `portfolio_weight` to the result
4. Writes every result row to two places (see below)
5. Logs a summary: how many succeeded, how many failed, and which symbols
   failed — a single bad ticker never stops the run

## 7. How the output is delivered

Two files, both under `data/`, both written on every run:

**`data/latest.csv`** — one row per watchlist entry, overwritten each run.
This is a snapshot: "as of the last refresh, here's every metric for every
ticker." It's what the dashboard's Watchlist tab reads directly, and it's
plain enough to open in Excel or diff in a git commit.

**`data/history.db`** (SQLite, table `valuation_snapshots`) — every run
ever, appended, never overwritten. Same columns as the CSV plus nothing is
lost between runs. This is what powers the "P/S over time" trend line in
the dashboard's detail tab, and it's queryable directly:
```sql
SELECT as_of, current_ps FROM valuation_snapshots WHERE ticker = 'CGPOWER.NS' ORDER BY as_of;
```

**Delivery mechanism:** `.github/workflows/refresh.yml` runs `engine/run.py`
on a schedule (default: weekdays after market close), then commits the
updated `data/` folder straight back into the repo. There's no external
database and no server to maintain — GitHub Actions is the scheduler,
GitHub itself is the datastore, and Streamlit Community Cloud redeploys
automatically on every push, including those automated data-refresh
commits. The dashboard never calls Yahoo Finance directly — it only ever
reads files that are already sitting in the repo.

## 8. End-to-end flow, summarized

```
tickers.yaml
     │  (flatten: section × layer × ticker → portfolio_weight)
     ▼
fetch.py  ──(Yahoo Finance)──►  raw revenue, price history, estimates, fundamentals
     │
     ▼
metrics.py  (pure computation)  ──►  P/S percentiles, growth gap, burden score, quality flag
     │
     ▼
run.py  ──►  data/latest.csv  (snapshot, overwritten)
        └─►  data/history.db  (time series, appended)
     │
     ▼  (GitHub Actions commits data/ on a schedule)
     ▼
app.py  ──►  reads data/ only, no live calls  ──►  Streamlit dashboard
```

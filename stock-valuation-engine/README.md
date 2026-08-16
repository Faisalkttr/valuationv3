# Stock valuation engine

Pulls fundamentals from Yahoo Finance, computes a valuation-vs-history
model (current P/S against its own historical percentile distribution,
plus a forward growth "gap" analysis) for a 50+ ticker watchlist, and
serves the results through a Streamlit dashboard. Fully automated via
GitHub Actions -- no server to run yourself.

## How it works

1. **`config/tickers.yaml`** -- your watchlist, structured as `sections -> layers -> tickers`
   matching a target-allocation grid (e.g. INFRA 14% -> Grid & Utilities 40% -> LIN, ETN, ...).
   Each ticker's `portfolio_weight` is derived as `section.target_pct x layer.weight x share-within-layer`.
   Edit this to add/remove tickers, sections, or layers -- no code changes needed.
2. **`engine/fetch.py`** -- pulls raw price, revenue, and analyst estimate data via `yfinance`.
3. **`engine/metrics.py`** -- pure computation: historical P/S percentiles, forward P/S,
   required growth to "normalise" the valuation, and the growth gap between
   what's required and what analysts expect.
4. **`engine/run.py`** -- orchestrates 1-3 across the whole watchlist, writes:
   - `data/latest.csv` -- one row per ticker, most recent snapshot
   - `data/history.db` -- SQLite table (`valuation_snapshots`) with every run ever, so you get trend charts over time
5. **`.github/workflows/refresh.yml`** -- runs `engine/run.py` on a schedule
   (default: weekdays 18:00 UTC) and commits the refreshed data back to the repo.
6. **`app.py`** -- Streamlit dashboard reading purely from `data/`, so it loads
   instantly with no live API calls at view time.

## Local setup

```bash
pip install -r requirements.txt
python -m engine.run          # populates data/latest.csv and data/history.db
streamlit run app.py
```

## Deploying

1. Push this repo to GitHub.
2. In the repo's Settings > Actions > General, make sure "Read and write permissions"
   is enabled for the `GITHUB_TOKEN` (needed for the workflow to commit data back).
3. Trigger the workflow once manually (Actions tab > Refresh valuation data > Run workflow)
   to populate `data/` for the first time.
4. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo, point it at
   `app.py`. It redeploys automatically on every push -- including the daily
   automated data-refresh commits.

## Ticker symbols to verify

`config/tickers.yaml` maps every symbol from the source allocation grid to a Yahoo Finance
ticker. Most are unambiguous (US-listed large caps), but a handful were best-guesses for
non-US listings or shorthand names in the original grid -- each is flagged inline with
`# verify`. Run `python -m engine.run` once and check the failure list in the log; any
wrong symbol will simply fail to fetch (and get skipped) rather than breaking the run.

## Fundamentals overlay

Alongside the P/S valuation model, each ticker now also carries a quality/risk
snapshot pulled from the income statement, balance sheet, and cash flow statement:

- **Margins** -- gross, operating, net (TTM)
- **Leverage** -- net debt / EBITDA, interest coverage
- **Cash flow** -- free cash flow (TTM), FCF margin, cash conversion (FCF / net income)
- **`quality_flag`** -- a plain-language read ("Solid" / "Watch: ..." / "Caution: ...")
  flagging unprofitability, leverage above 3x net debt/EBITDA, or negative free cash flow

This doesn't replace the valuation model -- it's there so "cheap on P/S" can be checked
against "actually profitable and not over-levered" before treating a growth-gap number
as good news. Every field is optional: tickers where a statement isn't available
(common for some foreign listings) just show "n/a" rather than failing the whole fetch.

## Durability, management, and risk overlay

Three more layers on top of the valuation model, all descriptive -- none of them are
combined into a composite score, since weighting them into a single number is a judgment
call this engine deliberately leaves to you:

- **Growth durability** -- 3Y/5Y revenue CAGR from annual financials. Compare this to the
  forward growth estimate: if the forecast is far above the historical trend, that's a
  bigger ask than the growth-gap number alone suggests. Coverage is capped by however many
  annual periods `yfinance` exposes (typically ~4), so a 5-year CAGR is often unavailable.
- **Management** -- ROIC (NOPAT / invested capital), 3-year share count CAGR (negative =
  net buybacks, positive = dilution), and TTM cash actually deployed to buybacks/dividends/
  M&A. The split is shown, not graded -- this engine doesn't judge whether an acquisition
  was a good one.
- **Risk & market context** -- 6-month price return vs. a region-appropriate benchmark
  index (S&P 500 default, regional index for `.NS`/`.T`/`.PA`/etc. -- see
  `_BENCHMARK_BY_SUFFIX` in `engine/fetch.py`), and 30-day analyst EPS revision counts
  where available. Estimate-revision coverage is genuinely patchy outside large,
  well-covered names -- expect more "n/a" here than anywhere else in the engine.

**Deliberately left out**: customer concentration, recurring-revenue %, organic-vs-M&A
growth split, moat/competitive-position, and insider buying/selling. These live in prose
(10-Ks, earnings calls) or behind data most free sources don't reliably expose --
computing a number here would be guessing dressed up as analysis, not a real metric.

## Dashboard

`app.py` is a dark, terminal-style dashboard. Each section from the allocation grid
(INFRA, Energy & Commodity, AI/Semis, EM, Business & Futuristic Overlay) gets its own
accent color, carried through the watchlist chips, table, and per-ticker detail cards --
so which sub-portfolio a name belongs to is visible at a glance without reading labels.

- **Watchlist tab** -- summary strip (ticker count, average growth gap, expectations
  breakdown), section filter, sortable table with a progress bar on target weight
- **Ticker detail tab** -- valuation cards, a plain-English one-line conclusion (see
  below), the valuation bar chart (median / 75th / current / 90th / forward P/S), a
  P/S-over-time trend line once the engine has run more than once, and four fundamentals
  sections: Fundamentals (margins/leverage/FCF + quality badge), Growth durability
  (revenue CAGR), Management (ROIC, share dilution, capital allocation split), and
  Risk & market context (relative strength, EPS revisions)
- **Ad-hoc search tab** -- look up any ticker not in `config/tickers.yaml`. Runs a live
  fetch against Yahoo Finance on demand (cached 10 minutes per symbol in-session), reuses
  the exact same computation and rendering as the tracked watchlist, but nothing here
  gets written to `data/` -- it's a one-off lookup, not added to the tracked history.

### Plain-English conclusions

Every ticker's detail view now includes a one-line, non-technical readout, e.g.:

> Needs 17% growth to look fairly valued by its own history; analysts expect 77% --
> the bar is comfortably cleared if the forecast is anywhere close to right.

This exists so the cards and chart above it don't need decoding -- it states directly
what "Manageable" / "Elevated" / "Stretched" actually means for that specific ticker's
numbers, and calls out the currency-mismatch warning (below) first if applicable.

### Fixes worth knowing about

- **Burden-score edge case**: when a stock's current P/S is already at or below its own
  historical median (no growth is needed to look "fairly valued"), the classification
  now correctly reads "Manageable" regardless of the forward growth estimate. Previously
  the coverage-ratio math inverted sign in this case and could misclassify a genuinely
  favorable setup as "Stretched" -- see `engine/metrics.py`'s `_burden_score()`.
- **Currency normalization**: some tickers (typically ADRs and foreign filers, e.g. TSM)
  report revenue in their home currency while trading in another. `engine/fetch.py` now
  detects this via `financialCurrency` vs. the trading currency, converts revenue using a
  live FX rate before computing P/S, and flags the ticker with a visible warning if no FX
  rate can be resolved -- rather than silently producing a P/S that's off by the FX ratio.

## Notes and known approximations

- **Historical P/S** is built from quarterly revenue x historical close price x current
  shares outstanding. Share count drift over the lookback window is ignored --
  fine for a 5-year window on large caps, less accurate for names with heavy
  buybacks/dilution.
- **Forward revenue estimate** comes from `yfinance`'s analyst estimate table, which
  isn't available for every ticker (thinly covered small caps especially).
  When missing, forward-looking fields (`forward_ps`, `growth_gap`, etc.) are left null
  and the dashboard shows "n/a" rather than guessing.
- **Rate limiting**: `engine/fetch.py` pauses briefly between tickers and retries
  transient failures. For 50+ tickers a full run can take a few minutes --
  that's expected and fine for a scheduled job.
- Yahoo Finance's public endpoints are unofficial and can change shape without
  notice; if a run starts failing broadly, check the `yfinance` changelog first.

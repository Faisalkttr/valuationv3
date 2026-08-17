"""Entry point run by GitHub Actions (and locally: `python -m engine.run`)."""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from engine.fetch import fetch_watchlist
from engine.metrics import compute_valuation, compute_buy_priority_scores

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("engine.run")

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "tickers.yaml"
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "history.db"
LATEST_CSV_PATH = DATA_DIR / "latest.csv"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def flatten_watchlist(config: dict) -> list[dict]:
    rows = []
    for section in config["sections"]:
        section_name = section["name"]
        section_target = section["target_pct"]
        for layer in section["layers"]:
            layer_name = layer["name"]
            layer_weight = layer["weight"]
            tickers = layer["tickers"]
            if tickers and isinstance(tickers[0], dict):
                entries = [(t["ticker"], t["weight"]) for t in tickers]
            else:
                even_share = 1 / len(tickers) if tickers else 0
                entries = [(t, even_share) for t in tickers]
            for symbol, in_layer_weight in entries:
                rows.append({
                    "symbol": symbol,
                    "section": section_name,
                    "section_target_pct": section_target,
                    "layer": layer_name,
                    "layer_weight": layer_weight,
                    "portfolio_weight": section_target * layer_weight * in_layer_weight,
                    "thesis": layer.get("thesis", "").strip(),
                })
    return rows


def write_to_history_db(rows: list[dict]) -> None:
    """Appends `rows` to the valuation_snapshots table.

    The schema evolves as new metrics are added (e.g. ROIC persistence,
    buy-priority score) -- a plain `to_sql(..., if_exists="append")` would
    fail with "table has no column named X" the first time a new column
    shows up in `rows` but not in an existing history.db. To stay backward
    compatible with any history.db already on disk, we read whatever's
    there, union the columns with the new rows, and rewrite the table --
    the appended new rows get real values, old rows get NaN for the new
    columns rather than the run crashing outright.
    """
    DATA_DIR.mkdir(exist_ok=True)
    new_df = pd.DataFrame(rows)

    existing_df = pd.DataFrame()
    if DB_PATH.exists():
        try:
            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='valuation_snapshots'"
                )
                if cursor.fetchone() is not None:
                    existing_df = pd.read_sql("SELECT * FROM valuation_snapshots", conn)
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not read existing history.db (%s) -- starting a fresh table.", exc)
            existing_df = pd.DataFrame()

    combined = pd.concat([existing_df, new_df], ignore_index=True, sort=False) if not existing_df.empty else new_df

    with sqlite3.connect(DB_PATH) as conn:
        combined.to_sql("valuation_snapshots", conn, if_exists="replace", index=False)


def write_latest_csv(rows: list[dict]) -> None:
    DATA_DIR.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(LATEST_CSV_PATH, index=False)


def main() -> None:
    config = load_config()
    watchlist_rows = flatten_watchlist(config)
    history_years = config.get("history_years", 5)

    unique_symbols = list(dict.fromkeys(row["symbol"] for row in watchlist_rows))

    log.info("Running valuation engine for %d tickers (%d watchlist entries)",
             len(unique_symbols), len(watchlist_rows))

    fetched: dict = {}
    failures: list[str] = []

    for symbol, outcome in fetch_watchlist(unique_symbols, history_years=history_years):
        if isinstance(outcome, Exception):
            failures.append(symbol)
        else:
            fetched[symbol] = outcome

    results: list[dict] = []
    as_of = datetime.now(timezone.utc)

    for row in watchlist_rows:
        outcome = fetched.get(row["symbol"])
        if outcome is None:
            continue
        try:
            result = compute_valuation(
                ticker=row["symbol"],
                current_revenue_ttm=outcome.current_revenue_ttm,
                current_market_cap=outcome.current_market_cap,
                historical_ps_series=outcome.historical_ps_series,
                forward_revenue_estimate=outcome.forward_revenue_estimate,
                revenue_cadence=outcome.revenue_cadence,
                as_of=as_of,
                gross_margin=outcome.gross_margin,
                operating_margin=outcome.operating_margin,
                net_margin=outcome.net_margin,
                net_debt_to_ebitda=outcome.net_debt_to_ebitda,
                interest_coverage=outcome.interest_coverage,
                free_cash_flow_ttm=outcome.free_cash_flow_ttm,
                fcf_margin=outcome.fcf_margin,
                cash_conversion=outcome.cash_conversion,
                revenue_currency=outcome.revenue_currency,
                price_currency=outcome.price_currency,
                fx_rate_applied=outcome.fx_rate_applied,
                currency_note=outcome.currency_note,
                revenue_cagr_3y=outcome.revenue_cagr_3y,
                revenue_cagr_5y=outcome.revenue_cagr_5y,
                roic=outcome.roic,
                share_count_cagr_3y=outcome.share_count_cagr_3y,
                buybacks_ttm=outcome.buybacks_ttm,
                dividends_ttm=outcome.dividends_ttm,
                acquisitions_ttm=outcome.acquisitions_ttm,
                price_return_6m=outcome.price_return_6m,
                benchmark_symbol=outcome.benchmark_symbol,
                benchmark_return_6m=outcome.benchmark_return_6m,
                relative_strength_6m=outcome.relative_strength_6m,
                eps_revisions_up_30d=outcome.eps_revisions_up_30d,
                eps_revisions_down_30d=outcome.eps_revisions_down_30d,
                owner_earnings_ttm=outcome.owner_earnings_ttm,
                owner_earnings_yield=outcome.owner_earnings_yield,
                maintenance_capex_ttm=outcome.maintenance_capex_ttm,
                owner_earnings_method=outcome.owner_earnings_method,
                price_cagr_3y=outcome.price_cagr_3y,
                market_cap_cagr_3y=outcome.market_cap_cagr_3y,
                roic_avg_3y=outcome.roic_avg_3y,
                roic_trend=outcome.roic_trend,
            )

            record = result.to_dict()
            record.update({
                "section": row["section"],
                "section_target_pct": row["section_target_pct"],
                "layer": row["layer"],
                "portfolio_weight": row["portfolio_weight"],
                "thesis": row["thesis"],
            })
            results.append(record)

            log.info("%s: P/S %.2f vs median %.2f -- %s",
                     row["symbol"], result.current_ps, result.hist_median_ps,
                     result.expectations_classification)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to compute metrics for %s: %s", row["symbol"], exc)
            failures.append(row["symbol"])

    if results:
        # Buy Priority Score (Problem #5) is cross-sectional -- rank every
        # ticker against its peers in this run before persisting.
        scored_df = compute_buy_priority_scores(pd.DataFrame(results))
        results = scored_df.to_dict(orient="records")
        write_to_history_db(results)
        write_latest_csv(results)

    log.info("Done. %d succeeded, %d failed.", len(results), len(failures))
    if failures:
        log.info("Failed tickers: %s", ", ".join(sorted(set(failures))))


if __name__ == "__main__":
    main()

"""
Data fetch layer -- the only module that talks to the network.

This version is deliberately more inclusive:
- supports quarterly, semiannual, and annual revenue cadence
- falls back across multiple yfinance statement attributes
- annualizes partial periods where reasonable
- tolerates thin-data foreign listings better
- keeps owner earnings and anti-bubble inputs
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass

import pandas as pd
import yfinance as yf
from tenacity import retry, stop_after_attempt, wait_exponential

log = logging.getLogger("engine.fetch")


@dataclass
class RawTickerData:
    ticker: str
    current_revenue_ttm: float
    current_market_cap: float
    historical_ps_series: pd.Series
    forward_revenue_estimate: float | None
    revenue_cadence: str = "quarterly"

    # Fundamentals overlay
    gross_margin: float | None = None
    operating_margin: float | None = None
    net_margin: float | None = None
    net_debt_to_ebitda: float | None = None
    interest_coverage: float | None = None
    free_cash_flow_ttm: float | None = None
    fcf_margin: float | None = None
    cash_conversion: float | None = None

    # Currency normalization metadata
    revenue_currency: str | None = None
    price_currency: str | None = None
    fx_rate_applied: float | None = None
    currency_note: str = ""

    # Growth durability
    revenue_cagr_3y: float | None = None
    revenue_cagr_5y: float | None = None

    # Management / capital allocation
    roic: float | None = None
    share_count_cagr_3y: float | None = None
    buybacks_ttm: float | None = None
    dividends_ttm: float | None = None
    acquisitions_ttm: float | None = None

    # Risk / market context
    price_return_6m: float | None = None
    benchmark_symbol: str | None = None
    benchmark_return_6m: float | None = None
    relative_strength_6m: float | None = None
    eps_revisions_up_30d: int | None = None
    eps_revisions_down_30d: int | None = None

    # Owner earnings overlay
    owner_earnings_ttm: float | None = None
    owner_earnings_yield: float | None = None
    maintenance_capex_ttm: float | None = None
    owner_earnings_method: str | None = None

    # Anti-bubble / multiple expansion overlay
    price_cagr_3y: float | None = None
    market_cap_cagr_3y: float | None = None


# ---------------------------------------------------------------------------
# Generic safe helpers
# ---------------------------------------------------------------------------

def _safe_get(obj, key, default=None):
    """
    Safely read a value from yfinance metadata objects.
    """
    if obj is None:
        return default

    try:
        getter = getattr(obj, "get", None)
        if callable(getter):
            return getter(key, default)
    except Exception:
        pass

    try:
        return getattr(obj, key, default)
    except Exception:
        return default


def _as_float(value):
    """Convert a value to float if possible. Returns None for NaN/invalid values."""
    if value is None:
        return None

    try:
        out = float(value)
    except (TypeError, ValueError):
        return None

    try:
        if pd.isna(out):
            return None
    except Exception:
        pass

    return out


def _first_float(values):
    """Return the first successfully parsed float from an iterable of candidates."""
    for value in values:
        parsed = _as_float(value)
        if parsed is not None:
            return parsed
    return None


def _get_metadata(tkr: yf.Ticker):
    """
    Returns (info, fast_info), with defensive fallbacks.
    """
    try:
        info = tkr.info
    except Exception:
        info = None

    if info is None:
        info = {}

    try:
        fast_info = tkr.fast_info
    except Exception:
        fast_info = None

    return info, fast_info


def _market_data(tkr: yf.Ticker):
    """
    Returns (market_cap, last_price, shares_outstanding).
    """
    info, fast_info = _get_metadata(tkr)

    price = _first_float(
        (
            _safe_get(fast_info, "last_price"),
            _safe_get(info, "currentPrice"),
            _safe_get(info, "regularMarketPrice"),
            _safe_get(info, "previousClose"),
        )
    )

    shares = _first_float(
        (
            _safe_get(fast_info, "shares_outstanding"),
            _safe_get(info, "sharesOutstanding"),
            _safe_get(info, "impliedSharesOutstanding"),
        )
    )

    market_cap = _first_float(
        (
            _safe_get(fast_info, "market_cap"),
            _safe_get(info, "marketCap"),
        )
    )

    if market_cap is None and price is not None and shares is not None:
        market_cap = price * shares

    if shares is None and market_cap is not None and price is not None and price > 0:
        shares = market_cap / price

    return market_cap, price, shares


def _last_price_from_ticker(tkr: yf.Ticker) -> float | None:
    """
    Best-effort last price for any yfinance ticker, including FX pairs.
    """
    _, price, _ = _market_data(tkr)
    if price is not None:
        return price

    try:
        hist = tkr.history(period="1d", interval="1d")
        if hist is not None and not hist.empty and "Close" in hist.columns:
            return _as_float(hist["Close"].iloc[-1])
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Core fetch
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, min=2, max=20))
def _get_ticker(symbol: str) -> yf.Ticker:
    return yf.Ticker(symbol)


_REVENUE_ROW_CANDIDATES = (
    "Total Revenue",
    "Total Revenues",
    "Revenue",
    "Net Revenue",
    "Operating Revenue",
    "Revenue From Continuing Operations",
    "Total Operating Revenue",
)


def _get_statement(tkr: yf.Ticker, attr_names: tuple[str, ...]) -> pd.DataFrame | None:
    """
    Try multiple yfinance statement attributes and return the first non-empty one.
    """
    for attr in attr_names:
        try:
            statement = getattr(tkr, attr, None)
            if statement is not None and not getattr(statement, "empty", True):
                return statement
        except Exception:
            pass

    return None


def _statement_series(statement: pd.DataFrame | None, *row_names: str) -> pd.Series | None:
    """
    Full historical series for the first matching row name found.
    """
    if statement is None or getattr(statement, "empty", True):
        return None

    row_lookup = {str(idx).strip(): idx for idx in statement.index}

    for name in row_names:
        actual_row = row_lookup.get(str(name).strip())
        if actual_row is None:
            continue

        raw = statement.loc[actual_row]

        if isinstance(raw, pd.DataFrame):
            raw = raw.iloc[0]

        series = pd.to_numeric(raw, errors="coerce").dropna().sort_index()

        if not series.empty:
            return series

    return None


def _extract_revenue_series(statement: pd.DataFrame | None) -> pd.Series:
    """
    Extract revenue series from an income statement.
    """
    series = _statement_series(statement, *_REVENUE_ROW_CANDIDATES)
    return series if series is not None else pd.Series(dtype=float)


def _infer_cadence(series: pd.Series, source_hint: str | None = None) -> str:
    """
    Infer cadence from the spacing of statement dates.

    quarterly  -> roughly 90 days
    semiannual -> roughly 180 days
    annual     -> roughly 365 days
    """
    try:
        idx = pd.to_datetime(series.index)

        if getattr(idx, "tz", None) is not None:
            try:
                idx = idx.tz_localize(None)
            except Exception:
                pass

        if len(idx) < 2:
            return source_hint or "annual"

        gaps = pd.Series(idx).diff().dropna().dt.days

        if gaps.empty:
            return source_hint or "annual"

        median_gap = float(gaps.median())

        if median_gap <= 120:
            return "quarterly"

        if median_gap <= 220:
            return "semiannual"

        return "annual"

    except Exception:
        return source_hint or "annual"


def _periods_for_cadence(cadence: str) -> int:
    return {
        "quarterly": 4,
        "semiannual": 2,
        "annual": 1,
    }.get(cadence, 4)


def _series_period_sum(series: pd.Series | None, cadence: str) -> float | None:
    """
    Sum the appropriate trailing periods.

    If fewer periods are available than the cadence implies, annualize the
    average of what is available. This is approximate, but more inclusive
    for thin-data tickers.
    """
    if series is None or series.empty:
        return None

    periods = _periods_for_cadence(cadence)

    if periods <= 1:
        return _as_float(series.iloc[-1])

    window = series.iloc[-periods:]

    if window.empty:
        return None

    if len(window) >= periods:
        return float(window.sum())

    return float(window.mean() * periods)


def _revenue_ttm_from_series(series: pd.Series, cadence: str) -> float | None:
    """
    Convert a revenue series into a TTM-like revenue figure based on cadence.
    """
    return _series_period_sum(series, cadence)


def _revenue_series_with_cadence(tkr: yf.Ticker) -> tuple[pd.Series, str]:
    """
    Try to get revenue data in this order:

    1. quarterly / half-year statement
    2. annual statement

    Returns:
        revenue_series, cadence
    """
    quarterly_stmt = _get_statement(
        tkr,
        (
            "quarterly_income_stmt",
            "quarterly_financials",
        ),
    )

    rev = _extract_revenue_series(quarterly_stmt)

    if not rev.empty:
        cadence = _infer_cadence(rev, source_hint="quarterly")
        return rev, cadence

    annual_stmt = _get_statement(
        tkr,
        (
            "income_stmt",
            "financials",
        ),
    )

    rev = _extract_revenue_series(annual_stmt)

    if not rev.empty:
        cadence = _infer_cadence(rev, source_hint="annual")
        return rev, cadence

    raise ValueError(
        "No quarterly, semiannual, or annual revenue statement available. "
        "This ticker may be an ETF/fund, a thinly listed foreign security, "
        "or unsupported by Yahoo Finance. Try the primary exchange-listed "
        "symbol or a US ADR where available."
    )


def _row_period_sum(statement: pd.DataFrame | None, cadence: str, *row_names: str) -> float | None:
    """
    Cadence-aware sum of income statement / cash flow statement items.
    """
    series = _statement_series(statement, *row_names)
    return _series_period_sum(series, cadence)


def _row_latest(statement: pd.DataFrame | None, *row_names: str) -> float | None:
    """
    Most recent value of the first matching row.
    Used for balance-sheet items.
    """
    series = _statement_series(statement, *row_names)

    if series is None or series.empty:
        return None

    return _as_float(series.iloc[-1])


def _row_series(statement: pd.DataFrame | None, *row_names: str) -> pd.Series | None:
    """
    Full historical series for the first matching row name.
    """
    return _statement_series(statement, *row_names)


def _cagr(series: pd.Series | None, years: int) -> float | None:
    """
    Compound annual growth rate over `years` periods.
    """
    if series is None or len(series) <= years:
        return None

    start = _as_float(series.iloc[-(years + 1)])
    end = _as_float(series.iloc[-1])

    if start is None or end is None or start <= 0 or end <= 0:
        return None

    return (end / start) ** (1 / years) - 1


def _historical_ps_series(
    tkr: yf.Ticker,
    rev_series: pd.Series,
    years: int,
    shares: float | None = None,
    cadence: str = "quarterly",
) -> tuple[pd.Series, pd.Series]:
    """
    Builds a historical trailing-P/S series.

    Supports:
        quarterly   -> rolling 4 periods
        semiannual  -> rolling 2 periods
        annual      -> annual revenue directly

    For partial early windows, values are annualized using the cadence.
    """
    if shares is None:
        _, _, shares = _market_data(tkr)

    shares = _as_float(shares)

    if not shares:
        raise ValueError("No shares outstanding data available")

    periods = _periods_for_cadence(cadence)

    if periods > 1:
        rolling_ttm = rev_series.rolling(periods, min_periods=1).mean() * periods
    else:
        rolling_ttm = rev_series.copy()

    rolling_ttm = rolling_ttm.dropna()

    if rolling_ttm.empty:
        raise ValueError("Not enough revenue history to build a P/S series")

    try:
        rolling_ttm.index = pd.to_datetime(rolling_ttm.index)
    except Exception:
        pass

    if getattr(rolling_ttm.index, "tz", None) is not None:
        try:
            rolling_ttm = rolling_ttm.copy()
            rolling_ttm.index = rolling_ttm.index.tz_localize(None)
        except Exception:
            pass

    cutoff = rolling_ttm.index.max() - pd.DateOffset(years=years)
    rolling_ttm = rolling_ttm[rolling_ttm.index >= cutoff]

    if rolling_ttm.empty:
        raise ValueError("Not enough revenue history inside the requested lookback window")

    start = rolling_ttm.index.min() - pd.Timedelta(days=30)
    end = pd.Timestamp.today()

    try:
        history = tkr.history(start=start, end=end, interval="1d")
        prices = history["Close"] if history is not None and not history.empty else pd.Series(dtype=float)
    except Exception as exc:
        raise ValueError("No price history available") from exc

    if prices.empty:
        raise ValueError("No price history available")

    prices = prices.copy()

    if getattr(prices.index, "tz", None) is not None:
        try:
            prices.index = prices.index.tz_localize(None)
        except Exception:
            pass

    tolerance_days = {
        "quarterly": 5,
        "semiannual": 10,
        "annual": 20,
    }.get(cadence, 10)

    ps_points = []

    for period_end, ttm_rev in rolling_ttm.items():
        window = prices[
            (prices.index >= period_end - pd.Timedelta(days=tolerance_days))
            & (prices.index <= period_end + pd.Timedelta(days=tolerance_days))
        ]

        ttm_rev = _as_float(ttm_rev)

        if window.empty or ttm_rev is None or ttm_rev <= 0:
            continue

        price = _as_float(window.iloc[-1])

        if price is None:
            continue

        market_cap_at_time = price * shares
        ps_points.append((period_end, market_cap_at_time / ttm_rev))

    if not ps_points:
        raise ValueError("No historical P/S observations could be constructed")

    ps_index, ps_values = zip(*ps_points)

    return pd.Series(list(ps_values), index=list(ps_index)), prices


def _fallback_price_series(tkr: yf.Ticker, years: int = 5) -> pd.Series:
    """
    Fallback price series when historical P/S construction fails.
    """
    for period in (f"{years}y", "3y", "1y", "6mo"):
        try:
            hist = tkr.history(period=period, interval="1d")
            if hist is not None and not hist.empty and "Close" in hist.columns:
                prices = hist["Close"].dropna()

                if not prices.empty:
                    if getattr(prices.index, "tz", None) is not None:
                        try:
                            prices.index = prices.index.tz_localize(None)
                        except Exception:
                            pass
                    return prices
        except Exception:
            pass

    return pd.Series(dtype=float)


def _forward_revenue_estimate(tkr: yf.Ticker) -> float | None:
    """
    Pulls the analyst +1y forward revenue estimate if available.
    """
    try:
        est = tkr.get_revenue_estimate()
    except Exception:
        return None

    if est is None or est.empty:
        return None

    row_lookup = {str(idx).strip().lower(): idx for idx in est.index}

    for row_label in ("+1y", "1y", "0y"):
        actual_row = row_lookup.get(row_label)
        if actual_row is None:
            continue

        row = est.loc[actual_row]

        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]

        col_lookup = {str(c).strip().lower(): c for c in row.index}

        for col_label in ("avg", "avg. estimate", "average"):
            actual_col = col_lookup.get(col_label)
            if actual_col is None:
                continue

            value = _as_float(row[actual_col])
            if value is not None:
                return value

    return None


def _price_cagr(price_series: pd.Series | None, years: int = 3) -> float | None:
    """
    CAGR of the price series over approximately `years` years.
    """
    if price_series is None or price_series.empty:
        return None

    try:
        s = price_series.copy()

        if getattr(s.index, "tz", None) is not None:
            try:
                s.index = s.index.tz_localize(None)
            except Exception:
                pass

        if len(s) < 2:
            return None

        end_date = s.index.max()
        target_date = end_date - pd.DateOffset(years=years)

        if s.index.min() > target_date:
            return None

        start_candidates = s[s.index <= target_date]

        if start_candidates.empty:
            return None

        start_date = start_candidates.index[-1]
        start_price = _as_float(start_candidates.iloc[-1])
        end_price = _as_float(s.iloc[-1])

        if start_price is None or end_price is None:
            return None

        if start_price <= 0 or end_price <= 0:
            return None

        actual_years = (end_date - start_date).days / 365.25

        if actual_years <= 0:
            return None

        return (end_price / start_price) ** (1 / actual_years) - 1

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Statement selection by cadence
# ---------------------------------------------------------------------------

def _income_statement_for_cadence(tkr: yf.Ticker, cadence: str) -> pd.DataFrame | None:
    if cadence == "annual":
        return _get_statement(tkr, ("income_stmt", "financials"))

    return (
        _get_statement(tkr, ("quarterly_income_stmt", "quarterly_financials"))
        or _get_statement(tkr, ("income_stmt", "financials"))
    )


def _cashflow_statement_for_cadence(tkr: yf.Ticker, cadence: str) -> pd.DataFrame | None:
    if cadence == "annual":
        return _get_statement(tkr, ("cashflow",))

    return (
        _get_statement(tkr, ("quarterly_cashflow",))
        or _get_statement(tkr, ("cashflow",))
    )


def _balance_sheet_for_cadence(tkr: yf.Ticker, cadence: str) -> pd.DataFrame | None:
    if cadence == "annual":
        return _get_statement(tkr, ("balance_sheet",))

    return (
        _get_statement(tkr, ("quarterly_balance_sheet",))
        or _get_statement(tkr, ("balance_sheet",))
    )


# ---------------------------------------------------------------------------
# Fundamentals overlay
# ---------------------------------------------------------------------------

def _growth_durability(tkr: yf.Ticker) -> dict:
    """
    Revenue CAGR from annual financials.
    """
    out: dict = {}

    try:
        annual_stmt = _get_statement(tkr, ("income_stmt", "financials"))
        annual_rev = _statement_series(annual_stmt, *_REVENUE_ROW_CANDIDATES)

        for years in (3, 5):
            cagr = _cagr(annual_rev, years)
            if cagr is not None:
                out[f"revenue_cagr_{years}y"] = cagr

    except Exception:
        pass

    return out


def _roic(tkr: yf.Ticker) -> float | None:
    """
    Return on invested capital, from the most recent annual filing.
    """
    try:
        income = _get_statement(tkr, ("income_stmt", "financials"))
        balance = _get_statement(tkr, ("balance_sheet",))

        operating_income = _as_float(_row_latest(income, "Operating Income", "EBIT"))

        if operating_income is None:
            return None

        pretax = _as_float(_row_latest(income, "Pretax Income"))
        tax = _as_float(_row_latest(income, "Tax Provision"))

        if pretax is not None and pretax > 0 and tax is not None:
            tax_rate = tax / pretax
        else:
            tax_rate = 0.21

        tax_rate = min(max(tax_rate, 0.0), 0.5)

        nopat = operating_income * (1 - tax_rate)

        total_debt = _as_float(_row_latest(balance, "Total Debt")) or 0.0

        equity = _as_float(
            _row_latest(
                balance,
                "Stockholders Equity",
                "Common Stock Equity",
            )
        ) or 0.0

        cash = _as_float(
            _row_latest(
                balance,
                "Cash And Cash Equivalents",
                "Cash Cash Equivalents And Short Term Investments",
            )
        ) or 0.0

        invested_capital = total_debt + equity - cash

        if invested_capital > 0:
            return nopat / invested_capital

    except Exception:
        pass

    return None


def _share_dilution(tkr: yf.Ticker) -> float | None:
    """
    3-year CAGR of shares outstanding, from annual balance sheet history.
    """
    try:
        balance = _get_statement(tkr, ("balance_sheet",))

        shares = _row_series(
            balance,
            "Ordinary Shares Number",
            "Share Issued",
            "Shares Issued",
            "Common Shares Outstanding",
            "Number of Shares Outstanding",
        )

        return _cagr(shares, 3)

    except Exception:
        return None


def _capital_allocation(tkr: yf.Ticker, cadence: str) -> dict:
    """
    Cash deployed to buybacks, dividends, and acquisitions.
    """
    out: dict = {}

    try:
        cashflow = _cashflow_statement_for_cadence(tkr, cadence)

        buybacks = _row_period_sum(cashflow, cadence, "Repurchase Of Capital Stock")
        dividends = _row_period_sum(cashflow, cadence, "Cash Dividends Paid", "Common Stock Dividend Paid")
        acquisitions = _row_period_sum(
            cashflow,
            cadence,
            "Net Business Purchase And Sale",
            "Purchase Of Business",
        )

        if buybacks is not None:
            out["buybacks_ttm"] = abs(buybacks)

        if dividends is not None:
            out["dividends_ttm"] = abs(dividends)

        if acquisitions is not None:
            out["acquisitions_ttm"] = abs(acquisitions)

    except Exception:
        pass

    return out


def _owner_earnings(
    tkr: yf.Ticker,
    market_cap: float | None,
    fx_rate: float | None,
    revenue_ccy: str | None,
    price_ccy: str | None,
    cadence: str,
) -> dict:
    """
    Buffett-style owner earnings approximation, cadence-aware.
    """
    out: dict = {}

    try:
        income = _income_statement_for_cadence(tkr, cadence)
        cashflow = _cashflow_statement_for_cadence(tkr, cadence)

        net_income = _row_period_sum(
            income,
            cadence,
            "Net Income",
            "Net Income Common Stockholders",
        )

        op_cf = _row_period_sum(
            cashflow,
            cadence,
            "Operating Cash Flow",
            "Cash Flow From Continuing Operating Activities",
        )

        da = _row_period_sum(
            cashflow,
            cadence,
            "Depreciation And Amortization",
            "Depreciation & Amortization",
            "Depreciation Amortization Depletion",
            "Depreciation, Amortization & Depletion",
            "D&A",
        )

        if da is None:
            da = _row_period_sum(
                income,
                cadence,
                "Depreciation And Amortization",
                "Depreciation & Amortization",
                "Depreciation Amortization Depletion",
                "Depreciation, Amortization & Depletion",
                "D&A",
            )

        capex = _row_period_sum(cashflow, cadence, "Capital Expenditure")
        capex_abs = abs(capex) if capex is not None else None

        maintenance_capex = None

        if capex_abs is not None:
            if da is not None and da > 0:
                maintenance_capex = min(capex_abs, da)
            else:
                maintenance_capex = capex_abs

        owner_native = None
        method = None

        if net_income is not None and da is not None and maintenance_capex is not None:
            owner_native = net_income + da - maintenance_capex
            method = "Net income + D&A - maintenance capex"

        elif op_cf is not None and maintenance_capex is not None:
            owner_native = op_cf - maintenance_capex
            method = "Operating cash flow - maintenance capex"

        if owner_native is None or method is None:
            return out

        def _to_price_ccy(value: float | None) -> float | None:
            if value is None:
                return None

            if fx_rate is not None:
                return value * fx_rate

            if revenue_ccy is None and price_ccy is None:
                return value

            if revenue_ccy == price_ccy:
                return value

            return None

        owner_price_ccy = _to_price_ccy(owner_native)
        maintenance_price_ccy = _to_price_ccy(maintenance_capex)

        if maintenance_price_ccy is not None:
            out["maintenance_capex_ttm"] = maintenance_price_ccy

        if owner_price_ccy is not None:
            out["owner_earnings_ttm"] = owner_price_ccy
            out["owner_earnings_method"] = method

            if market_cap and market_cap > 0:
                out["owner_earnings_yield"] = owner_price_ccy / market_cap

        return out

    except Exception:
        return out


def _fundamentals(tkr: yf.Ticker, revenue_ttm: float, cadence: str) -> dict:
    """
    Best-effort margin, leverage, and cash-flow overlay.
    """
    out: dict = {}

    ebitda: float | None = None
    net_income: float | None = None

    try:
        income = _income_statement_for_cadence(tkr, cadence)

        gross_profit = _row_period_sum(income, cadence, "Gross Profit")
        operating_income = _row_period_sum(income, cadence, "Operating Income", "EBIT")
        net_income = _row_period_sum(income, cadence, "Net Income", "Net Income Common Stockholders")
        interest_expense = _row_period_sum(income, cadence, "Interest Expense", "Interest Expense Non Operating")
        ebitda = _row_period_sum(income, cadence, "EBITDA", "Normalized EBITDA")

        if revenue_ttm:
            if gross_profit is not None:
                out["gross_margin"] = gross_profit / revenue_ttm

            if operating_income is not None:
                out["operating_margin"] = operating_income / revenue_ttm

            if net_income is not None:
                out["net_margin"] = net_income / revenue_ttm

        if operating_income is not None and interest_expense:
            out["interest_coverage"] = operating_income / abs(interest_expense)

    except Exception:
        log.debug("%s: income statement fundamentals unavailable", getattr(tkr, "ticker", "?"))

    try:
        balance = _balance_sheet_for_cadence(tkr, cadence)

        total_debt = _row_latest(balance, "Total Debt")
        cash = _row_latest(
            balance,
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents And Short Term Investments",
        )

        if total_debt is not None and cash is not None and ebitda:
            out["net_debt_to_ebitda"] = (total_debt - cash) / ebitda

    except Exception:
        pass

    try:
        cashflow = _cashflow_statement_for_cadence(tkr, cadence)

        fcf = _row_period_sum(cashflow, cadence, "Free Cash Flow")

        if fcf is None:
            op_cf = _row_period_sum(
                cashflow,
                cadence,
                "Operating Cash Flow",
                "Cash Flow From Continuing Operating Activities",
            )
            capex = _row_period_sum(cashflow, cadence, "Capital Expenditure")

            if op_cf is not None and capex is not None:
                fcf = op_cf - abs(capex)

        if fcf is not None:
            out["free_cash_flow_ttm"] = fcf

            if revenue_ttm:
                out["fcf_margin"] = fcf / revenue_ttm

            if net_income:
                out["cash_conversion"] = fcf / net_income

    except Exception:
        pass

    return out


# ---------------------------------------------------------------------------
# Benchmark / relative strength / revisions
# ---------------------------------------------------------------------------

_BENCHMARK_BY_SUFFIX = {
    ".NS": "^NSEI",
    ".BO": "^BSESN",
    ".T": "^N225",
    ".SW": "^SSMI",
    ".PA": "^FCHI",
    ".L": "^FTSE",
    ".HK": "^HSI",
    ".SR": "^TASI.SR",
    ".AD": "^ADI",
    ".AX": "^AXJO",
    ".ST": "^OMXS30",
    ".HE": "^OMXH25",
    ".CO": "^OMXC25",
    ".MI": "^FTSEMIB",
    ".DE": "^GDAXI",
    ".AS": "^AEX",
    ".MC": "^IBEX",
}


def _benchmark_for(symbol: str) -> str:
    for suffix, bm in _BENCHMARK_BY_SUFFIX.items():
        if symbol.endswith(suffix):
            return bm

    return "^GSPC"


def _period_return(price_series: pd.Series, months: int) -> float | None:
    if price_series is None or price_series.empty:
        return None

    try:
        price_series = price_series.copy()

        if getattr(price_series.index, "tz", None) is not None:
            try:
                price_series.index = price_series.index.tz_localize(None)
            except Exception:
                pass

        cutoff = price_series.index.max() - pd.DateOffset(months=months)
        window = price_series[price_series.index >= cutoff]

        if len(window) < 2:
            return None

        start_price = _as_float(window.iloc[0])
        end_price = _as_float(window.iloc[-1])

        if start_price is None or end_price is None or start_price == 0:
            return None

        return float(end_price / start_price - 1)

    except Exception:
        return None


def _relative_strength(
    tkr: yf.Ticker,
    symbol: str,
    price_series: pd.Series | None = None,
) -> dict:
    """
    6-month return vs. a region-appropriate benchmark index.
    """
    out: dict = {}

    try:
        if price_series is not None and len(price_series) > 5:
            stock_hist = price_series
        else:
            stock_hist = tkr.history(period="7mo")["Close"]

        stock_return = _period_return(stock_hist, 6)

        if stock_return is None:
            return out

        out["price_return_6m"] = stock_return

        bm_symbol = _benchmark_for(symbol)
        out["benchmark_symbol"] = bm_symbol

        bm_hist = _get_ticker(bm_symbol).history(period="7mo")["Close"]
        bm_return = _period_return(bm_hist, 6)

        if bm_return is not None:
            out["benchmark_return_6m"] = bm_return
            out["relative_strength_6m"] = stock_return - bm_return

    except Exception:
        pass

    return out


def _eps_revisions(tkr: yf.Ticker) -> dict:
    """
    Analyst EPS revision counts, 30-day.
    """
    out: dict = {}

    try:
        rev = tkr.get_eps_revisions()

        if rev is None or rev.empty:
            return out

        row_lookup = {str(idx).strip().lower(): idx for idx in rev.index}

        for period_label in ("0y", "+1y", "0q", "+1q"):
            actual_row = row_lookup.get(period_label)
            if actual_row is None:
                continue

            row = rev.loc[actual_row]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            up = None
            down = None

            for col in row.index:
                col_lower = str(col).strip().lower()

                if up is None and "up" in col_lower and "30" in col_lower:
                    up = _as_float(row[col])

                if down is None and "down" in col_lower and "30" in col_lower:
                    down = _as_float(row[col])

            if up is not None:
                out["eps_revisions_up_30d"] = int(up)

            if down is not None:
                out["eps_revisions_down_30d"] = int(down)

            if up is not None or down is not None:
                break

    except Exception:
        pass

    return out


# ---------------------------------------------------------------------------
# Currency normalization
# ---------------------------------------------------------------------------

def _fx_rate(from_ccy: str, to_ccy: str) -> float | None:
    """
    Units of to_ccy per 1 unit of from_ccy.
    """
    if not from_ccy or not to_ccy:
        return None

    if from_ccy == to_ccy:
        return 1.0

    try:
        direct_tkr = yf.Ticker(f"{from_ccy}{to_ccy}=X")
        direct = _last_price_from_ticker(direct_tkr)

        if direct:
            return float(direct)

    except Exception:
        pass

    try:
        inverse_tkr = yf.Ticker(f"{to_ccy}{from_ccy}=X")
        inverse = _last_price_from_ticker(inverse_tkr)

        if inverse:
            return 1.0 / float(inverse)

    except Exception:
        pass

    return None


def _resolve_fx(tkr: yf.Ticker) -> tuple[str | None, str | None, float | None, str]:
    """
    Detects when a ticker's financial statements are reported in a
    different currency than its market cap/share price.
    """
    try:
        info, fast_info = _get_metadata(tkr)

        revenue_ccy = _safe_get(info, "financialCurrency")
        price_ccy = _safe_get(fast_info, "currency") or _safe_get(info, "currency")

        revenue_ccy = str(revenue_ccy).strip().upper() if revenue_ccy else None
        price_ccy = str(price_ccy).strip().upper() if price_ccy else None

    except Exception:
        return None, None, None, "Currency metadata unavailable."

    if not revenue_ccy or not price_ccy:
        return (
            revenue_ccy,
            price_ccy,
            None,
            "Currency metadata incomplete -- unable to verify consistency.",
        )

    if revenue_ccy == price_ccy:
        return revenue_ccy, price_ccy, 1.0, ""

    rate = _fx_rate(revenue_ccy, price_ccy)

    if rate is None:
        return revenue_ccy, price_ccy, None, (
            f"Revenue reported in {revenue_ccy} but priced in {price_ccy}, and no FX rate could be "
            f"resolved -- P/S and margin figures for this ticker are UNRELIABLE, treat with caution."
        )

    return revenue_ccy, price_ccy, rate, (
        f"Revenue converted from {revenue_ccy} to {price_ccy} at {rate:.4g} "
        f"to match the share price currency."
    )


# ---------------------------------------------------------------------------
# Main fetch entrypoints
# ---------------------------------------------------------------------------

def fetch_ticker(symbol: str, history_years: int = 5) -> RawTickerData:
    tkr = _get_ticker(symbol)

    rev_series, revenue_cadence = _revenue_series_with_cadence(tkr)

    native_revenue_ttm = _revenue_ttm_from_series(rev_series, revenue_cadence)

    if native_revenue_ttm is None or native_revenue_ttm <= 0:
        raise ValueError(f"{symbol}: no positive TTM-like revenue available")

    revenue_ccy, price_ccy, fx_rate, currency_note = _resolve_fx(tkr)

    ps_rev_series = rev_series * fx_rate if fx_rate else rev_series

    if fx_rate:
        current_revenue_ttm = _revenue_ttm_from_series(ps_rev_series, revenue_cadence)
    else:
        current_revenue_ttm = native_revenue_ttm

    if current_revenue_ttm is None or current_revenue_ttm <= 0:
        raise ValueError(f"{symbol}: no positive converted TTM-like revenue available")

    market_cap, _, shares = _market_data(tkr)

    if market_cap is None or market_cap <= 0:
        raise ValueError(
            f"{symbol}: no market cap available from Yahoo Finance metadata. "
            "The symbol may be invalid, thinly traded, delisted, or unsupported by yfinance."
        )

    try:
        hist_ps, price_series = _historical_ps_series(
            tkr,
            ps_rev_series,
            years=history_years,
            shares=shares,
            cadence=revenue_cadence,
        )
    except Exception:
        # Inclusive fallback:
        # if no historical P/S can be built, use current P/S as a single
        # low-confidence observation instead of failing the whole ticker.
        current_ps = market_cap / current_revenue_ttm
        hist_ps = pd.Series([current_ps], index=[pd.Timestamp.today()])
        price_series = _fallback_price_series(tkr, years=history_years)

    forward_rev = _forward_revenue_estimate(tkr)

    if forward_rev and fx_rate:
        forward_rev = forward_rev * fx_rate

    fundamentals = _fundamentals(tkr, native_revenue_ttm, revenue_cadence)

    extras: dict = {}
    extras.update(_growth_durability(tkr))

    roic_val = _roic(tkr)
    if roic_val is not None:
        extras["roic"] = roic_val

    dilution = _share_dilution(tkr)
    if dilution is not None:
        extras["share_count_cagr_3y"] = dilution

    extras.update(_capital_allocation(tkr, revenue_cadence))
    extras.update(_owner_earnings(tkr, market_cap, fx_rate, revenue_ccy, price_ccy, revenue_cadence))
    extras.update(_relative_strength(tkr, symbol, price_series=price_series))
    extras.update(_eps_revisions(tkr))

    price_cagr_3y = _price_cagr(price_series, 3)
    market_cap_cagr_3y = None

    if price_cagr_3y is not None:
        share_cagr = extras.get("share_count_cagr_3y")

        if share_cagr is not None and share_cagr > -1:
            market_cap_cagr_3y = ((1 + price_cagr_3y) * (1 + share_cagr)) - 1
        else:
            market_cap_cagr_3y = price_cagr_3y

    extras["price_cagr_3y"] = price_cagr_3y
    extras["market_cap_cagr_3y"] = market_cap_cagr_3y

    return RawTickerData(
        ticker=symbol,
        current_revenue_ttm=current_revenue_ttm,
        current_market_cap=float(market_cap),
        historical_ps_series=hist_ps,
        forward_revenue_estimate=forward_rev,
        revenue_cadence=revenue_cadence,
        revenue_currency=revenue_ccy,
        price_currency=price_ccy,
        fx_rate_applied=fx_rate,
        currency_note=currency_note,
        **fundamentals,
        **extras,
    )


def fetch_watchlist(tickers: list[str], history_years: int = 5, pause_seconds: float = 1.0):
    """
    Fetches each ticker in turn, yielding (symbol, RawTickerData | Exception).
    """
    for symbol in tickers:
        try:
            yield symbol, fetch_ticker(symbol, history_years=history_years)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to fetch %s: %s", symbol, exc)
            yield symbol, exc

        time.sleep(pause_seconds)
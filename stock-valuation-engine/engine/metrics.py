"""
Valuation metrics engine.

Pure computation layer.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime

import numpy as np
import pandas as pd


@dataclass
class ValuationResult:
    ticker: str
    as_of: str
    current_revenue_ttm: float
    current_market_cap: float
    current_ps: float
    hist_median_ps: float
    hist_p75_ps: float
    hist_p90_ps: float
    forward_revenue_estimate: float | None
    forward_revenue_growth: float | None
    forward_ps: float | None
    required_revenue: float
    required_growth: float
    growth_gap: float | None
    years_to_normalise: float | None
    expectations_burden_score: float | None
    expectations_classification: str
    plain_explanation: str
    target_multiple_value: float
    target_multiple_label: str
    valuation_anchor_confidence: str
    valuation_anchor_observation_count: int
    revenue_data_cadence: str

    # Fundamentals overlay
    gross_margin: float | None
    operating_margin: float | None
    net_margin: float | None
    net_debt_to_ebitda: float | None
    interest_coverage: float | None
    free_cash_flow_ttm: float | None
    fcf_margin: float | None
    cash_conversion: float | None
    quality_flag: str

    # Currency normalization metadata
    revenue_currency: str | None
    price_currency: str | None
    fx_rate_applied: float | None
    currency_note: str

    # Growth durability, management, and risk overlay
    revenue_cagr_3y: float | None
    revenue_cagr_5y: float | None
    roic: float | None
    share_count_cagr_3y: float | None
    buybacks_ttm: float | None
    dividends_ttm: float | None
    acquisitions_ttm: float | None
    price_return_6m: float | None
    benchmark_symbol: str | None
    benchmark_return_6m: float | None
    relative_strength_6m: float | None
    eps_revisions_up_30d: int | None
    eps_revisions_down_30d: int | None

    # Owner earnings overlay
    owner_earnings_ttm: float | None = None
    owner_earnings_yield: float | None = None
    maintenance_capex_ttm: float | None = None
    owner_earnings_method: str | None = None

    # Anti-bubble / multiple expansion overlay
    price_cagr_3y: float | None = None
    market_cap_cagr_3y: float | None = None
    multiple_expansion_gap_3y: float | None = None
    anti_bubble_flag: str = "Insufficient data"
    anti_bubble_note: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _is_number(value) -> bool:
    try:
        return value is not None and not pd.isna(value)
    except (TypeError, ValueError):
        return False


def _classify_confidence(n_obs: int) -> str:
    if n_obs >= 40:
        return "high"
    if n_obs >= 15:
        return "medium"
    return "low"


def _classify_expectations(burden_score: float | None) -> str:
    if burden_score is None:
        return "Insufficient data"
    if burden_score < 25:
        return "Forward Expectations Manageable"
    if burden_score < 60:
        return "Forward Expectations Elevated"
    return "Forward Expectations Stretched"


def _burden_score(required_growth: float, forward_growth: float) -> float:
    """
    0-100: how much of the growth implied by the current price is NOT
    already covered by the forward estimate.
    """
    if required_growth <= 0:
        return 0.0

    coverage = forward_growth / required_growth

    return float(np.clip(100 * (1 - min(coverage, 1)), 0, 100))


def _plain_explanation(required_growth: float, forward_growth: float, classification: str) -> str:
    """A one-line, non-technical readout."""
    req_pct = f"{required_growth * 100:.0f}%"
    fwd_pct = f"{forward_growth * 100:.0f}%"

    if required_growth <= 0:
        return (
            f"Already trades at or below its historical valuation norm (needs 0% growth) -- "
            f"{fwd_pct} forecast growth would be upside on top of an already-fair price."
        )

    if classification == "Forward Expectations Manageable":
        return (
            f"Needs {req_pct} growth to look fairly valued by its own history; analysts expect "
            f"{fwd_pct} -- the bar is comfortably cleared if the forecast is anywhere close to right."
        )

    if classification == "Forward Expectations Elevated":
        return (
            f"Needs {req_pct} growth to look fairly valued; analysts expect {fwd_pct} -- covers part "
            f"of what's required, but the price still leans on the forecast coming through."
        )

    return (
        f"Needs {req_pct} growth to look fairly valued by its own history, but analysts only expect "
        f"{fwd_pct} -- the price is asking for more growth than is currently forecast."
    )


def _classify_quality(
    operating_margin: float | None,
    net_debt_to_ebitda: float | None,
    fcf_margin: float | None,
) -> str:
    """
    A simple, explainable read on whether the growth story sits on solid fundamentals.
    """
    if operating_margin is None and net_debt_to_ebitda is None and fcf_margin is None:
        return "Insufficient data"

    flags = []

    if operating_margin is not None and operating_margin < 0:
        flags.append("unprofitable")

    if net_debt_to_ebitda is not None and net_debt_to_ebitda > 3:
        flags.append("high leverage")

    if fcf_margin is not None and fcf_margin < 0:
        flags.append("cash burning")

    if not flags:
        return "Solid"

    if len(flags) == 1:
        return f"Watch: {flags[0]}"

    return f"Caution: {', '.join(flags)}"


def _anti_bubble_read(
    market_cap_cagr_3y: float | None,
    revenue_cagr_3y: float | None,
    current_ps: float,
    hist_p75: float,
    hist_p90: float,
) -> tuple[float | None, str, str | None]:
    """
    Anti-bubble detector:

        Multiple Expansion Gap = 3Y Market Cap CAGR - 3Y Revenue CAGR
    """
    if not _is_number(market_cap_cagr_3y) or not _is_number(revenue_cagr_3y):
        return None, "Insufficient data", None

    gap = float(market_cap_cagr_3y) - float(revenue_cagr_3y)

    if gap <= 0.05:
        flag = "No bubble signal"
    elif gap <= 0.15:
        flag = "Mild multiple expansion"
    elif gap <= 0.30:
        flag = "Elevated multiple expansion"
    else:
        flag = "High multiple expansion risk"

    note = None

    if (
        gap > 0.30
        and _is_number(current_ps)
        and _is_number(hist_p90)
        and current_ps > hist_p90
    ):
        note = (
            "Market cap growth is far ahead of revenue growth while P/S is near historical highs -- "
            "multiple expansion is doing most of the work."
        )

    elif (
        gap > 0.20
        and _is_number(current_ps)
        and _is_number(hist_p75)
        and current_ps > hist_p75
    ):
        note = (
            "Market cap growth is outpacing revenue growth and the current multiple is above its "
            "historical 75th percentile -- expansion risk is elevated."
        )

    elif gap > 0.20:
        note = (
            "Market cap growth is outpacing revenue growth, but the current multiple is not at an "
            "extreme historical level."
        )

    elif gap > 0.05:
        note = "Market cap has grown somewhat faster than revenue over the last 3 years."

    else:
        note = "Market cap growth has not materially outpaced revenue growth over the last 3 years."

    return gap, flag, note


def compute_valuation(
    ticker: str,
    current_revenue_ttm: float,
    current_market_cap: float,
    historical_ps_series: pd.Series,
    forward_revenue_estimate: float | None,
    revenue_cadence: str = "quarterly",
    as_of: datetime | None = None,
    gross_margin: float | None = None,
    operating_margin: float | None = None,
    net_margin: float | None = None,
    net_debt_to_ebitda: float | None = None,
    interest_coverage: float | None = None,
    free_cash_flow_ttm: float | None = None,
    fcf_margin: float | None = None,
    cash_conversion: float | None = None,
    revenue_currency: str | None = None,
    price_currency: str | None = None,
    fx_rate_applied: float | None = None,
    currency_note: str = "",
    revenue_cagr_3y: float | None = None,
    revenue_cagr_5y: float | None = None,
    roic: float | None = None,
    share_count_cagr_3y: float | None = None,
    buybacks_ttm: float | None = None,
    dividends_ttm: float | None = None,
    acquisitions_ttm: float | None = None,
    price_return_6m: float | None = None,
    benchmark_symbol: str | None = None,
    benchmark_return_6m: float | None = None,
    relative_strength_6m: float | None = None,
    eps_revisions_up_30d: int | None = None,
    eps_revisions_down_30d: int | None = None,
    owner_earnings_ttm: float | None = None,
    owner_earnings_yield: float | None = None,
    maintenance_capex_ttm: float | None = None,
    owner_earnings_method: str | None = None,
    price_cagr_3y: float | None = None,
    market_cap_cagr_3y: float | None = None,
) -> ValuationResult:
    """
    historical_ps_series:
        pandas Series of historical trailing P/S observations.
    """
    as_of = as_of or datetime.utcnow()

    current_ps = (
        current_market_cap / current_revenue_ttm
        if current_revenue_ttm
        else float("nan")
    )

    clean = historical_ps_series.dropna() if historical_ps_series is not None else pd.Series(dtype=float)
    clean = clean[clean > 0]

    n_obs = len(clean)

    hist_median = float(clean.median()) if n_obs else float("nan")
    hist_p75 = float(clean.quantile(0.75)) if n_obs else float("nan")
    hist_p90 = float(clean.quantile(0.90)) if n_obs else float("nan")

    target_multiple_value = hist_median if _is_number(hist_median) and hist_median > 0 else float("nan")
    target_multiple_label = "Historical Median Tactical Anchor"

    required_revenue = (
        current_market_cap / target_multiple_value
        if _is_number(target_multiple_value) and target_multiple_value > 0
        else float("nan")
    )

    required_growth = (
        (required_revenue / current_revenue_ttm) - 1
        if _is_number(required_revenue) and current_revenue_ttm
        else float("nan")
    )

    forward_growth = None
    forward_ps = None
    growth_gap = None
    years_to_normalise = None
    burden_score = None

    if (
        forward_revenue_estimate
        and forward_revenue_estimate > 0
        and _is_number(current_revenue_ttm)
        and current_revenue_ttm > 0
    ):
        forward_growth = (forward_revenue_estimate / current_revenue_ttm) - 1
        forward_ps = current_market_cap / forward_revenue_estimate

        if _is_number(required_growth):
            growth_gap = required_growth - forward_growth

            if forward_growth > 0:
                years_to_normalise = np.log(required_revenue / current_revenue_ttm) / np.log(1 + forward_growth)
                years_to_normalise = max(years_to_normalise, 0)

            burden_score = _burden_score(required_growth, forward_growth)

    classification = _classify_expectations(burden_score)

    if not _is_number(required_growth):
        explanation = (
            "Not enough historical P/S observations to estimate required growth reliably -- "
            "current P/S is shown as a low-confidence reference."
        )
    elif forward_growth is None:
        explanation = (
            f"No analyst forward revenue estimate available -- would need {required_growth*100:.0f}% "
            f"growth to look fairly valued by its own history, but there's no forecast to check that against."
        )
    else:
        explanation = _plain_explanation(required_growth, forward_growth, classification)

    if currency_note and fx_rate_applied is None and revenue_currency and price_currency:
        explanation = f"⚠ {currency_note} {explanation}"

    multiple_expansion_gap_3y, anti_bubble_flag, anti_bubble_note = _anti_bubble_read(
        market_cap_cagr_3y=market_cap_cagr_3y,
        revenue_cagr_3y=revenue_cagr_3y,
        current_ps=current_ps,
        hist_p75=hist_p75,
        hist_p90=hist_p90,
    )

    return ValuationResult(
        ticker=ticker,
        as_of=as_of.isoformat(),
        current_revenue_ttm=current_revenue_ttm,
        current_market_cap=current_market_cap,
        current_ps=current_ps,
        hist_median_ps=hist_median,
        hist_p75_ps=hist_p75,
        hist_p90_ps=hist_p90,
        forward_revenue_estimate=forward_revenue_estimate,
        forward_revenue_growth=forward_growth,
        forward_ps=forward_ps,
        required_revenue=required_revenue,
        required_growth=required_growth,
        growth_gap=growth_gap,
        years_to_normalise=years_to_normalise,
        expectations_burden_score=burden_score,
        expectations_classification=classification,
        plain_explanation=explanation,
        target_multiple_value=target_multiple_value,
        target_multiple_label=target_multiple_label,
        valuation_anchor_confidence=_classify_confidence(n_obs),
        valuation_anchor_observation_count=n_obs,
        revenue_data_cadence=revenue_cadence,
        gross_margin=gross_margin,
        operating_margin=operating_margin,
        net_margin=net_margin,
        net_debt_to_ebitda=net_debt_to_ebitda,
        interest_coverage=interest_coverage,
        free_cash_flow_ttm=free_cash_flow_ttm,
        fcf_margin=fcf_margin,
        cash_conversion=cash_conversion,
        quality_flag=_classify_quality(operating_margin, net_debt_to_ebitda, fcf_margin),
        revenue_currency=revenue_currency,
        price_currency=price_currency,
        fx_rate_applied=fx_rate_applied,
        currency_note=currency_note,
        revenue_cagr_3y=revenue_cagr_3y,
        revenue_cagr_5y=revenue_cagr_5y,
        roic=roic,
        share_count_cagr_3y=share_count_cagr_3y,
        buybacks_ttm=buybacks_ttm,
        dividends_ttm=dividends_ttm,
        acquisitions_ttm=acquisitions_ttm,
        price_return_6m=price_return_6m,
        benchmark_symbol=benchmark_symbol,
        benchmark_return_6m=benchmark_return_6m,
        relative_strength_6m=relative_strength_6m,
        eps_revisions_up_30d=eps_revisions_up_30d,
        eps_revisions_down_30d=eps_revisions_down_30d,
        owner_earnings_ttm=owner_earnings_ttm,
        owner_earnings_yield=owner_earnings_yield,
        maintenance_capex_ttm=maintenance_capex_ttm,
        owner_earnings_method=owner_earnings_method,
        price_cagr_3y=price_cagr_3y,
        market_cap_cagr_3y=market_cap_cagr_3y,
        multiple_expansion_gap_3y=multiple_expansion_gap_3y,
        anti_bubble_flag=anti_bubble_flag,
        anti_bubble_note=anti_bubble_note,
    )
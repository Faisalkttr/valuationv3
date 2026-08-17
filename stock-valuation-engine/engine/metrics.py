"""
Valuation metrics engine -- pure computation layer.
Includes owner-earnings pass-through, the anti-bubble detector, and a
data-quality guard layer that neutralises/flags implausible yfinance inputs
(units mismatches, broken histories, extreme analyst estimates).
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

    gross_margin: float | None
    operating_margin: float | None
    net_margin: float | None
    net_debt_to_ebitda: float | None
    interest_coverage: float | None
    free_cash_flow_ttm: float | None
    fcf_margin: float | None
    cash_conversion: float | None
    quality_flag: str

    revenue_currency: str | None
    price_currency: str | None
    fx_rate_applied: float | None
    currency_note: str

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

    owner_earnings_ttm: float | None = None
    owner_earnings_yield: float | None = None
    maintenance_capex_ttm: float | None = None
    owner_earnings_method: str | None = None

    price_cagr_3y: float | None = None
    market_cap_cagr_3y: float | None = None
    multiple_expansion_gap_3y: float | None = None
    anti_bubble_flag: str = "Insufficient data"
    anti_bubble_note: str | None = None

    # ROIC persistence (Problem #2: use ROIC "more aggressively")
    roic_avg_3y: float | None = None
    roic_trend: float | None = None
    roic_persistence_score: float | None = None
    roic_persistence_label: str = "Insufficient data"

    # Owner-earnings valuation pillar for mature/quality compounders
    # (Problem #1: promote Owner Earnings Yield to a first-class anchor)
    is_mature_company: bool = False
    owner_earnings_richness_score: float | None = None
    valuation_burden_score_blended: float | None = None
    valuation_pillar: str = "P/S growth-gap"

    # Dynamic DCA / payday allocation (Problem #3)
    allocation_classification: str = "Insufficient data"
    allocation_multiplier: float = 1.0

    # Expected forward CAGR (Problem #4)
    expected_cagr: float | None = None
    expected_cagr_growth_component: float | None = None
    expected_cagr_income_component: float | None = None
    expected_cagr_reversion_component: float | None = None

    # Buy Priority Score (Problem #5) -- filled in cross-sectionally by
    # compute_buy_priority_scores() once every ticker in a run is known;
    # left at None for a single ad-hoc lookup where there's no peer set.
    buy_priority_score: float | None = None

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
    if required_growth <= 0:
        return 0.0
    coverage = forward_growth / required_growth
    return float(np.clip(100 * (1 - min(coverage, 1)), 0, 100))


def _plain_explanation(required_growth: float, forward_growth: float, classification: str) -> str:
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


def _classify_quality(operating_margin, net_debt_to_ebitda, fcf_margin, net_margin=None) -> str:
    if (operating_margin is None and net_debt_to_ebitda is None
            and fcf_margin is None and net_margin is None):
        return "Insufficient data"

    flags = []

    if operating_margin is not None and operating_margin < 0:
        flags.append("unprofitable")
    elif operating_margin is None and net_margin is not None and net_margin < 0:
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


def _roic_persistence(roic, roic_avg_3y, roic_trend):
    """Score 0-100 (higher = more durable/improving returns on capital).

    Combines the *level* of ROIC (is it a good business at all) with its
    *trend* (is that advantage widening or eroding), so a high-but-fading
    ROIC scores lower than a high-and-rising one.
    """
    level = roic_avg_3y if _is_number(roic_avg_3y) else roic
    if not _is_number(level):
        return None, "Insufficient data"

    # Up to 70 pts for the level, mapped linearly across a 5%-30% ROIC band.
    level_score = float(np.clip((level - 0.05) / (0.30 - 0.05), 0, 1) * 70)

    # Up to +/-30 pts for the trend, mapped across a +/-5pp swing.
    trend_score = 0.0
    if _is_number(roic_trend):
        trend_score = float(np.clip(roic_trend / 0.05, -1, 1) * 30)

    score = float(np.clip(level_score + trend_score, 0, 100))

    if score >= 70:
        label = "High & durable"
    elif score >= 45:
        label = "Moderate"
    elif score >= 20:
        label = "Weak / inconsistent"
    else:
        label = "Poor"
    return score, label


# Rough required owner-earnings yield used as the "fair value" bar for the
# owner-earnings pillar -- i.e. what an owner-earnings-yield needs to clear
# to look fairly priced. This is a simplifying constant, not a market-implied
# discount rate; treat the resulting score as directional, not precise.
OWNER_EARNINGS_REQUIRED_YIELD = 0.045


def _oe_richness_score(owner_earnings_yield, required_yield=OWNER_EARNINGS_REQUIRED_YIELD):
    """Score 0-100 on the SAME scale as the P/S growth-gap burden score
    (0 = cheap, 100 = expensive), so the two can be blended directly.
    """
    if not _is_number(owner_earnings_yield) or required_yield <= 0:
        return None
    coverage = owner_earnings_yield / required_yield
    return float(np.clip(100 * (1 - min(coverage, 2) / 2), 0, 100))


# Finer-grained allocation tiers for dynamic-DCA "payday" sizing. Each
# threshold is the UPPER bound (exclusive) of the blended burden score that
# falls into that tier.
_ALLOCATION_TIERS = (
    (15, "Deep Undervalued", 2.0),
    (35, "Undervalued", 1.5),
    (55, "Fair Value", 1.0),
    (75, "Expensive", 0.75),
    (101, "Very Expensive", 0.5),
)


def _allocation_tier(blended_score: float | None) -> tuple[str, float]:
    if not _is_number(blended_score):
        return "Insufficient data", 1.0
    for upper_bound, label, multiplier in _ALLOCATION_TIERS:
        if blended_score < upper_bound:
            return label, multiplier
    return "Very Expensive", 0.5


def _expected_cagr(
    forward_growth: float | None,
    revenue_cagr_3y: float | None,
    owner_earnings_yield: float | None,
    target_multiple_value: float | None,
    current_ps: float | None,
    horizon_years: float = 3.0,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Expected CAGR ~= growth + owner-earnings yield + valuation mean reversion.

    Returns (expected_cagr, growth_component, income_component, reversion_component).
    All components are None if the required inputs aren't available, but a
    missing income or reversion component still allows a partial estimate
    (they're just treated as 0 contribution) as long as a growth component exists.
    """
    growth_component = forward_growth if _is_number(forward_growth) else revenue_cagr_3y
    if not _is_number(growth_component):
        return None, None, None, None

    income_component = owner_earnings_yield if _is_number(owner_earnings_yield) else 0.0

    reversion_component = 0.0
    if (
        _is_number(target_multiple_value) and target_multiple_value > 0
        and _is_number(current_ps) and current_ps > 0
        and horizon_years > 0
    ):
        total_reversion = (target_multiple_value / current_ps) - 1
        reversion_component = total_reversion / horizon_years

    expected = float(growth_component) + float(income_component) + float(reversion_component)
    return expected, float(growth_component), float(income_component), float(reversion_component)


def _anti_bubble_read(market_cap_cagr_3y, revenue_cagr_3y, current_ps, hist_p75, hist_p90):
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
    if gap > 0.30 and _is_number(current_ps) and _is_number(hist_p90) and current_ps > hist_p90:
        note = ("Market cap growth is far ahead of revenue growth while P/S is near historical highs -- "
                "multiple expansion is doing most of the work.")
    elif gap > 0.20 and _is_number(current_ps) and _is_number(hist_p75) and current_ps > hist_p75:
        note = ("Market cap growth is outpacing revenue growth and the current multiple is above its "
                "historical 75th percentile -- expansion risk is elevated.")
    elif gap > 0.20:
        note = ("Market cap growth is outpacing revenue growth, but the current multiple is not at an "
                "extreme historical level.")
    elif gap > 0.05:
        note = "Market cap has grown somewhat faster than revenue over the last 3 years."
    else:
        note = "Market cap growth has not materially outpaced revenue growth over the last 3 years."

    return gap, flag, note


# Guardrail constants
PS_MIN, PS_MAX = 0.05, 100.0          # plausible P/S band for operating companies
STRUCTURAL_RATIO = 3.0                # current vs median P/S break threshold
FWD_RATIO_MIN, FWD_RATIO_MAX = 0.2, 5.0  # forward estimate vs TTM revenue sanity band
FWD_GROWTH_CAP = 1.0                  # 100% cap for gap/burden math


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
    roic_avg_3y: float | None = None,
    roic_trend: float | None = None,
) -> ValuationResult:
    as_of = as_of or datetime.utcnow()

    current_ps = current_market_cap / current_revenue_ttm if current_revenue_ttm else float("nan")

    clean = historical_ps_series.dropna() if historical_ps_series is not None else pd.Series(dtype=float)
    clean = clean[clean > 0]
    n_obs = len(clean)

    hist_median = float(clean.median()) if n_obs else float("nan")
    hist_p75 = float(clean.quantile(0.75)) if n_obs else float("nan")
    hist_p90 = float(clean.quantile(0.90)) if n_obs else float("nan")

    target_multiple_value = hist_median if _is_number(hist_median) and hist_median > 0 else float("nan")
    target_multiple_label = "Historical Median Tactical Anchor"

    # ------------------------------------------------------------------
    # ROIC persistence -- a business with a high and rising ROIC deserves a
    # permanently higher multiple than its own stale history implies (and
    # vice versa for a fading one), so nudge the tactical anchor by up to
    # +/-20% based on the persistence score rather than trusting the
    # historical median blindly.
    # ------------------------------------------------------------------
    roic_persistence_score, roic_persistence_label = _roic_persistence(roic, roic_avg_3y, roic_trend)

    if _is_number(target_multiple_value) and target_multiple_value > 0 and roic_persistence_score is not None:
        adj_factor = 1 + ((roic_persistence_score - 50) / 50) * 0.20  # +/-20% max
        target_multiple_value = target_multiple_value * adj_factor
        target_multiple_label = "Historical Median Tactical Anchor (ROIC-adjusted)"

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

    # ------------------------------------------------------------------
    # DATA-QUALITY GUARDS -- yfinance numbers are trusted only after
    # passing simple plausibility checks. Failures never crash the row;
    # they neutralise the affected input and/or add a visible warning.
    # ------------------------------------------------------------------
    sanity_notes: list[str] = []

    # Guard 1: P/S outside any plausible band -> units/currency mismatch.
    if _is_number(current_ps) and (current_ps < PS_MIN or current_ps > PS_MAX):
        sanity_notes.append(
            f"Current P/S {current_ps:.2f} is outside the plausible {PS_MIN}-{PS_MAX:.0f} band -- "
            "likely a units/currency mismatch in the source data; treat all multiples with caution."
        )

    # Guard 2: current P/S far away from its own history -> structural break
    # (ADR ratio change, share-count jump, FX break, broken history).
    if (
        n_obs >= 8
        and _is_number(current_ps) and current_ps > 0
        and _is_number(hist_median) and hist_median > 0
    ):
        ps_ratio = current_ps / hist_median
        if ps_ratio > STRUCTURAL_RATIO or ps_ratio < 1.0 / STRUCTURAL_RATIO:
            sanity_notes.append(
                f"Current P/S is {ps_ratio:.1f}x its own historical median -- suspect structural break "
                "(ADR/share ratio change, share-count jump, FX, or broken history); percentiles and "
                "growth-gap are unreliable for this ticker."
            )

    # Guard 3: forward estimate in different units than reported revenue.
    if forward_revenue_estimate and forward_revenue_estimate > 0 and current_revenue_ttm:
        fwd_ratio = forward_revenue_estimate / current_revenue_ttm
        if fwd_ratio < FWD_RATIO_MIN or fwd_ratio > FWD_RATIO_MAX:
            sanity_notes.append(
                f"Forward revenue estimate is {fwd_ratio:.1f}x TTM revenue (outside {FWD_RATIO_MIN}-"
                f"{FWD_RATIO_MAX}x) -- units mismatch suspected; forward fields ignored."
            )
            forward_revenue_estimate = None

    # Guard 4: honesty note when history is FX-converted at today's rate.
    if fx_rate_applied is not None and fx_rate_applied != 1.0:
        sanity_notes.append(
            "Historical P/S converts all past periods at TODAY's FX rate -- percentiles are "
            "approximate for foreign-currency reporters (material mainly for high-inflation currencies)."
        )

    # ------------------------------------------------------------------
    # Forward block (with extreme-growth cap for gap/burden math)
    # ------------------------------------------------------------------
    forward_growth = None
    forward_ps = None
    growth_gap = None
    years_to_normalise = None
    burden_score = None

    if (forward_revenue_estimate and forward_revenue_estimate > 0
            and _is_number(current_revenue_ttm) and current_revenue_ttm > 0):
        forward_growth = (forward_revenue_estimate / current_revenue_ttm) - 1
        forward_ps = current_market_cap / forward_revenue_estimate

        growth_for_math = forward_growth
        if forward_growth is not None and forward_growth > FWD_GROWTH_CAP:
            growth_for_math = FWD_GROWTH_CAP
            sanity_notes.append(
                f"Analyst forward growth {forward_growth*100:.0f}% is extreme; capped at 100% for "
                "growth-gap/burden math (displayed forward growth is unchanged)."
            )

        if _is_number(required_growth):
            growth_gap = required_growth - growth_for_math

            if growth_for_math > 0:
                years_to_normalise = np.log(required_revenue / current_revenue_ttm) / np.log(1 + growth_for_math)
                years_to_normalise = max(years_to_normalise, 0)

            burden_score = _burden_score(required_growth, growth_for_math)

    # ------------------------------------------------------------------
    # Owner-earnings pillar (Problem #1) -- for mature, low-growth compounders
    # revenue growth is not the main value driver, so P/S alone can misprice
    # them. Blend in an owner-earnings-yield richness score 50/50 once a
    # company looks "mature" (revenue growth has slowed).
    # ------------------------------------------------------------------
    is_mature_company = _is_number(revenue_cagr_3y) and revenue_cagr_3y < 0.12
    oe_richness_score = _oe_richness_score(owner_earnings_yield)

    valuation_burden_score_blended = burden_score
    valuation_pillar = "P/S growth-gap"

    if is_mature_company and oe_richness_score is not None:
        if burden_score is not None:
            valuation_burden_score_blended = 0.5 * burden_score + 0.5 * oe_richness_score
            valuation_pillar = "Blended 50/50: P/S growth-gap + Owner Earnings Yield (mature-company mode)"
        else:
            valuation_burden_score_blended = oe_richness_score
            valuation_pillar = "Owner Earnings Yield only (mature-company mode, no forward P/S read)"

    classification = _classify_expectations(valuation_burden_score_blended)
    allocation_classification, allocation_multiplier = _allocation_tier(valuation_burden_score_blended)

    expected_cagr, expected_cagr_growth, expected_cagr_income, expected_cagr_reversion = _expected_cagr(
        forward_growth=forward_growth,
        revenue_cagr_3y=revenue_cagr_3y,
        owner_earnings_yield=owner_earnings_yield,
        target_multiple_value=target_multiple_value,
        current_ps=current_ps,
    )

    if not _is_number(required_growth):
        explanation = ("Not enough historical P/S observations to estimate required growth reliably -- "
                       "current P/S is shown as a low-confidence reference.")
    elif forward_growth is None:
        explanation = (
            f"No usable analyst forward revenue estimate -- would need {required_growth*100:.0f}% "
            f"growth to look fairly valued by its own history, but there's no reliable forecast to "
            f"check that against."
        )
    else:
        explanation = _plain_explanation(required_growth, forward_growth, classification)

    if is_mature_company and oe_richness_score is not None:
        oey_text = f"{owner_earnings_yield*100:.1f}%" if _is_number(owner_earnings_yield) else "n/a"
        explanation = (
            f"{explanation} Mature-company read: owner earnings yield ({oey_text}) is blended 50/50 "
            f"with the P/S growth-gap for the final classification, since revenue growth alone is a "
            f"weak value driver here."
        )

    if sanity_notes:
        explanation = "⚠ " + " ⚠ ".join(sanity_notes) + " " + explanation

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
        quality_flag=_classify_quality(operating_margin, net_debt_to_ebitda, fcf_margin, net_margin),
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
        roic_avg_3y=roic_avg_3y,
        roic_trend=roic_trend,
        roic_persistence_score=roic_persistence_score,
        roic_persistence_label=roic_persistence_label,
        is_mature_company=is_mature_company,
        owner_earnings_richness_score=oe_richness_score,
        valuation_burden_score_blended=valuation_burden_score_blended,
        valuation_pillar=valuation_pillar,
        allocation_classification=allocation_classification,
        allocation_multiplier=allocation_multiplier,
        expected_cagr=expected_cagr,
        expected_cagr_growth_component=expected_cagr_growth,
        expected_cagr_income_component=expected_cagr_income,
        expected_cagr_reversion_component=expected_cagr_reversion,
    )


# ---------------------------------------------------------------------------
# Buy Priority Score (Problem #5) -- cross-sectional, so it only makes sense
# once every ticker in a run has been scored. Call this on the full results
# DataFrame after compute_valuation() has run for every ticker.
# ---------------------------------------------------------------------------

_QUALITY_SCORE_MAP_PREFIX = {
    "Solid": 100.0,
    "Watch": 60.0,
    "Caution": 20.0,
}


def _quality_to_score(flag) -> float | None:
    if not isinstance(flag, str) or not flag or flag == "Insufficient data":
        return None
    for prefix, score in _QUALITY_SCORE_MAP_PREFIX.items():
        if flag.startswith(prefix):
            return score
    return None


def _pct_rank_0_100(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Cross-sectional percentile rank on a 0-100 scale, NaN-safe.

    Ties get the average rank (pandas default), and rows with no data get
    NaN back rather than being silently dropped from the index.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    ranked = numeric.rank(pct=True, na_option="keep")
    if not higher_is_better:
        ranked = 1 - ranked
    return ranked * 100


def compute_buy_priority_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a `buy_priority_score` (0-100) column ranking every row against
    its peers in `df`, per the weighting scheme:

        30% Valuation (cheapness, from the blended burden score)
      + 25% Quality (fundamentals flag)
      + 20% Growth durability (revenue CAGR 3Y/5Y)
      + 15% Owner earnings yield
      + 10% Anti-bubble (low multiple-expansion gap)

    Each component is a cross-sectional percentile rank so the score is
    always relative to the current watchlist, not an absolute grade.
    Returns a copy of `df` with the new column; does not mutate the input.
    """
    out = df.copy()
    if out.empty:
        out["buy_priority_score"] = pd.Series(dtype=float)
        return out

    valuation_source = (
        out["valuation_burden_score_blended"]
        if "valuation_burden_score_blended" in out.columns
        else out.get("expectations_burden_score", pd.Series(index=out.index, dtype=float))
    )
    valuation_score = _pct_rank_0_100(valuation_source, higher_is_better=False)  # lower burden = cheaper = better

    quality_score = (
        out["quality_flag"].apply(_quality_to_score)
        if "quality_flag" in out.columns
        else pd.Series(index=out.index, dtype=float)
    )

    growth_source = out.get("revenue_cagr_3y")
    if growth_source is None or growth_source.isna().all():
        growth_source = out.get("revenue_cagr_5y", pd.Series(index=out.index, dtype=float))
    growth_score = _pct_rank_0_100(growth_source, higher_is_better=True)

    oey_score = _pct_rank_0_100(out.get("owner_earnings_yield", pd.Series(index=out.index, dtype=float)),
                                 higher_is_better=True)

    bubble_score = _pct_rank_0_100(out.get("multiple_expansion_gap_3y", pd.Series(index=out.index, dtype=float)),
                                    higher_is_better=False)  # smaller gap = less bubbly = better

    weighted = pd.concat(
        {
            "valuation": valuation_score * 0.30,
            "quality": quality_score * 0.25,
            "growth": growth_score * 0.20,
            "oey": oey_score * 0.15,
            "bubble": bubble_score * 0.10,
        },
        axis=1,
    )

    # Renormalise per-row over whichever components actually have data, so a
    # ticker missing e.g. owner-earnings data isn't unfairly penalised versus
    # one where every component is populated.
    weights_used = pd.Series(
        {"valuation": 0.30, "quality": 0.25, "growth": 0.20, "oey": 0.15, "bubble": 0.10}
    )
    present_weight = weighted.notna().mul(weights_used, axis=1).sum(axis=1)
    score = weighted.sum(axis=1, min_count=1) / present_weight.replace(0, np.nan)

    out["buy_priority_score"] = score.round(1)
    return out

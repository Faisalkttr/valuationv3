"""Offline sanity check for the new metrics -- no network calls, just
fabricated inputs to compute_valuation() and compute_buy_priority_scores()
to catch exceptions / obviously-wrong output before touching real data.
"""
import pandas as pd

from engine.metrics import compute_valuation, compute_buy_priority_scores

hist_ps = pd.Series([10, 11, 12, 13, 14, 15], index=pd.date_range("2021-01-01", periods=6, freq="YE"))

# Case 1: high-growth, cheap-vs-history, no owner earnings -> plain P/S path
r1 = compute_valuation(
    ticker="GROWCO",
    current_revenue_ttm=1_000_000_000,
    current_market_cap=10_000_000_000,
    historical_ps_series=hist_ps,
    forward_revenue_estimate=1_300_000_000,
    revenue_cagr_3y=0.25,
    owner_earnings_yield=None,
    roic=0.18,
    roic_avg_3y=0.20,
    roic_trend=0.03,
)
assert r1.is_mature_company is False, "25% revenue growth should not be flagged mature"
assert r1.valuation_pillar == "P/S growth-gap"
assert r1.roic_persistence_score is not None and r1.roic_persistence_score > 50
assert r1.allocation_multiplier in (0.5, 0.75, 1.0, 1.5, 2.0)
print("Case 1 OK:", r1.expectations_classification, r1.allocation_classification,
      r1.allocation_multiplier, r1.roic_persistence_label, r1.expected_cagr)

# Case 2: mature, slow-growth compounder with strong owner earnings yield ->
# blended P/S + owner-earnings pillar should kick in
r2 = compute_valuation(
    ticker="MATURECO",
    current_revenue_ttm=5_000_000_000,
    current_market_cap=40_000_000_000,
    historical_ps_series=hist_ps,
    forward_revenue_estimate=5_250_000_000,  # 5% growth
    revenue_cagr_3y=0.05,
    owner_earnings_yield=0.06,  # rich owner earnings yield
    roic=0.10,
    roic_avg_3y=0.09,
    roic_trend=-0.01,
)
assert r2.is_mature_company is True
assert "Blended" in r2.valuation_pillar or "Owner Earnings" in r2.valuation_pillar
assert r2.owner_earnings_richness_score is not None
print("Case 2 OK:", r2.valuation_pillar, r2.valuation_burden_score_blended,
      r2.allocation_classification, r2.expected_cagr)

# Case 3: missing almost everything -> should not crash, should degrade gracefully
r3 = compute_valuation(
    ticker="THIN",
    current_revenue_ttm=100_000_000,
    current_market_cap=500_000_000,
    historical_ps_series=pd.Series(dtype=float),
    forward_revenue_estimate=None,
)
assert r3.roic_persistence_score is None
assert r3.allocation_classification == "Insufficient data"
assert r3.expected_cagr is None
print("Case 3 OK: degrades gracefully with no data")

# Cross-sectional buy priority score across the three cases
df = pd.DataFrame([r1.to_dict(), r2.to_dict(), r3.to_dict()])
scored = compute_buy_priority_scores(df)
assert "buy_priority_score" in scored.columns
assert scored["buy_priority_score"].notna().sum() >= 2
print("Buy priority scores:\n", scored[["ticker", "buy_priority_score"]])

print("\nAll offline sanity checks passed.")

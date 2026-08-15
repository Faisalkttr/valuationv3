"""
Streamlit dashboard for the valuation engine.
"""

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.fetch import fetch_ticker
from engine.metrics import compute_valuation

ROOT = Path(__file__).resolve().parent
LATEST_CSV = ROOT / "data" / "latest.csv"
HISTORY_DB = ROOT / "data" / "history.db"

st.set_page_config(page_title="Valuation Terminal", layout="wide", initial_sidebar_state="collapsed")


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

SECTION_COLORS = {
    "INFRA": "#5B8DBE",
    "ENERGY & COMMODITY": "#C97C3D",
    "AI / SEMIS": "#8B6FD6",
    "EM": "#D4A94A",
    "Business & Futuristic Overlay": "#D65F7A",
}

DEFAULT_ACCENT = "#8B949E"

STATUS_COLORS = {
    "Forward Expectations Manageable": "#4FBF7A",
    "Forward Expectations Elevated": "#E0A63A",
    "Forward Expectations Stretched": "#E0584F",
    "Insufficient data": "#8B949E",
}

BG = "#0D1117"
SURFACE = "#161B22"
BORDER = "#262C36"
TEXT = "#E6EDF3"
TEXT_MUTED = "#8B949E"


def section_color(section: str) -> str:
    return SECTION_COLORS.get(section, DEFAULT_ACCENT)


def status_color(classification: str) -> str:
    return STATUS_COLORS.get(classification, DEFAULT_ACCENT)


def quality_color(flag: str) -> str:
    if flag == "Solid":
        return "#4FBF7A"
    if flag.startswith("Watch"):
        return "#E0A63A"
    if flag.startswith("Caution"):
        return "#E0584F"
    return "#8B949E"


def anti_bubble_color(flag: str) -> str:
    if flag == "No bubble signal":
        return "#4FBF7A"
    if flag == "Mild multiple expansion":
        return "#E0A63A"
    if flag in ("Elevated multiple expansion", "High multiple expansion risk"):
        return "#E0584F"
    return "#8B949E"


def safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def fmt_multiple(value) -> str:
    v = safe_float(value)
    return "n/a" if v is None else f"{v:.2f}x"


def fmt_pct(value, digits: int = 1) -> str:
    v = safe_float(value)
    return "n/a" if v is None else f"{v * 100:.{digits}f}%"


def fmt_signed_pct(value, digits: int = 1) -> str:
    v = safe_float(value)
    return "n/a" if v is None else f"{v * 100:+.{digits}f}%"


def fmt_signed_pp(value, digits: int = 1) -> str:
    v = safe_float(value)
    return "n/a" if v is None else f"{v * 100:+.{digits}f}pp"


def fmt_large_amount(value) -> str:
    v = safe_float(value)

    if v is None:
        return "n/a"

    abs_value = abs(v)

    if abs_value >= 1_000_000_000:
        return f"${v / 1_000_000_000:,.2f}B"

    if abs_value >= 1_000_000:
        return f"${v / 1_000_000:,.0f}M"

    if abs_value >= 1_000:
        return f"${v / 1_000:,.0f}K"

    return f"${v:,.0f}"


st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.stApp {{
    background-color: {BG};
}}

h1, h2, h3 {{
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.01em;
}}

[data-testid="stMetricValue"], .mono {{
    font-family: 'IBM Plex Mono', monospace !important;
}}

.terminal-header {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    border-bottom: 1px solid {BORDER};
    padding-bottom: 14px;
    margin-bottom: 6px;
}}

.terminal-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: {TEXT};
    margin: 0;
}}

.terminal-sub {{
    color: {TEXT_MUTED};
    font-size: 0.85rem;
    font-family: 'IBM Plex Mono', monospace;
}}

.summary-row {{
    display: flex;
    gap: 10px;
    margin: 18px 0 22px 0;
    flex-wrap: wrap;
}}

.summary-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 12px 18px;
    flex: 1;
    min-width: 140px;
}}

.summary-card .label {{
    color: {TEXT_MUTED};
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 4px;
}}

.summary-card .value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.4rem;
    font-weight: 600;
    color: {TEXT};
}}

.chip-row {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-bottom: 16px;
}}

.chip {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 999px;
    padding: 5px 12px;
    font-size: 0.78rem;
    color: {TEXT};
}}

.chip .dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
}}

.section-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-left: 4px solid var(--accent);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}}

.section-card .sc-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    color: {TEXT};
    margin-bottom: 2px;
}}

.section-card .sc-sub {{
    color: {TEXT_MUTED};
    font-size: 0.8rem;
}}

.detail-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-top: 3px solid var(--accent);
    border-radius: 10px;
    padding: 14px 16px;
    height: 100%;
}}

.detail-card .dc-label {{
    color: {TEXT_MUTED};
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}}

.detail-card .dc-value {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: {TEXT};
}}

.badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 500;
}}

[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_latest() -> pd.DataFrame:
    if not LATEST_CSV.exists():
        return pd.DataFrame()

    return pd.read_csv(LATEST_CSV)


@st.cache_data(ttl=3600)
def load_history(ticker: str) -> pd.DataFrame:
    if not HISTORY_DB.exists():
        return pd.DataFrame()

    with sqlite3.connect(HISTORY_DB) as conn:
        df = pd.read_sql(
            "SELECT * FROM valuation_snapshots WHERE ticker = ? ORDER BY as_of",
            conn,
            params=(ticker,),
        )

    if not df.empty:
        df["as_of"] = pd.to_datetime(df["as_of"])

    return df


def render_ticker_detail(row, show_trend: bool = True):
    """
    Renders the full ticker breakdown.
    Works for both stored rows and ad-hoc dictionary results.
    """
    ticker = row.get("ticker", "Unknown")
    accent = section_color(row.get("section", "")) if pd.notna(row.get("section")) else DEFAULT_ACCENT

    header_bits = [ticker]

    if pd.notna(row.get("section")):
        header_bits.append(f"{row['section']} · {row.get('layer', '')}")

    st.markdown(
        f'<div style="color:{TEXT_MUTED};font-size:0.85rem;margin-bottom:10px;">{" · ".join(header_bits)}</div>',
        unsafe_allow_html=True,
    )

    if row.get("currency_note") and pd.isna(row.get("fx_rate_applied")) and pd.notna(row.get("revenue_currency")):
        st.warning(f"⚠ {row['currency_note']}")

    def card(col, label, value, color=None):
        color = color or accent

        col.markdown(
            f"""
            <div class="detail-card" style="--accent:{color}">
                <div class="dc-label">{label}</div>
                <div class="dc-value">{value}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------
    # Valuation
    # -----------------------------------------------------------------
    st.markdown("##### Valuation")

    c1, c2, c3, c4 = st.columns(4)

    card(c1, "Current P/S", fmt_multiple(row.get("current_ps")))
    card(c2, "Historical median", fmt_multiple(row.get("hist_median_ps")))
    card(c3, "Forward P/S", fmt_multiple(row.get("forward_ps")))

    classification = row.get("expectations_classification", "Insufficient data")

    card(
        c4,
        "Expectations",
        classification.replace("Forward Expectations ", ""),
        color=status_color(classification),
    )

    if row.get("plain_explanation"):
        st.markdown(
            f'<div style="background:{SURFACE};border:1px solid {BORDER};border-left:4px solid {accent};'
            f'border-radius:8px;padding:12px 16px;margin:12px 0;color:{TEXT};font-size:0.92rem;line-height:1.5;">'
            f'{row["plain_explanation"]}</div>',
            unsafe_allow_html=True,
        )

    median_ps = safe_float(row.get("hist_median_ps"))
    p75_ps = safe_float(row.get("hist_p75_ps"))
    current_ps = safe_float(row.get("current_ps"))
    p90_ps = safe_float(row.get("hist_p90_ps"))
    forward_ps = safe_float(row.get("forward_ps"))

    fig = go.Figure(
        go.Bar(
            x=["Median", "75th pct", "Current", "90th pct", "Forward"],
            y=[median_ps, p75_ps, current_ps, p90_ps, forward_ps],
            marker_color=[TEXT_MUTED, TEXT_MUTED, accent, TEXT_MUTED, "#4FBF7A"],
            text=[
                f"{v:.1f}x" if v is not None else ""
                for v in [median_ps, p75_ps, current_ps, p90_ps, forward_ps]
            ],
            textposition="outside",
        )
    )

    fig.update_layout(
        height=320,
        margin=dict(t=20, b=10),
        plot_bgcolor=SURFACE,
        paper_bgcolor=BG,
        font=dict(color=TEXT, family="IBM Plex Mono"),
        yaxis=dict(title="Price / Sales (x)", gridcolor=BORDER),
        xaxis=dict(gridcolor=BORDER),
    )

    st.plotly_chart(fig, use_container_width=True)

    if show_trend:
        hist = load_history(ticker)

        if len(hist) > 1:
            st.markdown("##### P/S over time")

            line = go.Figure(
                go.Scatter(
                    x=hist["as_of"],
                    y=hist["current_ps"],
                    mode="lines+markers",
                    line=dict(color=accent, width=2),
                    marker=dict(size=5),
                )
            )

            line.update_layout(
                height=260,
                margin=dict(t=10, b=10),
                plot_bgcolor=SURFACE,
                paper_bgcolor=BG,
                font=dict(color=TEXT, family="IBM Plex Mono"),
                yaxis=dict(title="Current P/S (x)", gridcolor=BORDER),
                xaxis=dict(gridcolor=BORDER),
            )

            st.plotly_chart(line, use_container_width=True)

    else:
        st.caption("Trend history isn't tracked for ad-hoc lookups -- only for tickers in the configured watchlist.")

    # -----------------------------------------------------------------
    # Fundamentals overlay
    # -----------------------------------------------------------------
    st.markdown("##### Fundamentals")

    has_any_fundamental = any(
        pd.notna(row.get(f))
        for f in [
            "gross_margin",
            "operating_margin",
            "net_margin",
            "net_debt_to_ebitda",
            "fcf_margin",
            "cash_conversion",
        ]
    )

    if not has_any_fundamental:
        st.caption("No fundamentals data available for this ticker.")
    else:
        f1, f2, f3, f4 = st.columns(4)

        card(f1, "Operating margin", fmt_pct(row.get("operating_margin")))
        card(f2, "Net margin", fmt_pct(row.get("net_margin")))
        card(f3, "Net debt / EBITDA", fmt_multiple(row.get("net_debt_to_ebitda")))
        card(f4, "FCF margin", fmt_pct(row.get("fcf_margin")))

    # -----------------------------------------------------------------
    # Owner earnings overlay
    # -----------------------------------------------------------------
    owner_yield = safe_float(row.get("owner_earnings_yield"))

    if owner_yield is not None:
        if owner_yield >= 0.04:
            owner_yield_color = "#4FBF7A"
        elif owner_yield >= 0.0:
            owner_yield_color = "#E0A63A"
        else:
            owner_yield_color = "#E0584F"
    else:
        owner_yield_color = DEFAULT_ACCENT

    if any(
        pd.notna(row.get(f))
        for f in [
            "owner_earnings_yield",
            "owner_earnings_ttm",
            "maintenance_capex_ttm",
        ]
    ):
        oe1, oe2, oe3 = st.columns(3)

        card(
            oe1,
            "Owner earnings yield",
            fmt_pct(owner_yield),
            color=owner_yield_color,
        )

        card(
            oe2,
            "Owner earnings (TTM)",
            fmt_large_amount(row.get("owner_earnings_ttm")),
        )

        card(
            oe3,
            "Maintenance capex (TTM)",
            fmt_large_amount(row.get("maintenance_capex_ttm")),
        )

        if row.get("owner_earnings_method"):
            st.caption(f"Owner earnings method: {row['owner_earnings_method']}")

    if has_any_fundamental:
        quality = row.get("quality_flag", "Insufficient data")
        qcolor = quality_color(quality)

        st.markdown(
            f'<div style="margin-top:12px;">'
            f'<span class="badge" style="background:{qcolor}22;color:{qcolor};border:1px solid {qcolor}55;">'
            f'{quality}</span></div>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Fundamentals read: flags unprofitability, high leverage, or negative free cash flow."
        )

    # -----------------------------------------------------------------
    # Growth durability
    # -----------------------------------------------------------------
    st.markdown("##### Growth durability")

    if pd.isna(row.get("revenue_cagr_3y")) and pd.isna(row.get("revenue_cagr_5y")):
        st.caption("Not enough annual history available to compute a multi-year revenue CAGR.")
    else:
        d1, d2 = st.columns(2)

        card(d1, "Revenue CAGR (3Y)", fmt_pct(row.get("revenue_cagr_3y")))
        card(d2, "Revenue CAGR (5Y)", fmt_pct(row.get("revenue_cagr_5y")))

        st.caption(
            "Compare this to the forward growth estimate above."
        )

    # -----------------------------------------------------------------
    # Management
    # -----------------------------------------------------------------
    st.markdown("##### Management")

    has_mgmt = any(
        pd.notna(row.get(f))
        for f in [
            "roic",
            "share_count_cagr_3y",
            "buybacks_ttm",
            "dividends_ttm",
            "acquisitions_ttm",
        ]
    )

    if not has_mgmt:
        st.caption("No management/capital-allocation data available for this ticker.")
    else:
        m1, m2, m3 = st.columns(3)

        card(m1, "ROIC", fmt_pct(row.get("roic")))

        dilution = safe_float(row.get("share_count_cagr_3y"))

        if dilution is not None:
            dilution_text = f"{dilution * 100:+.1f}%"
            dilution_color = "#4FBF7A" if dilution < 0 else ("#E0A63A" if dilution < 0.03 else "#E0584F")
        else:
            dilution_text, dilution_color = "n/a", accent

        card(m2, "Share count CAGR (3Y)", dilution_text, color=dilution_color)

        allocation_parts = []

        for label, key in [
            ("Buybacks", "buybacks_ttm"),
            ("Dividends", "dividends_ttm"),
            ("M&A", "acquisitions_ttm"),
        ]:
            val = safe_float(row.get(key))

            if val is not None and val:
                allocation_parts.append(f"{label} ${val / 1e6:,.0f}M")

        card(m3, "Capital deployed (TTM)", " · ".join(allocation_parts) if allocation_parts else "n/a")

        st.caption(
            "Share count CAGR: negative = net buybacks, positive = dilution."
        )

    # -----------------------------------------------------------------
    # Risk / market context
    # -----------------------------------------------------------------
    st.markdown("##### Risk & market context")

    has_risk = pd.notna(row.get("relative_strength_6m")) or pd.notna(row.get("eps_revisions_up_30d"))

    if not has_risk:
        st.caption("No relative-strength or estimate-revision data available for this ticker.")
    else:
        k1, k2, k3 = st.columns(3)

        card(k1, "6M return", fmt_signed_pct(row.get("price_return_6m")))

        bm = row.get("benchmark_symbol", "benchmark")
        rel = safe_float(row.get("relative_strength_6m"))

        rel_color = (
            "#4FBF7A" if rel is not None and rel > 0
            else "#E0584F" if rel is not None and rel < 0
            else accent
        )

        card(k2, f"Vs. {bm} (6M)", fmt_signed_pp(rel), color=rel_color)

        up, down = row.get("eps_revisions_up_30d"), row.get("eps_revisions_down_30d")

        if pd.notna(up) or pd.notna(down):
            card(k3, "EPS revisions (30D)", f"↑{int(up) if pd.notna(up) else 0} / ↓{int(down) if pd.notna(down) else 0}")
        else:
            card(k3, "EPS revisions (30D)", "n/a")

        st.caption(
            "Relative strength compares 6-month price return to a region-appropriate index."
        )

    # -----------------------------------------------------------------
    # Anti-bubble detector
    # -----------------------------------------------------------------
    st.markdown("##### Anti-bubble detector")

    if pd.isna(row.get("multiple_expansion_gap_3y")):
        st.caption(
            "Not enough 3-year market cap and revenue history available to compute multiple expansion."
        )
    else:
        b1, b2, b3, b4 = st.columns(4)

        card(b1, "Market cap CAGR (3Y)", fmt_signed_pct(row.get("market_cap_cagr_3y")))
        card(b2, "Revenue CAGR (3Y)", fmt_signed_pct(row.get("revenue_cagr_3y")))

        gap = safe_float(row.get("multiple_expansion_gap_3y"))

        if gap is not None:
            if gap <= 0.05:
                gap_color = "#4FBF7A"
            elif gap <= 0.20:
                gap_color = "#E0A63A"
            else:
                gap_color = "#E0584F"

            gap_text = f"{gap * 100:+.1f}pp"
        else:
            gap_color = DEFAULT_ACCENT
            gap_text = "n/a"

        card(b3, "Multiple expansion gap", gap_text, color=gap_color)

        bubble_flag = row.get("anti_bubble_flag", "Insufficient data")

        card(
            b4,
            "Bubble read",
            bubble_flag,
            color=anti_bubble_color(bubble_flag),
        )

        if row.get("anti_bubble_note"):
            st.caption(row["anti_bubble_note"])
        else:
            st.caption(
                "Gap = 3-year market cap growth minus 3-year revenue growth."
            )


@st.cache_data(ttl=600, show_spinner=False)
def fetch_adhoc(symbol: str, history_years: int = 5) -> dict:
    """
    Live fetch + compute for a ticker outside the configured watchlist.
    """
    raw = fetch_ticker(symbol, history_years=history_years)

    result = compute_valuation(
        ticker=symbol,
        current_revenue_ttm=raw.current_revenue_ttm,
        current_market_cap=raw.current_market_cap,
        historical_ps_series=raw.historical_ps_series,
        forward_revenue_estimate=raw.forward_revenue_estimate,
        revenue_cadence=raw.revenue_cadence,
        gross_margin=raw.gross_margin,
        operating_margin=raw.operating_margin,
        net_margin=raw.net_margin,
        net_debt_to_ebitda=raw.net_debt_to_ebitda,
        interest_coverage=raw.interest_coverage,
        free_cash_flow_ttm=raw.free_cash_flow_ttm,
        fcf_margin=raw.fcf_margin,
        cash_conversion=raw.cash_conversion,
        revenue_currency=raw.revenue_currency,
        price_currency=raw.price_currency,
        fx_rate_applied=raw.fx_rate_applied,
        currency_note=raw.currency_note,
        revenue_cagr_3y=raw.revenue_cagr_3y,
        revenue_cagr_5y=raw.revenue_cagr_5y,
        roic=raw.roic,
        share_count_cagr_3y=raw.share_count_cagr_3y,
        buybacks_ttm=raw.buybacks_ttm,
        dividends_ttm=raw.dividends_ttm,
        acquisitions_ttm=raw.acquisitions_ttm,
        price_return_6m=raw.price_return_6m,
        benchmark_symbol=raw.benchmark_symbol,
        benchmark_return_6m=raw.benchmark_return_6m,
        relative_strength_6m=raw.relative_strength_6m,
        eps_revisions_up_30d=raw.eps_revisions_up_30d,
        eps_revisions_down_30d=raw.eps_revisions_down_30d,
        owner_earnings_ttm=raw.owner_earnings_ttm,
        owner_earnings_yield=raw.owner_earnings_yield,
        maintenance_capex_ttm=raw.maintenance_capex_ttm,
        owner_earnings_method=raw.owner_earnings_method,
        price_cagr_3y=raw.price_cagr_3y,
        market_cap_cagr_3y=raw.market_cap_cagr_3y,
    )

    return result.to_dict()


df = load_latest()

st.markdown(
    f"""
    <div class="terminal-header">
        <div class="terminal-title">Valuation Terminal</div>
        <div class="terminal-sub">{len(df)} tickers tracked · last refreshed {df['as_of'].max() if not df.empty else '—'}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if df.empty:
    st.info(
        "No watchlist data yet. Run `python -m engine.run` locally, or trigger the "
        "'Refresh valuation data' workflow from the GitHub Actions tab. The Ad-hoc "
        "search tab below works regardless."
    )


# ---------------------------------------------------------------------------
# Summary strip
# ---------------------------------------------------------------------------

if not df.empty:
    if "expectations_classification" in df.columns:
        n_manageable = (df["expectations_classification"] == "Forward Expectations Manageable").sum()
        n_elevated = (df["expectations_classification"] == "Forward Expectations Elevated").sum()
        n_stretched = (df["expectations_classification"] == "Forward Expectations Stretched").sum()
    else:
        n_manageable = n_elevated = n_stretched = 0

    avg_gap = None

    if "growth_gap" in df.columns and df["growth_gap"].notna().any():
        avg_gap = df["growth_gap"].dropna().mean() * 100

    st.markdown(
        f"""
        <div class="summary-row">
            <div class="summary-card"><div class="label">Tickers</div><div class="value">{len(df)}</div></div>
            <div class="summary-card"><div class="label">Avg growth gap</div>
                <div class="value">{f'{avg_gap:+.1f}%' if avg_gap is not None else 'n/a'}</div></div>
            <div class="summary-card"><div class="label" style="color:{STATUS_COLORS['Forward Expectations Manageable']}">Manageable</div>
                <div class="value">{n_manageable}</div></div>
            <div class="summary-card"><div class="label" style="color:{STATUS_COLORS['Forward Expectations Elevated']}">Elevated</div>
                <div class="value">{n_elevated}</div></div>
            <div class="summary-card"><div class="label" style="color:{STATUS_COLORS['Forward Expectations Stretched']}">Stretched</div>
                <div class="value">{n_stretched}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


tab_overview, tab_detail, tab_search = st.tabs(["Watchlist", "Ticker detail", "Ad-hoc search"])


# ---------------------------------------------------------------------------
# Watchlist tab
# ---------------------------------------------------------------------------

def render_watchlist_tab():
    if df.empty:
        st.caption("Nothing to show yet -- run the engine to populate the watchlist.")
        return

    has_sections = "section" in df.columns and df["section"].notna().any()

    if has_sections:
        chips = "".join(
            f'<span class="chip"><span class="dot" style="background:{section_color(s)}"></span>{s}</span>'
            for s in sorted(df["section"].dropna().unique())
        )

        st.markdown(f'<div class="chip-row">{chips}</div>', unsafe_allow_html=True)

        sections = ["All"] + sorted(df["section"].dropna().unique())
        section_filter = st.selectbox("Filter by section", sections, label_visibility="collapsed")

        filtered = df if section_filter == "All" else df[df["section"] == section_filter]
    else:
        filtered = df

    cols = [
        "ticker",
        "current_ps",
        "hist_median_ps",
        "forward_ps",
        "forward_revenue_growth",
        "growth_gap",
        "expectations_classification",
    ]

    if has_sections:
        cols = ["ticker", "section", "layer", "portfolio_weight"] + cols[1:]

    if "quality_flag" in filtered.columns:
        cols.append("quality_flag")

    view = filtered[[c for c in cols if c in filtered.columns]].copy()

    if "forward_revenue_growth" in view.columns:
        view["forward_revenue_growth"] = view["forward_revenue_growth"] * 100

    if "growth_gap" in view.columns:
        view["growth_gap"] = view["growth_gap"] * 100

    if "portfolio_weight" in view.columns:
        view["portfolio_weight"] = view["portfolio_weight"] * 100

    rename_map = {
        "section": "Section",
        "layer": "Layer",
        "portfolio_weight": "Target wt %",
        "current_ps": "Current P/S",
        "hist_median_ps": "Median P/S",
        "forward_ps": "Forward P/S",
        "forward_revenue_growth": "Fwd growth %",
        "growth_gap": "Growth gap %",
        "expectations_classification": "Expectations",
        "quality_flag": "Fundamentals",
    }

    view = view.rename(columns=rename_map)

    sort_options = [c for c in view.columns if c != "ticker"]
    default_sort = "Growth gap %" if "Growth gap %" in sort_options else sort_options[0]

    sort_col = st.selectbox("Sort by", sort_options, index=sort_options.index(default_sort))

    view = view.sort_values(sort_col, ascending=sort_col in ("Ticker",))

    column_config = {
        "Current P/S": st.column_config.NumberColumn(format="%.2fx"),
        "Median P/S": st.column_config.NumberColumn(format="%.2fx"),
        "Forward P/S": st.column_config.NumberColumn(format="%.2fx"),
        "Fwd growth %": st.column_config.NumberColumn(format="%.1f%%"),
        "Growth gap %": st.column_config.NumberColumn(format="%+.1f%%"),
    }

    if "Target wt %" in view.columns:
        max_wt = float(view["Target wt %"].max()) if not view.empty and pd.notna(view["Target wt %"].max()) else 1.0

        column_config["Target wt %"] = st.column_config.ProgressColumn(
            format="%.2f%%",
            min_value=0,
            max_value=max_wt,
        )

    st.dataframe(view, use_container_width=True, hide_index=True, column_config=column_config)


with tab_overview:
    render_watchlist_tab()


# ---------------------------------------------------------------------------
# Ticker detail tab
# ---------------------------------------------------------------------------

with tab_detail:
    if df.empty:
        st.caption("Nothing to show yet -- run the engine to populate the watchlist.")
    else:
        ticker = st.selectbox("Ticker", sorted(df["ticker"].unique()))
        row = df[df["ticker"] == ticker].iloc[0]

        render_ticker_detail(row, show_trend=True)


# ---------------------------------------------------------------------------
# Ad-hoc search tab
# ---------------------------------------------------------------------------

with tab_search:
    st.caption(
        "Look up any ticker outside your configured watchlist -- runs a live fetch "
        "against Yahoo Finance, separate from the tracked data above."
    )

    col_a, col_b = st.columns([4, 1])

    symbol_input = col_a.text_input(
        "Ticker symbol",
        placeholder="e.g. MSFT, RELIANCE.NS, 6758.T, BHP.AX",
        label_visibility="collapsed",
    )

    lookup_clicked = col_b.button("Look up", use_container_width=True)

    if lookup_clicked and symbol_input.strip():
        symbol = symbol_input.strip().upper()

        with st.spinner(f"Fetching live data for {symbol}..."):
            try:
                st.session_state["adhoc_result"] = fetch_adhoc(symbol)
                st.session_state["adhoc_error"] = None
            except Exception as exc:  # noqa: BLE001
                st.session_state["adhoc_result"] = None
                st.session_state["adhoc_error"] = f"Couldn't fetch {symbol}: {exc}"

    if st.session_state.get("adhoc_error"):
        st.error(st.session_state["adhoc_error"])

    elif st.session_state.get("adhoc_result"):
        render_ticker_detail(st.session_state["adhoc_result"], show_trend=False)

    elif not lookup_clicked:
        st.caption("Enter a ticker above and click Look up.")

"""
Streamlit dashboard for the valuation engine.

Reads data/latest.csv (written by engine/run.py, refreshed daily by
GitHub Actions) -- no live network calls happen here, so the dashboard
loads instantly even with 100+ tickers.

The Ad-hoc search tab is the only place that performs a live fetch.

Run locally with:  streamlit run app.py
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


# ---------------------------------------------------------------------------
# Safe formatting helpers -- the dashboard must never crash on NaN/missing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Ticker detail renderer
# ---------------------------------------------------------------------------

def render_ticker_detail(row, show_trend: bool = True):
    """
    Renders the full ticker breakdown.

    Works for both stored rows (pandas Series from data/latest.csv) and
    ad-hoc results (plain dict from ValuationResult.to_dict()).
    All fields are read defensively so missing/NaN values show "n/a".
    """
    ticker = row.get("ticker", "Unknown")

    section_val = row.get("section")
    accent = section_color(section_val) if pd.notna(section_val) else DEFAULT_ACCENT

    header_bits = [str(ticker)]

    if pd.notna(section_val):
        header_bits.append(f"{section_val} · {row.get('layer', '')}")

    st.markdown(
        f'<div style="color:{TEXT_MUTED};font-size:0.85rem;margin-bottom:10px;">{" · ".join(header_bits)}</div>',
        unsafe_allow_html=True,
    )

    # Currency warning first -- it undermines every number below it.
    if (
        row.get("currency_note")
        and pd.isna(row.get("fx_rate_applied"))
        and pd.notna(row.get("revenue_currency"))
    ):
        st.warning(f"⚠ {row['currency_note']}")

    def card(col, label, value, color=None):
        color = color or accent
        col.markdown(
            f'<div class="detail-card" style="--accent:{color}">'
            f'<div class="dc-label">{label}</div>'
            f'<div class="dc-value">{value}</div>'
            '</div>',
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

    classification = row.get("expectations_classification")
    if classification is None or pd.isna(classification):
        classification = "Insufficient data"

    card(
        c4,
        "Expectations",
        str(classification).replace("Forward Expectations ", ""),
        color=status_color(classification),
    )

    cadence = row.get("revenue_data_cadence")
    if cadence is not None and str(cadence).strip().lower() not in ("", "quarterly"):
        st.caption(
            f"Revenue reporting cadence: {cadence} -- TTM figures are approximated "
            f"from the latest reported periods (common for semiannual/annual reporters)."
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
        hist = load_history(str(ticker))

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
        st.caption(
            "Trend history isn't tracked for ad-hoc lookups -- only for tickers in the configured watchlist."
        )

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
        st.caption("No fundamentals data available for this ticker (common for some foreign listings).")
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
        for f in ["owner_earnings_yield", "owner_earnings_ttm", "maintenance_capex_ttm"]
    ):
        oe1, oe2, oe3 = st.columns(3)

        card(oe1, "Owner earnings yield", fmt_pct(owner_yield), color=owner_yield_color)
        card(oe2, "Owner earnings (TTM)", fmt_large_amount(row.get("owner_earnings_ttm")))
        card(oe3, "Maintenance capex (TTM)", fmt_large_amount(row.get("maintenance_capex_ttm")))

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
            "Fundamentals read: flags unprofitability (operating margin < 0), high leverage "
            "(net debt/EBITDA > 3x), or negative free cash flow -- context for whether the "
            "growth story above is backed by a healthy business, not a replacement for it."
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
            "Compare this to the forward growth estimate above -- if the forecast is far above "
            "the historical CAGR, that's a bigger ask than the growth gap alone suggests."
        )

    # -----------------------------------------------------------------
    # Management
    # -----------------------------------------------------------------
    st.markdown("##### Management")

    has_mgmt = any(
        pd.notna(row.get(f))
        for f in ["roic", "share_count_cagr_3y", "buybacks_ttm", "dividends_ttm", "acquisitions_ttm"]
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

        for label, key in [("Buybacks", "buybacks_ttm"), ("Dividends", "dividends_ttm"), ("M&A", "acquisitions_ttm")]:
            val = safe_float(row.get(key))
            if val is not None and val:
                allocation_parts.append(f"{label} {fmt_large_amount(val)}")

        card(m3, "Capital deployed (TTM)", " · ".join(allocation_parts) if allocation_parts else "n/a")

        st.caption(
            "Share count CAGR: negative = net buybacks shrinking the share count, positive = "
            "dilution. Capital deployed is descriptive only -- it shows the split, not a grade "
            "of whether it was spent well."
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

        up = safe_float(row.get("eps_revisions_up_30d"))
        down = safe_float(row.get("eps_revisions_down_30d"))

        if up is not None or down is not None:
            card(
                k3,
                "EPS revisions (30D)",
                f"↑{int(up) if up is not None else 0} / ↓{int(down) if down is not None else 0}",
            )
        else:
            card(k3, "EPS revisions (30D)", "n/a")

        st.caption(
            "Relative strength compares 6-month price return to a region-appropriate index -- "
            "positive means the stock has outperformed its market, not an absolute judgment of "
            "quality. Estimate revision coverage is patchy outside large, well-covered names."
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
        if bubble_flag is None or pd.isna(bubble_flag):
            bubble_flag = "Insufficient data"

        card(b4, "Bubble read", bubble_flag, color=anti_bubble_color(bubble_flag))

        if row.get("anti_bubble_note"):
            st.caption(row["anti_bubble_note"])
        else:
            st.caption(
                "Gap = 3-year market cap growth minus 3-year revenue growth. "
                "Positive values mean multiple expansion."
            )

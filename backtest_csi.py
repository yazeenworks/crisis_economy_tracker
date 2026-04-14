 # -*- coding: utf-8 -*-
# ============================================================
# Validates the Crisis Sentiment Index against 3 real crises
# Run: python backtest_csi.py
# ============================================================

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CRISIS PERIODS TO VALIDATE AGAINST
# ============================================================

CRISIS_PERIODS = {
    'Global Financial Crisis (2008)': {
        'start':       '2008-01-01',
        'end':         '2009-06-30',
        'peak_crisis': '2008-10-10',   # S&P lowest point
        'description': 'Lehman Brothers collapse, global banking meltdown',
        'color':       '#FF4444',
    },
    'COVID-19 Crash (2020)': {
        'start':       '2020-01-01',
        'end':         '2020-12-31',
        'peak_crisis': '2020-03-23',   # S&P lowest point
        'description': 'Pandemic-driven market crash, fastest 30% drop in history',
        'color':       '#FF8C00',
    },
    'SVB & Banking Crisis (2023)': {
        'start':       '2023-01-01',
        'end':         '2023-12-31',
        'peak_crisis': '2023-03-13',   # SVB collapse peak fear
        'description': 'Silicon Valley Bank collapse, regional banking contagion',
        'color':       '#FFD700',
    },
}

# Market tickers for historical data
TICKERS = {
    'SP500':   '^GSPC',
    'VIX':     '^VIX',
    'GOLD':    'GC=F',
    'USD_DXY': 'DX-Y.NYB',
    'TNX':     '^TNX',
    'OIL':     'CL=F',
}

# ============================================================
# STEP 1: FETCH HISTORICAL MARKET DATA
# ============================================================

def fetch_historical_data(start: str, end: str) -> pd.DataFrame:
    """
    Pull daily OHLCV for all tickers across a date range.
    Returns a merged DataFrame with one row per trading day.
    """
    print(f"  Fetching market data: {start} → {end}")
    frames = {}

    for name, ticker in TICKERS.items():
        try:
            t    = yf.Ticker(ticker)
            hist = t.history(start=start, end=end, interval='1d')
            if not hist.empty:
                frames[name] = hist['Close'].rename(name)
                print(f"    ✅ {name}: {len(hist)} trading days")
            else:
                print(f"    ⚠️  {name}: no data returned")
        except Exception as e:
            print(f"    ❌ {name}: {e}")

    if not frames:
        raise ValueError("No market data fetched — check your internet connection.")

    df = pd.concat(frames.values(), axis=1).dropna(how='all')
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


# ============================================================
# STEP 2: CALCULATE CSI FROM HISTORICAL MARKET DATA
# ============================================================

def calculate_historical_csi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct the CSI formula using only market data
    (no live news — we approximate news component from VIX velocity).

    Components:
      A. Market Drawdown (30%)   — SP500 % below rolling 60d high
      B. VIX Level Score (25%)   — Fear gauge normalized
      C. VIX Velocity (20%)      — Proxy for news panic speed
      D. Safe Haven Flow (15%)   — Gold rising + Dollar rising
      E. Bond Stress (10%)       — 10Y yield spike = credit fear
    """
    result = df.copy()

    # A. Market Drawdown — how far SP500 is below its recent peak
    if 'SP500' in result.columns:
        rolling_max           = result['SP500'].rolling(60, min_periods=1).max()
        result['drawdown_pct']= (result['SP500'] - rolling_max) / rolling_max * 100
        result['A_drawdown']  = np.clip(-result['drawdown_pct'] / 30, 0, 1)
    else:
        result['A_drawdown']  = 0

    # B. VIX Level — normalized (15 = calm, 45 = extreme panic)
    if 'VIX' in result.columns:
        result['B_vix_level'] = np.clip((result['VIX'] - 15) / 30, 0, 1)
    else:
        result['B_vix_level'] = 0

    # C. VIX Velocity — rate of change in fear (proxy for news panic)
    if 'VIX' in result.columns:
        result['vix_change_5d']  = result['VIX'].pct_change(5) * 100
        result['C_vix_velocity'] = np.clip(result['vix_change_5d'] / 50, 0, 1)
    else:
        result['C_vix_velocity'] = 0

    # D. Safe Haven Flow — gold + dollar both rising = flight to safety
    if 'GOLD' in result.columns and 'USD_DXY' in result.columns:
        gold_chg              = result['GOLD'].pct_change(5) * 100
        usd_chg               = result['USD_DXY'].pct_change(5) * 100
        result['D_safe_haven']= np.clip((gold_chg + usd_chg) / 10, 0, 1)
    else:
        result['D_safe_haven'] = 0

    # E. Bond Stress — sharp yield spike = credit market fear
    if 'TNX' in result.columns:
        yield_spike           = result['TNX'].pct_change(5) * 100
        result['E_bond_stress']= np.clip(yield_spike.abs() / 20, 0, 1)
    else:
        result['E_bond_stress'] = 0

    # Weighted composite CSI
    result['CSI_raw'] = (
        0.30 * result['A_drawdown']   +
        0.25 * result['B_vix_level']  +
        0.20 * result['C_vix_velocity']+
        0.15 * result['D_safe_haven'] +
        0.10 * result['E_bond_stress']
    )
    result['CSI'] = (result['CSI_raw'] * 100).round(1)
    result['CSI'] = result['CSI'].fillna(0)  # Replace NaN with 0

    # 7-day smoothed CSI (removes day-to-day noise)
    result['CSI_smooth'] = result['CSI'].rolling(7, min_periods=1).mean().round(1)
    result['CSI_smooth'] = result['CSI_smooth'].fillna(result['CSI'])  # Fallback to unsmoothed if still NaN

    # Traffic light label
    def label(score):
        if score < 20:   return 'Calm'
        elif score < 40: return 'Mild Tension'
        elif score < 60: return 'Elevated Stress'
        elif score < 80: return 'Crisis Mode'
        else:            return 'Extreme Crisis'

    result['CSI_label'] = result['CSI_smooth'].apply(label)

    return result


# ============================================================
# STEP 3: SCORE THE CRISIS — DID CSI CATCH IT?
# ============================================================

def score_crisis_detection(
    df:           pd.DataFrame,
    peak_date:    str,
    crisis_name:  str
) -> dict:
    """
    Measures how well CSI detected the crisis:
    - Was CSI elevated BEFORE the market peaked?
    - What was peak CSI vs. baseline CSI?
    - How many days early did the signal appear?
    """
    peak = pd.to_datetime(peak_date)

    # 30-day window before crash peak
    pre_crisis  = df[df.index < peak].tail(30)
    at_crisis   = df[df.index >= peak].head(10)
    baseline    = df.head(20)  # first 20 days = "normal" period

    baseline_csi    = baseline['CSI_smooth'].mean()
    pre_crisis_csi  = pre_crisis['CSI_smooth'].mean() if not pre_crisis.empty else 0
    peak_csi        = at_crisis['CSI_smooth'].max() if not at_crisis.empty else 0

    # How many days before the crash did CSI cross 40 (elevated)?
    elevated_dates = df[df['CSI_smooth'] > 40].index
    early_warning_days = 0
    if not elevated_dates.empty:
        first_elevated = elevated_dates[elevated_dates < peak]
        if not first_elevated.empty:
            early_warning_days = (peak - first_elevated[0]).days

    sp500_drawdown = 0
    if 'SP500' in df.columns and 'drawdown_pct' in df.columns:
        sp500_drawdown = df.loc[df.index >= peak, 'drawdown_pct'].min()

    detection_score = min(100, int(
        (peak_csi / 100 * 50) +
        (min(early_warning_days, 30) / 30 * 30) +
        (min(pre_crisis_csi / baseline_csi, 3) / 3 * 20)
        if baseline_csi > 0 else 0
    ))

    return {
        'crisis':              crisis_name,
        'baseline_csi':        round(baseline_csi, 1),
        'pre_crisis_csi':      round(pre_crisis_csi, 1),
        'peak_csi':            round(peak_csi, 1),
        'early_warning_days':  early_warning_days,
        'sp500_max_drawdown':  round(sp500_drawdown, 1),
        'detection_score':     detection_score,
        'verdict': (
            '✅ Strong Detection'   if detection_score >= 70 else
            '🟡 Partial Detection'  if detection_score >= 40 else
            '❌ Weak Detection'
        )
    }


# ============================================================
# STEP 4: BUILD VALIDATION CHARTS
# ============================================================

def build_validation_chart(results: dict) -> go.Figure:
    """
    Multi-panel chart: CSI vs SP500 vs VIX for each crisis.
    This is your resume screenshot.
    """
    def hex_to_rgba(hex_color, alpha=0.12):
        hex_color = hex_color.lstrip('#')
        # Convert hex to RGB integers
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r}, {g}, {b}, {alpha})"

    crisis_names = list(results.keys())
    n            = len(crisis_names)

    fig = make_subplots(
        rows=n, cols=1,
        subplot_titles=[
            f"{name}  |  Peak CSI: {results[name]['csi_df']['CSI_smooth'].max():.0f}"
            for name in crisis_names
        ],
        vertical_spacing=0.08,
        shared_xaxes=False,
    )

    for i, (name, data) in enumerate(results.items(), start=1):
        df          = data['csi_df']
        crisis_info = CRISIS_PERIODS[name]
        peak_date   = pd.to_datetime(crisis_info['peak_crisis'])
        color       = crisis_info['color']

        # CSI line
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['CSI_smooth'],
            fill='tozeroy',
            fillcolor=hex_to_rgba(color, 0.12) if color.startswith('#') else color,
            name=f'CSI — {name}',
            line=dict(color=color, width=2),
        ), row=i, col=1)

        # VIX overlay (scaled to 0-100 for comparison)
        if 'VIX' in df.columns:
            vix_scaled = (df['VIX'] / df['VIX'].max() * 100)
            fig.add_trace(go.Scatter(
                x=df.index, y=vix_scaled,
                name=f'VIX (scaled) — {name}',
                line=dict(color='#AAAAAA', width=1, dash='dot'),
                opacity=0.6,
            ), row=i, col=1)

        # Threshold lines (only add hlines, skip vlines due to plotly timestamp issues)
        for threshold, label, tcolor in [
            (40, 'Elevated', '#FFD700'),
            (60, 'Crisis',   '#FF4444'),
        ]:
            fig.add_hline(
                y=threshold, line_dash='dot',
                line_color=tcolor, line_width=1,
                row=i, col=1,
                annotation_text=label,
                annotation_font_color=tcolor,
                annotation_font_size=9,
                annotation_position='right',
            )

    fig.update_layout(
        title=dict(
            text='Crisis Sentiment Index — Backtested Against 3 Real Crises',
            font=dict(size=18, color='white'),
        ),
        template    = 'plotly_dark',
        height      = 350 * n,
        showlegend  = False,
        paper_bgcolor = '#0e1117',
        plot_bgcolor  = '#0e1117',
    )
    fig.update_yaxes(title_text='CSI Score (0–100)', range=[0, 105])

    return fig


# ============================================================
# STEP 5: VALIDATION REPORT
# ============================================================

def print_validation_report(scores: list):
    print("\n" + "="*65)
    print("      CSI BACKTESTING VALIDATION REPORT")
    print("="*65)

    total_score = 0
    for s in scores:
        print(f"\n  📌 {s['crisis']}")
        print(f"     Baseline CSI (normal period) : {s['baseline_csi']}")
        print(f"     Pre-crisis CSI (30d before)  : {s['pre_crisis_csi']}")
        print(f"     Peak CSI (at crash)           : {s['peak_csi']}")
        print(f"     Early warning lead time       : {s['early_warning_days']} days before peak")
        print(f"     SP500 max drawdown            : {s['sp500_max_drawdown']}%")
        print(f"     Detection Score               : {s['detection_score']}/100")
        print(f"     Verdict                       : {s['verdict']}")
        total_score += s['detection_score']

    avg = total_score / len(scores)
    print("\n" + "-"*65)
    print(f"  OVERALL CSI ACCURACY SCORE : {avg:.0f}/100")
    print(f"  OVERALL VERDICT            : ", end='')
    if avg >= 70:
        print("✅ CSI is a RELIABLE crisis indicator")
    elif avg >= 40:
        print("🟡 CSI shows PARTIAL crisis detection ability")
    else:
        print("❌ CSI needs recalibration")
    print("="*65)
    print("\n  📄 Chart saved to: csi_backtest_chart.html")
    print("  📄 Data saved to : csi_backtest_results.csv")


# ============================================================
# MAIN
# ============================================================

def run_backtest():
    print("\n🔍 CSI BACKTESTING ENGINE — Starting...\n")

    all_results = {}
    all_scores  = []

    for crisis_name, crisis_info in CRISIS_PERIODS.items():
        print(f"\n{'='*55}")
        print(f"  📌 Running: {crisis_name}")
        print(f"  {crisis_info['description']}")
        print(f"{'='*55}")

        # Fetch data
        df = fetch_historical_data(crisis_info['start'], crisis_info['end'])

        # Calculate CSI
        df = calculate_historical_csi(df)

        # Score detection
        score = score_crisis_detection(df, crisis_info['peak_crisis'], crisis_name)
        all_scores.append(score)

        all_results[crisis_name] = {
            'csi_df':      df,
            'score':       score,
            'crisis_info': crisis_info,
        }

        print(f"  ✅ Done — Peak CSI: {df['CSI_smooth'].max():.1f} | "
              f"Early warning: {score['early_warning_days']} days")

    # Build chart
    print("\n📊 Building validation chart...")
    fig = build_validation_chart(all_results)
    fig.write_html('csi_backtest_chart.html')
    print("  ✅ Chart saved: csi_backtest_chart.html  (open in browser)")

    # Save data
    all_dfs = []
    for name, data in all_results.items():
        df_copy              = data['csi_df'].copy()
        df_copy['crisis']    = name
        all_dfs.append(df_copy[['SP500','VIX','CSI','CSI_smooth','CSI_label','crisis']])
    combined = pd.concat(all_dfs)
    combined.to_csv('csi_backtest_results.csv')
    print("  ✅ Data saved : csi_backtest_results.csv")

    # Print report
    print_validation_report(all_scores)

    return all_results, all_scores


if __name__ == '__main__':
    run_backtest()
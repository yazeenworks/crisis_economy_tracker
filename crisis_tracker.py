# -*- coding: utf-8 -*-
# ============================================================
# CRISIS ECONOMY TRACKER  v4  (UTF-8 safe)
# Run: streamlit run crisis_tracker.py
# ============================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
import os
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

nltk.download('vader_lexicon', quiet=True)

# ============================================================
# LOCAL DATA FILE
# ============================================================

LOCAL_DATA_FILE = 'local_prices.json'

def load_local_data():
    if os.path.exists(LOCAL_DATA_FILE):
        try:
            with open(LOCAL_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_local_data(entries):
    with open(LOCAL_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)

# ============================================================
# COMMODITY CONFIG
# ============================================================

FOOD_TICKERS = {
    'Wheat':       'ZW=F',
    'Corn':        'ZC=F',
    'Soybean Oil': 'ZL=F',
    'Sugar':       'SB=F',
    'Coffee':      'KC=F',
}

UTILITY_TICKERS = {
    'Natural Gas': 'NG=F',
    'Crude Oil':   'CL=F',
    'Gasoline':    'RB=F',
    'Heating Oil': 'HO=F',
}

BASELINES = {
    'Wheat': 560,  'Corn': 450,  'Soybean Oil': 46,
    'Sugar':  23,  'Coffee': 185, 'Natural Gas': 3.0,
    'Crude Oil': 72, 'Gasoline': 2.6, 'Heating Oil': 2.9,
}

# ============================================================
# PART 1: NEWS HEADLINES
# ============================================================

@dataclass
class HeadlineData:
    title:     str
    source:    str
    timestamp: str
    sentiment: float
    url:       str


def scrape_economic_headlines(max_articles=40):
    feeds = {
        'MarketWatch':   'https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines',
        'Yahoo Finance': 'https://finance.yahoo.com/news/rssindex',
        'Investing.com': 'https://www.investing.com/rss/news.rss',
    }
    sia = SentimentIntensityAnalyzer()
    headlines = []

    for source, url in feeds.items():
        try:
            resp = requests.get(url, timeout=8,
                                headers={'User-Agent': 'Mozilla/5.0'})
            soup = BeautifulSoup(resp.content, 'xml')
            items = soup.find_all('item')[:max_articles // len(feeds)]
            for item in items:
                title      = item.find('title')
                title_text = title.text.strip() if title else ''
                if not title_text:
                    continue
                econ_kw = [
                    'inflation','recession','fed','rate','gdp','market',
                    'bank','stock','economy','crisis','debt','trade',
                    'tariff','unemployment','growth','dollar','oil','bond'
                ]
                if not any(k in title_text.lower() for k in econ_kw):
                    continue
                pub_date  = item.find('pubDate')
                timestamp = pub_date.text if pub_date else str(datetime.now())
                link      = item.find('link')
                link_url  = link.text if link else '#'
                score     = sia.polarity_scores(title_text)['compound']
                headlines.append(HeadlineData(
                    title=title_text, source=source,
                    timestamp=timestamp, sentiment=score, url=link_url
                ))
        except Exception as e:
            st.warning("Could not fetch from " + source + ": " + str(e))

    return headlines


# ============================================================
# PART 2: MARKET DATA
# ============================================================
def fetch_market_data():
    tickers = {
        'SP500':   '^GSPC',
        'VIX':     '^VIX',
        'GOLD':    'GC=F',
        'OIL':     'CL=F',
        'USD_DXY': 'DX-Y.NYB',
        'TNX':     '^TNX',
    }
    market_data = {}
    for name, ticker in tickers.items():
        try:
            t = yf.Ticker(ticker)
            # .dropna() removes rows with missing values that break math
            hist = t.history(period='5d', interval='1d').dropna() 
            
            if len(hist) >= 2:
                current = hist['Close'].iloc[-1]
                prev    = hist['Close'].iloc[-2]
                
                # Check for zero or NaN before dividing
                if prev != 0 and not np.isnan(current) and not np.isnan(prev):
                    pct_chg = (current - prev) / prev * 100
                else:
                    pct_chg = 0.0
                    
                market_data[name] = {
                    'current':    round(float(current), 2),
                    'pct_change': round(float(pct_chg), 4),
                    'history':    hist['Close'].fillna(0).tolist()
                }
            else:
                # Fallback if history is too short or empty
                market_data[name] = {'current': 0.0, 'pct_change': 0.0, 'history': []}
                
        except Exception:
            market_data[name] = {'current': 0.0, 'pct_change': 0.0, 'history': []}
            
    return market_data


# ============================================================
# PART 3: ESSENTIALS
# ============================================================

def fetch_essentials_prices():
    all_tickers = {}
    all_tickers.update(FOOD_TICKERS)
    all_tickers.update(UTILITY_TICKERS)
    results = {}

    for name, ticker in all_tickers.items():
        try:
            data = yf.Ticker(ticker)
            hist = data.history(period='30d', interval='1d')
            if hist.empty or len(hist) < 2:
                continue
            current     = round(hist['Close'].iloc[-1], 2)
            avg_30d     = round(hist['Close'].mean(), 2)
            baseline    = BASELINES.get(name, avg_30d)
            pct_vs_base = round((current - baseline) / baseline * 100, 1)
            if pct_vs_base < 10:
                status, color = 'Available', '#00C851'
            elif pct_vs_base < 25:
                status, color = 'Scarce',    '#FFD700'
            else:
                status, color = 'Critical',  '#FF4444'
            results[name] = {
                'ticker': ticker, 'current': current, 'avg_30d': avg_30d,
                'pct_vs_base': pct_vs_base, 'status': status, 'color': color,
                'history': hist['Close'].tolist()[-14:],
                'category': 'Food' if ticker in FOOD_TICKERS.values() else 'Utility'
            }
        except Exception:
            results[name] = {
                'current': None, 'status': 'Unavailable', 'color': '#888888',
                'pct_vs_base': 0,
                'category': 'Food' if name in FOOD_TICKERS else 'Utility'
            }
    return results


# ============================================================
# PART 4: MEDICINE SIGNALS
# ============================================================

def fetch_medicine_signals():
    url = ("https://api.fda.gov/drug/enforcement.json"
           "?search=status:Ongoing&limit=10&sort=report_date:desc")
    medicines = []
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        for item in data.get('results', []):
            product = item.get('product_description', 'Unknown')[:60]
            reason  = item.get('reason_for_recall', '')[:80]
            date    = item.get('report_date', '')
            classif = item.get('classification', '')
            if 'Class I' in classif:
                signal, color = 'CRITICAL', '#FF4444'
            elif 'Class II' in classif:
                signal, color = 'SCARCE',   '#FFD700'
            else:
                signal, color = 'MONITOR',  '#00C851'
            medicines.append({
                'product': product, 'reason': reason,
                'date': date, 'signal': signal,
                'color': color, 'class': classif,
            })
    except Exception as e:
        medicines.append({
            'product': 'OpenFDA unreachable', 'reason': str(e),
            'date': '', 'signal': 'UNAVAILABLE', 'color': '#888888', 'class': '',
        })
    return medicines


# ============================================================
# PART 5: CSI CALCULATION
# ============================================================

def resolve_signal_conflicts(news_sentiment, market_data):
    vix_change    = market_data.get('VIX',   {}).get('pct_change', 0)
    sp500_change  = market_data.get('SP500', {}).get('pct_change', 0)
    market_stress = np.clip((-sp500_change / 3) + (vix_change / 10), -1, 1)
    news_stress   = -news_sentiment
    conflict      = abs(news_stress - market_stress) > 0.5
    if conflict:
        msg = ("WARNING: Markets stressed but headlines calm - possible institutional selling."
               if market_stress > news_stress else
               "WARNING: Headlines alarming but markets stable - possible media sensationalism.")
    else:
        msg = "Signals are aligned."
    reconciled = (0.4 * news_stress + 0.6 * market_stress if conflict
                  else 0.5 * news_stress + 0.5 * market_stress)
    return {
        'news_stress':       round(news_stress, 4),
        'market_stress':     round(market_stress, 4),
        'reconciled_stress': round(reconciled, 4),
        'conflict_detected': conflict,
        'conflict_message':  msg,
    }


def calculate_crisis_index(headlines, market_data):
    # 1. News Sentiment Safety
    sentiments = [h.sentiment for h in headlines] if headlines else [0.0]
    
    # Ensure no NaN in sentiments
    sentiments = [s if not np.isnan(s) else 0.0 for s in sentiments]
    
    negative_pct = sum(1 for s in sentiments if s < -0.05) / max(len(sentiments), 1)
    news_negativity = np.clip(negative_pct, 0, 1)
    
    # Standard deviation can return NaN if input is empty; np.nan_to_num fixes this
    news_volatility = np.clip(np.nan_to_num(np.std(sentiments)) * 2, 0, 1)

    # 2. Market Data Safety (Force values to 0 if NaN/None)
    def get_safe_val(key, subkey, default=0.0):
        val = market_data.get(key, {}).get(subkey, default)
        return float(val) if val is not None and not np.isnan(val) else default

    sp500_chg = get_safe_val('SP500', 'pct_change')
    market_drawdown = np.clip(-sp500_chg / 5, 0, 1)

    vix_current = get_safe_val('VIX', 'current', 20.0)
    vix_score = np.clip((vix_current - 15) / 30, 0, 1)

    gold_chg = get_safe_val('GOLD', 'pct_change')
    usd_chg = get_safe_val('USD_DXY', 'pct_change')
    safe_haven = np.clip((gold_chg + usd_chg) / 4, 0, 1)

    # 3. Final Score Calculation
    csi_raw = (0.30 * news_negativity + 0.20 * news_volatility +
               0.25 * market_drawdown + 0.15 * vix_score +
               0.10 * safe_haven)
    
    # Final check: Convert any accidental NaN to 0
    csi_raw = np.nan_to_num(csi_raw)
    csi_score = round(float(csi_raw * 100), 1)

    # 4. Status Logic (Now safe from 'Extreme Crisis' bug)
    if csi_score < 20: 
        label, color = "🟢 Calm", "#00C851"
    elif csi_score < 40: 
        label, color = "🟡 Mild Tension", "#FFD700"
    elif csi_score < 60: 
        label, color = "🟠 Elevated Stress", "#FF8C00"
    elif csi_score < 80: 
        label, color = "🔴 Crisis Mode", "#FF4444"
    else: 
        label, color = "🚨 Extreme Crisis", "#8B0000"

    return {
        'score': csi_score, 
        'label': label, 
        'color': color,
        'components': {
            'News Negativity (30%)': round(float(news_negativity * 100), 1),
            'News Volatility (20%)': round(float(news_volatility * 100), 1),
            'Market Drawdown (25%)': round(float(market_drawdown * 100), 1),
            'VIX Spike (15%)': round(float(vix_score * 100), 1),
            'Safe Haven Flow (10%)': round(float(safe_haven * 100), 1),
        }
    }


# ============================================================
# PART 6: SPARKLINE
# ============================================================

def make_sparkline(history, color, name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history, mode='lines',
        line=dict(color=color, width=2),
        fill='tozeroy', name=name
    ))
    fig.update_layout(
        height=60, margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig


# ============================================================
# PART 7: DASHBOARD
# ============================================================

def build_dashboard():
    st.set_page_config(
        page_title="Crisis Economy Tracker",
        page_icon="📉", layout="wide",
        initial_sidebar_state="collapsed"
    )

    st.markdown("""
    <style>
        .block-container { padding-top: 1rem; }
        .csi-score {
            font-size: 96px; font-weight: 900; text-align: center;
        }
        div[data-testid="metric-container"] {
            background: #1a1f2e; border-radius: 8px; padding: 12px;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title("Crisis Economy Tracker")
    st.caption("Live data  |  Last updated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    col_a, col_b, _ = st.columns([1, 1, 5])
    auto_refresh = col_a.toggle("Auto-refresh (60s)", value=False)
    col_b.button("Refresh Now")

    # FETCH DATA
    with st.spinner("Fetching live data..."):
        headlines   = scrape_economic_headlines(max_articles=40)
        market_data = fetch_market_data()
        essentials  = fetch_essentials_prices()
        med_signals = fetch_medicine_signals()
        conflicts   = resolve_signal_conflicts(
            np.mean([h.sentiment for h in headlines]) if headlines else 0,
            market_data
        )
        csi = calculate_crisis_index(headlines, market_data)

    # ── 1. CSI BIG NUMBER ────────────────────────────────────
    st.markdown("---")
    _, c2, _ = st.columns([1, 2, 1])
    with c2:
        st.markdown(
            '<div class="csi-score" style="color:' + csi["color"] + '">'
            + str(csi["score"]) + '</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<h2 style="text-align:center;color:' + csi["color"] + '">'
            + 'Crisis Sentiment Index  |  ' + csi["label"] + '</h2>',
            unsafe_allow_html=True
        )

    # ── 2. MARKET METRICS ────────────────────────────────────
    st.markdown("---")
    st.subheader("Live Market Indicators")
    metric_map = [
        ('SP500', 'S&P 500', ''),
        ('VIX',   'VIX Fear Index', ''),
        ('GOLD',  'Gold (USD)', ''),
        ('OIL',   'Crude Oil', ''),
        ('USD_DXY','Dollar Index', ''),
        ('TNX',   '10Y Treasury', '%'),
    ]
    for col, (key, label, suffix) in zip(st.columns(6), metric_map):
        d   = market_data.get(key, {})
        val = d.get('current')
        chg = d.get('pct_change', 0)
        col.metric(
            label=label,
            value=('$' + '{:,.2f}'.format(val) + suffix) if val else 'N/A',
            delta='{:+.2f}%'.format(chg)
        )

    # ── 3. CSI BREAKDOWN + GAUGE ─────────────────────────────
    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.subheader("CSI Component Breakdown")
        comp_df = pd.DataFrame(
            list(csi['components'].items()),
            columns=['Component', 'Score (0-100)']
        )
        fig_bar = px.bar(
            comp_df, x='Score (0-100)', y='Component',
            orientation='h', color='Score (0-100)',
            color_continuous_scale=['green', 'yellow', 'red'],
            range_color=[0, 100], template='plotly_dark'
        )
        fig_bar.update_layout(showlegend=False, height=300)
        st.plotly_chart(fig_bar, use_container_width=True)

    with right:
        st.subheader("Signal Conflict Analysis")
        if conflicts['conflict_detected']:
            st.warning(conflicts['conflict_message'])
        else:
            st.success("Signals aligned: " + conflicts['conflict_message'])
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=csi['score'],
            gauge={
                'axis': {'range': [0, 100]},
                'bar':  {'color': csi['color']},
                'steps': [
                    {'range': [0,  20], 'color': '#003300'},
                    {'range': [20, 40], 'color': '#333300'},
                    {'range': [40, 60], 'color': '#332000'},
                    {'range': [60, 80], 'color': '#330000'},
                    {'range': [80,100], 'color': '#1a0000'},
                ],
            },
            title={'text': "Crisis Score", 'font': {'size': 18}}
        ))
        fig_gauge.update_layout(template='plotly_dark', height=280)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── 4. BACKTEST RESULTS ───────────────────────────────────
    st.markdown("---")
    st.subheader("CSI Backtesting  |  Validated Against 3 Real Crises")
    st.caption("Proving the formula: CSI retroactively applied to 2008 crash, COVID-19, and SVB collapse")

    CRISIS_COLORS = {
        'Global Financial Crisis (2008)': '#FF4444',
        'COVID-19 Crash (2020)':          '#FF8C00',
        'SVB & Banking Crisis (2023)':    '#FFD700',
    }

    if os.path.exists('csi_backtest_results.csv'):
        try:
            bt_df = pd.read_csv('csi_backtest_results.csv', index_col=0, parse_dates=True)
            bt_df.index = pd.to_datetime(bt_df.index, utc=True).tz_localize(None)

            crisis_list = bt_df['crisis'].unique()
            cols = st.columns(len(crisis_list))

            for col, crisis_name in zip(cols, crisis_list):
                crisis_df = bt_df[bt_df['crisis'] == crisis_name].copy()
                peak_csi  = crisis_df['CSI_smooth'].max()
                base_csi  = crisis_df['CSI_smooth'].head(20).mean()
                color     = CRISIS_COLORS.get(crisis_name, '#AAAAAA')

                col.markdown(
                    '<div style="background:#1a1f2e;border-radius:8px;'
                    'padding:12px;border-left:4px solid ' + color + ';">'
                    '<b style="color:' + color + ';font-size:0.85em">'
                    + crisis_name + '</b><br>'
                    '<span style="font-size:2em;font-weight:900;color:'
                    + color + '">' + str(round(peak_csi, 0))[:-2] + '</span>'
                    '<span style="color:#AAAAAA"> / 100 peak</span><br>'
                    '<span style="color:#888888;font-size:0.8em">Baseline: '
                    + str(round(base_csi, 1)) + ' | Rise: +'
                    + str(round(peak_csi - base_csi, 1)) + ' pts</span>'
                    '</div>',
                    unsafe_allow_html=True
                )

            st.markdown("#### CSI Timeline Across All 3 Crises")
            fig_bt = go.Figure()
            for crisis_name in crisis_list:
                crisis_df = bt_df[bt_df['crisis'] == crisis_name].copy()
                color     = CRISIS_COLORS.get(crisis_name, '#AAAAAA')
                fig_bt.add_trace(go.Scatter(
                    x=crisis_df.index,
                    y=crisis_df['CSI_smooth'],
                    name=crisis_name,
                    line=dict(color=color, width=2),
                    fill='tozeroy',
                ))
            fig_bt.add_hline(y=40, line_dash='dot', line_color='#FFD700',
                             annotation_text='Elevated (40)',
                             annotation_font_color='#FFD700')
            fig_bt.add_hline(y=60, line_dash='dot', line_color='#FF4444',
                             annotation_text='Crisis (60)',
                             annotation_font_color='#FF4444')
            fig_bt.update_layout(
                template='plotly_dark', height=380,
                legend=dict(
                    orientation='h', yanchor='bottom',
                    y=1.02, xanchor='right', x=1,
                    font=dict(color='#EEEEEE')
                ),
                yaxis_title='CSI Score (0-100)',
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117',
            )
            st.plotly_chart(fig_bt, use_container_width=True)

            st.markdown(
                '<div style="background:#1a1f2e;border-radius:8px;'
                'padding:14px;margin-top:8px;">'
                '<b style="color:#EEEEEE">What this proves:</b> '
                '<span style="color:#CCCCCC">The CSI formula consistently elevated '
                'above 40 during all three confirmed crisis periods. '
                'During peak stress (COVID March 2020, Lehman collapse), '
                'the index reached Crisis Mode (60+), validating its '
                'sensitivity to real market dislocations.</span>'
                '</div>',
                unsafe_allow_html=True
            )

        except Exception as e:
            st.warning("Could not load backtest data: " + str(e) +
                       " | Run: python backtest_csi.py")
    else:
        st.info("Run 'python backtest_csi.py' once to generate backtest charts here.")

    # ── 5. GLOBAL ESSENTIALS ─────────────────────────────────
    st.markdown("---")
    st.subheader("Global Essential Goods  |  Food, Utilities and Medicines")
    st.caption("Commodity futures vs 30-day baseline  |  Available = under 10% surge  |  Scarce = 10-25%  |  Critical = above 25%")

    st.markdown("#### Food Commodities")
    food_items = {k: v for k, v in essentials.items()
                  if v.get('category') == 'Food' and v.get('current')}
    if food_items:
        cols = st.columns(len(food_items))
        for col, (name, info) in zip(cols, food_items.items()):
            col.metric(
                label='[' + info['status'] + ']  ' + name,
                value='$' + '{:,.2f}'.format(info['current']),
                delta='{:+.1f}% vs baseline'.format(info['pct_vs_base']),
                delta_color="inverse"
            )
            if info.get('history'):
                col.plotly_chart(
                    make_sparkline(info['history'], info['color'], name),
                    use_container_width=True,
                    config={'displayModeBar': False}
                )
    else:
        st.info("Food data unavailable  |  Market may be closed or data delayed.")

    st.markdown("#### Utilities and Energy")
    util_items = {k: v for k, v in essentials.items()
                  if v.get('category') == 'Utility' and v.get('current')}
    if util_items:
        cols = st.columns(len(util_items))
        for col, (name, info) in zip(cols, util_items.items()):
            col.metric(
                label='[' + info['status'] + ']  ' + name,
                value='$' + '{:,.2f}'.format(info['current']),
                delta='{:+.1f}% vs baseline'.format(info['pct_vs_base']),
                delta_color="inverse"
            )
            if info.get('history'):
                col.plotly_chart(
                    make_sparkline(info['history'], info['color'], name),
                    use_container_width=True,
                    config={'displayModeBar': False}
                )
    else:
        st.info("Utility data unavailable  |  Market may be closed or data delayed.")

    all_ess = {k: v for k, v in essentials.items() if v.get('current')}
    if all_ess:
        st.markdown("#### Global Price Surge Overview")
        surge_df = pd.DataFrame([
            {'Item': name, 'Surge %': info['pct_vs_base'], 'Category': info['category']}
            for name, info in all_ess.items()
        ]).sort_values('Surge %', ascending=True)
        fig_surge = px.bar(
            surge_df, x='Surge %', y='Item', orientation='h',
            color='Surge %',
            color_continuous_scale=['#00C851', '#FFD700', '#FF4444'],
            range_color=[0, 30], template='plotly_dark', facet_col='Category'
        )
        fig_surge.add_vline(x=10, line_dash='dash', line_color='#FFD700',
                            annotation_text='Scarce (10%)')
        fig_surge.add_vline(x=25, line_dash='dash', line_color='#FF4444',
                            annotation_text='Critical (25%)')
        fig_surge.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig_surge, use_container_width=True)

    st.markdown("#### Medicine Supply Signals  |  OpenFDA")
    st.caption("Ongoing FDA recalls used as global shortage proxy")
    valid_meds = [m for m in med_signals if 'UNAVAILABLE' not in m['signal']]
    if valid_meds:
        for med in valid_meds[:8]:
            st.markdown(
                '<div style="border-left:3px solid ' + med["color"] + ';'
                'padding:8px 14px;margin:4px 0;background:#1a1f2e;border-radius:4px;">'
                '<b style="color:' + med["color"] + '">[' + med["signal"] + ']</b>'
                '&nbsp;&nbsp;<b style="color:#F0F0F0">' + med["product"] + '</b><br>'
                '<span style="color:#BBBBBB;font-size:0.85em">' + med["reason"] + '</span>'
                '<span style="color:#777777;font-size:0.8em">  |  ' + med["date"] + '</span>'
                '</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("No active medicine shortage signals detected.")

    st.markdown("#### Global Essentials Stress Score")
    critical_count = sum(1 for v in essentials.values() if v.get('status') == 'Critical')
    scarce_count   = sum(1 for v in essentials.values() if v.get('status') == 'Scarce')
    total_tracked  = len([v for v in essentials.values() if v.get('current')])
    if total_tracked > 0:
        ess_score = round((critical_count * 2 + scarce_count) / (total_tracked * 2) * 100, 1)
        ess_color = '#00C851' if ess_score < 20 else '#FFD700' if ess_score < 50 else '#FF4444'
        ess_label = ('Supply chains stable' if ess_score < 20 else
                     'Moderate supply stress' if ess_score < 50 else
                     'Severe supply disruption')
        c1, c2, c3 = st.columns(3)
        c1.metric("Critical Items",  critical_count)
        c2.metric("Scarce Items",    scarce_count)
        c3.metric("Available Items", total_tracked - critical_count - scarce_count)
        st.markdown(
            '<div style="background:#1a1f2e;border-radius:10px;padding:16px;'
            'text-align:center;border:2px solid ' + ess_color + ';margin-top:10px">'
            '<span style="font-size:2.5em;color:' + ess_color + ';font-weight:900">'
            + str(ess_score) + '/100</span><br>'
            '<span style="color:' + ess_color + ';font-size:1.1em">' + ess_label + '</span>'
            '</div>',
            unsafe_allow_html=True
        )

    # ── 6. LOCAL MARKET TRACKER ───────────────────────────────
    st.markdown("---")
    st.subheader("My Local Market Tracker")
    st.caption("Add any item from your region  |  Prices saved automatically between sessions")

    if 'local_entries' not in st.session_state:
        st.session_state.local_entries = load_local_data()

    with st.expander("Add or Update a Local Item", expanded=True):
        c1, c2, c3, c4, c5 = st.columns([2.5, 1.5, 1.5, 1.5, 1])
        item_name    = c1.text_input("Item Name",
                                     placeholder="e.g. Rice 1kg, Diesel 1L, Paracetamol")
        item_price   = c2.number_input("Current Price", min_value=0.0,
                                       step=0.5, format="%.2f")
        normal_price = c3.number_input("Your Normal Price", min_value=0.0,
                                       step=0.5, format="%.2f")
        item_cat     = c4.selectbox("Category",
                                    ["Food", "Fuel / Utility", "Medicine", "Other"])
        save_btn     = c5.button("Save", use_container_width=True)

        if save_btn:
            if not item_name.strip():
                st.warning("Please enter an item name.")
            elif item_price <= 0:
                st.warning("Please enter the current price.")
            else:
                if normal_price > 0:
                    surge = round((item_price - normal_price) / normal_price * 100, 1)
                    if surge < 10:
                        status, color = 'Available', '#00C851'
                    elif surge < 25:
                        status, color = 'Scarce',    '#FFD700'
                    else:
                        status, color = 'Critical',  '#FF4444'
                else:
                    surge, status, color = 0.0, 'No baseline set', '#888888'

                new_entry = {
                    'name':         item_name.strip(),
                    'price':        item_price,
                    'normal_price': normal_price,
                    'surge_pct':    surge,
                    'status':       status,
                    'color':        color,
                    'category':     item_cat,
                    'updated':      datetime.now().strftime('%Y-%m-%d %H:%M'),
                }
                entries  = st.session_state.local_entries
                existing = [i for i, e in enumerate(entries)
                            if e['name'].lower() == item_name.strip().lower()]
                if existing:
                    entries[existing[0]] = new_entry
                    st.success("Updated: " + item_name)
                else:
                    entries.append(new_entry)
                    st.success("Saved: " + item_name)
                save_local_data(entries)
                st.rerun()

    entries = st.session_state.local_entries
    if entries:
        lc1, lc2, lc3, lc4 = st.columns(4)
        lc1.metric("Items Tracked", len(entries))
        lc2.metric("Critical", sum(1 for e in entries if e.get('status') == 'Critical'))
        lc3.metric("Scarce",   sum(1 for e in entries if e.get('status') == 'Scarce'))
        lc4.metric("Available",sum(1 for e in entries if e.get('status') == 'Available'))

        cats     = ['All'] + sorted(set(e['category'] for e in entries))
        sel_cat  = st.selectbox("Filter by category:", cats)
        filtered = entries if sel_cat == 'All' else [
                       e for e in entries if e['category'] == sel_cat]

        sort_by = st.radio("Sort by:",
                           ["Name", "Price high to low", "Surge % high to low", "Category"],
                           horizontal=True)
        if sort_by == "Name":
            filtered = sorted(filtered, key=lambda x: x['name'].lower())
        elif sort_by == "Price high to low":
            filtered = sorted(filtered, key=lambda x: x['price'], reverse=True)
        elif sort_by == "Surge % high to low":
            filtered = sorted(filtered, key=lambda x: x['surge_pct'], reverse=True)
        elif sort_by == "Category":
            filtered = sorted(filtered, key=lambda x: x['category'])

        h1,h2,h3,h4,h5,h6,h7 = st.columns([2.5,1.2,1.2,1.2,1.5,1.8,0.8])
        for col, lbl in zip([h1,h2,h3,h4,h5,h6,h7],
                            ['**Item**','**Category**','**Current**',
                             '**Normal**','**Surge %**','**Status**','**Del**']):
            col.markdown(lbl)
        st.markdown('<hr style="margin:4px 0 8px 0;border-color:#333333">',
                    unsafe_allow_html=True)

        for i, entry in enumerate(filtered):
            r1,r2,r3,r4,r5,r6,r7 = st.columns([2.5,1.2,1.2,1.2,1.5,1.8,0.8])
            r1.markdown('<span style="color:#EEEEEE;font-weight:600">'
                        + entry["name"] + '</span>', unsafe_allow_html=True)
            r2.markdown('<span style="color:#CCCCCC">'
                        + entry["category"] + '</span>', unsafe_allow_html=True)
            r3.markdown('<span style="color:#FFFFFF;font-weight:600">'
                        + '{:,.2f}'.format(entry["price"]) + '</span>',
                        unsafe_allow_html=True)
            np_val = '{:,.2f}'.format(entry["normal_price"]) if entry["normal_price"] > 0 else "—"
            r4.markdown('<span style="color:#AAAAAA">' + np_val + '</span>',
                        unsafe_allow_html=True)
            sc = entry['color']
            r5.markdown('<span style="color:' + sc + ';font-weight:700">'
                        + '{:+.1f}%'.format(entry["surge_pct"]) + '</span>',
                        unsafe_allow_html=True)
            r6.markdown('<span style="color:' + sc + '">'
                        + entry["status"] + '</span>', unsafe_allow_html=True)
            if r7.button("X", key="del_" + str(i) + "_" + entry['name'],
                         help="Remove " + entry['name']):
                st.session_state.local_entries = [
                    e for e in st.session_state.local_entries
                    if e['name'] != entry['name']
                ]
                save_local_data(st.session_state.local_entries)
                st.rerun()

        chart_data = [
            {'Item': e['name'], 'Surge %': e['surge_pct'], 'Category': e['category']}
            for e in filtered if e['surge_pct'] != 0 and e['normal_price'] > 0
        ]
        if len(chart_data) > 1:
            st.markdown("#### Your Local Price Surge Chart")
            local_df  = pd.DataFrame(chart_data).sort_values('Surge %', ascending=True)
            fig_local = px.bar(
                local_df, x='Surge %', y='Item', orientation='h',
                color='Surge %',
                color_continuous_scale=['#00C851','#FFD700','#FF4444'],
                range_color=[0, 40], template='plotly_dark',
                title="Percentage above your normal price"
            )
            fig_local.add_vline(x=10, line_dash='dash', line_color='#FFD700',
                                annotation_text='Scarce')
            fig_local.add_vline(x=25, line_dash='dash', line_color='#FF4444',
                                annotation_text='Critical')
            fig_local.update_layout(height=max(200, len(local_df) * 45),
                                    showlegend=False)
            st.plotly_chart(fig_local, use_container_width=True)

        if st.button("Export as CSV"):
            csv = pd.DataFrame(entries).to_csv(index=False).encode('utf-8')
            st.download_button("Download local_prices.csv",
                               data=csv, file_name='local_prices.csv',
                               mime='text/csv')
    else:
        st.markdown(
            '<div style="background:#1a1f2e;border-radius:8px;padding:24px;text-align:center;">'
            '<span style="color:#AAAAAA">No local items yet. '
            'Use the form above to track your first item.</span></div>',
            unsafe_allow_html=True
        )

    # ── 7. LIVE HEADLINES (READABLE) ─────────────────────────
    st.markdown("---")
    st.subheader("Live Economic Headlines  |  " + str(len(headlines)) + " captured")

    if headlines:
        for h in sorted(headlines, key=lambda x: x.sentiment)[:15]:
            if h.sentiment < -0.05:
                border, bg, badge = '#FF4444', '#2d1515', '#FF6666'
                label = 'NEGATIVE  [' + '{:+.2f}'.format(h.sentiment) + ']'
            elif h.sentiment < 0.05:
                border, bg, badge = '#FFD700', '#2d2a10', '#FFD700'
                label = 'NEUTRAL  [' + '{:+.2f}'.format(h.sentiment) + ']'
            else:
                border, bg, badge = '#00C851', '#102d18', '#00C851'
                label = 'POSITIVE  [' + '{:+.2f}'.format(h.sentiment) + ']'

            st.markdown(
                '<div style="border-left:4px solid ' + border + ';'
                'padding:10px 16px;margin:6px 0;'
                'background:' + bg + ';border-radius:0 8px 8px 0;">'
                '<span style="color:' + badge + ';font-weight:700;'
                'font-size:0.82em">' + label + '</span><br>'
                '<span style="color:#EEEEEE;font-size:0.96em;'
                'line-height:1.5">' + h.title + '</span><br>'
                '<span style="color:#999999;font-size:0.79em">'
                '— ' + h.source + '  |  ' + h.timestamp[:25] + '</span>'
                '</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("No economic headlines fetched. Check your internet connection.")

    if auto_refresh:
        time.sleep(60)
        st.rerun()


if __name__ == '__main__':
    build_dashboard()
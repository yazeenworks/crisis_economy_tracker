# Crisis Economy Tracker

Real-time economic crisis monitoring dashboard combining
live market data, news sentiment and commodity prices
into a single Crisis Sentiment Index (CSI) score.

## Features
- Live CSI score (0-100) with 5 weighted market signals
- Backtested against 2008 crash, COVID-19 and SVB collapse
- Food, utility and medicine price tracker
- Local market price input with persistent storage
- Live economic headlines with sentiment scoring
- News conflict detection algorithm

## Tech Stack
Python, Streamlit, yfinance, NLTK VADER,
Plotly, BeautifulSoup, OpenFDA API

## Run Locally
pip install -r requirements.txt
streamlit run crisis_tracker.py

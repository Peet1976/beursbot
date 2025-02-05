import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import numpy as np
from sklearn.linear_model import LinearRegression

st.title("📈 AI Beursbot - Aandelenanalyse en Voorspellingen")

# Selecteer voorspellingsperiode
prediction_days = st.selectbox("Selecteer de voorspellingsperiode:", [1, 2, 5, 10])

def get_news(ticker):
    url = f"https://www.google.com/search?q={ticker}+stock+news&tbm=nws"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    articles = [item.get_text() for item in soup.find_all("div", class_="BNeawe vvjwJb AP7Wnd")[:5]]
    return articles

def analyze_sentiment(news_list):
    sentiments = []
    for news in news_list:
        sentiment_score = TextBlob(news).sentiment.polarity
        sentiment = "Positief" if sentiment_score > 0 else "Negatief" if sentiment_score < 0 else "Neutraal"
        sentiments.append((news, sentiment))
    return sentiments

def predict_stock_prices(data, days):
    data = data.reset_index()
    data["Date"] = data["Date"].map(pd.Timestamp.toordinal)
    X = np.array(data["Date"]).reshape(-1, 1)
    y = np.array(data["Close"].reshape(-1, 1))
    
    if len(X) < 5:
        return None, None  # Voorkom fout bij te weinig data
    
    model = LinearRegression()
    model.fit(X, y)
    
    future_dates = [data["Date"].max() + i for i in range(1, days+1)]
    future_prices = model.predict(np.array(future_dates).reshape(-1, 1))
    
    future_dates = pd.to_datetime([pd.Timestamp.fromordinal(int(d)) for d in future_dates])
    
    return future_dates, future_prices

def get_stock_data(tickers, period="6mo"):
    stock_data = {}
    valid_tickers = []
    invalid_tickers = []
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        history = stock.history(period=period)
        if history.empty:
            invalid_tickers.append(ticker)
        else:
            stock_data[ticker] = history
            valid_tickers.append(ticker)
    
    if invalid_tickers:
        st.error(f"⚠️ De volgende aandelen zijn niet gevonden: {', '.join(invalid_tickers)}. Controleer de tickers en probeer opnieuw.")
    
    return stock_data, valid_tickers

# Aandelen invoeren
tickers = st.text_input("Voer de tickers van de aandelen in (gescheiden door komma's):", "AAPL, TSLA")
tickers = [ticker.strip().upper() for ticker in tickers.split(",")]

if tickers:
    stock_data, valid_tickers = get_stock_data(tickers)
    if not valid_tickers:
        st.warning("Geen geldige aandelen gevonden. Probeer opnieuw.")
    else:
        for ticker in valid_tickers:
            st.subheader(f"📊 Aandelenkoers van {ticker}")
            st.line_chart(stock_data[ticker]["Close"])
            
            st.subheader("📢 Recent Nieuws")
            news = get_news(ticker)
            analyzed_news = analyze_sentiment(news)
            for article, sentiment in analyzed_news:
                st.write(f"📰 {article} - **{sentiment}**")
            
            st.subheader(f"🔮 AI Voorspelde Koers ({prediction_days} dagen)")
            future_dates, future_prices = predict_stock_prices(stock_data[ticker], days=prediction_days)
            if future_dates is not None:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=stock_data[ticker].index, y=stock_data[ticker]["Close"], mode='lines', name='Historische Data'))
                fig.add_trace(go.Scatter(x=future_dates, y=future_prices.flatten(), mode='lines', name='Voorspelling', line=dict(dash='dot', color='blue')))
                st.plotly_chart(fig)
            else:
                st.warning(f"❌ Onvoldoende data beschikbaar om een voorspelling te maken voor {ticker}. Probeer een andere periode of aandeel.")

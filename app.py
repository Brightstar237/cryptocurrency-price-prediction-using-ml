import streamlit as st
import yfinance as yf
import plotly.express as px
import numpy as np
from datetime import datetime
import mwclient
import time
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from xgboost import XGBClassifier
from transformers import pipeline
from stocknews import StockNews

st.header("Cryptocurrency Price Predicting System")
#Defining tickers variable
Bitcoin = "BTC-USD" or 'Bitcoin'
Ethereum = "ETH-USD"
Ripple = "XRP-USD"
Bitcoincash = "BCH-USD"
#get the tickers
ticker = st.sidebar.selectbox("Tickers", options=[Bitcoin, Ethereum, Ripple, Bitcoincash])
start_date = st.sidebar.date_input('Start Date')
end_date = st.sidebar.date_input('End Date')
def load_sentiment(model_name):
        sentiment_pipeline = pipeline(model_name)
        return (sentiment_pipeline)
sentiment_pipeline =load_sentiment("sentiment-analysis")

data = yf.download(ticker, start_date, end_date)
fig = px.line(data, x = data.index, y = data['Adj Close'], title=ticker)
st.plotly_chart(fig)


pricing_data, fundamental_data, news = st.tabs(["Pricing Data", "Fundament Data", "News"])

with pricing_data:
    st.write('Price Mouvements')
    data2 = data
    data['% change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
    data2.dropna(inplace =  True)
    st.write(data2)
    annual_return = data2['% change'].mean()*252*100
    st.write('Annual Return is', annual_return, '%')
    

with news:
    st.header(f'News of {ticker}')
    sn = StockNews(ticker, save_news=False)
    df_news =sn.read_rss()
    for i in range(10):
        st.header(f'News {i + 1}')
        st.write(df_news['published'][i])
        st.write(df_news['title'][i])
        st.write(df_news['summary'][i])
        title_sentiment = df_news['sentiment_title'][i]
        st.write(f'Title Sentiment {title_sentiment}')
        news_sentiment = df_news["sentiment_summary"][i]
        st.write(f'News Sentiment {news_sentiment}')

with fundamental_data:
    st.header(f'fundamental data of {ticker}')
    st.subheader('sentiment analyses')
    site = mwclient.Site('en.wikipedia.org')
    x_page = st.sidebar.text_input("Enter the page name","Bitcoin")
    page = site.pages[x_page]
    revs = list(page.revisions())
    revs = sorted(revs, key=lambda rev: rev["timestamp"]) 
    #st.table(revs)

    @st.cache_resource(experimental_allow_widgets=True) 
    def load_sentiment(model_name):
        sentiment_pipeline = pipeline(model_name)
        return (sentiment_pipeline)
    sentiment_pipeline =load_sentiment("sentiment-analysis")

    def find_sentiment(text):
        sent = sentiment_pipeline([text[:250]])[0]
        score = sent["score"]
        if sent["label"] == "NEGATIVE":
            score *= -1
        return score
    edits = {}

    for rev in revs:        
        date = time.strftime("%Y-%m-%d", rev["timestamp"])
        if date not in edits:
            edits[date] = dict(sentiments=list(), edit_count=0)
        
        edits[date]["edit_count"] += 1
        
        comment = rev.get("comment", "")
        edits[date]["sentiments"].append(find_sentiment(comment))
    from statistics import mean

    for key in edits:
        if len(edits[key]["sentiments"]) > 0:
            edits[key]["sentiment"] = mean(edits[key]["sentiments"])
            edits[key]["neg_sentiment"] = len([s for s in edits[key]["sentiments"] if s < 0]) / len(edits[key]["sentiments"])
        else:
            edits[key]["sentiment"] = 0
            edits[key]["neg_sentiment"] = 0
        
        del edits[key]["sentiments"]
    

    edits_df = pd.DataFrame.from_dict(edits, orient="index")

    edits_df.index = pd.to_datetime(edits_df.index)
    dates = pd.date_range(start="2009-03-08",end=datetime.today())
    edits_df = edits_df.reindex(dates, fill_value=0)

    #st.write(edits_df)

    rolling_edits = edits_df.rolling(30, min_periods=30).mean()
    rolling_edits = rolling_edits.dropna()

    #st.write(rolling_edits)
    rolling_edits.to_csv("wikipedia_edits.csv")
    st.button("Sentiment Analysis", on_click=load_sentiment)
    wiki = pd.read_csv("wikipedia_edits.csv", index_col=0, parse_dates=True)
    data = data.merge(wiki,left_index=True, right_index=True)
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    st.write(data["Target"].value_counts())
    sentiment_button = st.button('Load Sentiment data', on_click=load_sentiment)
    st.write(data)




    #predicting phase
    model_selector = st.sidebar.selectbox("Select Your Model", options=["xgboost","RandomForest"])

    if model_selector == "xgboost":
        def predict(train, test, predictors, model):
            model.fit(train[predictors], train["Target"])
            preds = model.predict(test[predictors])
            preds = pd.Series(preds, index=test.index, name="predictions")
            combined = pd.concat([test["Target"], preds], axis=1)
            combined1 = combined.astype(str)
            return combined1
        def backtest(data1, model, predictors, start=1095, step=150):
            all_predictions = []

            for i in range(start, data1.shape[0], step):
                train = data1.iloc[0:i].copy()
                test = data1.iloc[i:(i+step)].copy()
                predictions = predict(train, test, predictors, model)
                all_predictions.append(predictions)
            
            return pd.concat(all_predictions)
        from xgboost import XGBClassifier
        predictors = ["Close", "Volume", "Open", "High", "Low", "edit_count", "sentiment", "neg_sentiment"]


        model = XGBClassifier(random_state=1, learning_rate=.1, n_estimators=200)
        predictions = backtest(data, model, predictors)

        st.write(predictions["predictions"].value_counts())
        st.write(precision_score(predictions["Target"], predictions["predictions"]))


        def compute_rolling(data):
            horizons = [2,7,60,365]
            new_predictors = ["Close", "sentiment", "neg_sentiment"]

            for horizon in horizons:
                rolling_averages = data.rolling(horizon, min_periods=1).mean()

                ratio_column = f"close_ratio_{horizon}"
                data[ratio_column] = data["Close"] / rolling_averages["Close"]
                
                edit_column = f"edit_{horizon}"
                data[edit_column] = rolling_averages["edit_count"]

                rolling = data.rolling(horizon, closed='left', min_periods=1).mean()
                trend_column = f"trend_{horizon}"
                data[trend_column] = rolling["Target"]

                new_predictors+= [ratio_column, trend_column, edit_column]
            return data, new_predictors

        data, new_predictors = compute_rolling(data.copy())
        predictions = backtest(data, model, new_predictors)
        st.write(precision_score(predictions["Target"], predictions["predictions"]))

        st.write(predictions)
    elif model_selector == "RandomForest":
        st.subheader("Random Forest Prediction")
        from sklearn.ensemble import RandomForestClassifier

        model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1, max_depth=50, min_samples_leaf=12)

        train = data.iloc[:-200]
        test = data[-200:]

        predictors = ["Close", "Volume", "Open", "High", "Low", "edit_count", "sentiment", "neg_sentiment"]
        model.fit(train[predictors], train["Target"])

        from sklearn.metrics import precision_score

        preds = model.predict(test[predictors])
        preds = pd.Series(preds, index=test.index)
        st.write(precision_score(test["Target"], preds))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from datetime import datetime
from tensorflow.keras.models import load_model
import openai
# from openai.error import AuthenticationError

# LangChain / RAG Imports
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# --------------------------------------------------
# CACHING FUNCTIONS
# --------------------------------------------------

@st.cache_data
def load_stock_data():
    stock_analysis = {}
    for f in os.listdir("news_data"):
        if f.endswith("_llm_analysis.csv"):
            ticker = f.replace("_llm_analysis.csv", "")
            df = pd.read_csv(os.path.join("news_data", f))
            df["time_published"] = pd.to_datetime(df["time_published"])
            stock_analysis[ticker] = df
    return stock_analysis

@st.cache_data
def analyze_lstm_predictions():
    predictions = {}
    for f in os.listdir("outputs/lstm_predictions"):
        if f.endswith("_predictions.csv"):
            ticker = f.replace("_predictions.csv", "")
            df = pd.read_csv(os.path.join("outputs/lstm_predictions", f))
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            last = df.sort_values("timestamp").iloc[-1]
            predictions[ticker] = {
                "last_close": last["close"],
                "predicted_movement": (
                    "Up" if last["lstm_predictions"] > last["close"] else "Down"
                ),
            }
    return predictions

@st.cache_resource
def load_lstm_model_and_scaler(ticker):
    model = load_model(f"models/LSTM/{ticker}.h5")
    scaler = pickle.load(open(f"models/Scalers/{ticker}.pkl", "rb"))
    return model, scaler

@st.cache_resource
def build_qa_chain(api_key: str):
    # Load and chunk documents
    docs = []
    for f in os.listdir("news_data"):
        if f.endswith("_llm_analysis.csv"):
            ticker = f.replace("_llm_analysis.csv", "")
            df = pd.read_csv(os.path.join("news_data", f))
            df["time_published"] = pd.to_datetime(df["time_published"])
            for _, row in df.iterrows():
                content = (
                    f"Ticker: {ticker}\n"
                    f"Date: {row['time_published'].date()}\n"
                    f"Title: {row.get('title','')}\n"
                    f"Summary: {row.get('summary','')}\n"
                    f"Sentiment: {row.get('overall_sentiment_label','')}\n"
                    f"LLM Prediction: {row.get('llm_predicted_movement','')}\n"
                    f"Explanation: {row.get('llm_explanation','')}\n"
                )
                docs.append(Document(page_content=content, metadata={"ticker": ticker}))

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = [
        Document(page_content=chunk, metadata=doc.metadata)
        for doc in docs
        for chunk in splitter.split_text(doc.page_content)
    ]

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 3

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    faiss_retriever = FAISS.from_documents(chunks, embeddings).as_retriever(search_kwargs={"k": 3})

    hybrid = EnsembleRetriever(
        retrievers=[bm25, faiss_retriever],
        weights=[0.5, 0.5]
    )

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.0,
        openai_api_key=api_key
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=hybrid,
        return_source_documents=True
    )

# --------------------------------------------------
# RAG CHATBOT FUNCTION
# --------------------------------------------------

def rag_chatbot_response(user_input, stock_analysis, lstm_predictions, qa_chain):
    ui = user_input.lower()
    query_support1 = ""
    query_support2 = (
        "Understand the number of stocks that user wants to know about. "
        "Pick stocks from the above list based on performance but provide user "
        "with necessary information of the exact number of stocks that the user "
        "wants to know about. If user does not mention any number of stocks, "
        "then pick top 5 stocks from the list."
    )

    if "top stocks" in ui:
        ups = [t for t, d in lstm_predictions.items() if d["predicted_movement"] == "Up"]
        query_support1 = f"Top stocks for the upcoming week: {', '.join(ups)}. " + query_support2
    if "falling stocks" in ui or "down" in ui or "avoid" in ui:
        downs = [t for t, d in lstm_predictions.items() if d["predicted_movement"] == "Down"]
        query_support1 = f"Stocks predicted to fall: {', '.join(downs)}. " + query_support2

    ticker = next(
        (t for t in stock_analysis.keys() if t in user_input.upper()),
        None
    )
    query = user_input if not ticker else f"{ticker} {user_input}"

    context = (
        "You are a helpful stock analysis assistant. "
        "Base your answer on the latest news, overall sentiment and LSTM predictions. "
        "Be actionable and support your claims with numbers from news wherever necessary. "
        "You are an expert in stock recommendation and you have a good understanding "
        "of when to be concise versus elaborative.\n\n"
    )

    qa_output = qa_chain({"query": context + query + query_support1})
    return qa_output["result"]

# --------------------------------------------------
# STREAMLIT APP
# --------------------------------------------------

st.set_page_config(layout="wide")
st.title("StonksAI: Stock Predictor")

# Load data & models (cached)
stock_analysis   = load_stock_data()
lstm_predictions = analyze_lstm_predictions()

# Sidebar: company selector
prediction_dir = "outputs/lstm_predictions"
tickers = sorted([
    f.replace("_predictions.csv", "")
    for f in os.listdir(prediction_dir)
    if f.endswith("_predictions.csv")
])
ticker = st.sidebar.selectbox("Select company", tickers)

# Tabs
tab1, tab2, tab3 = st.tabs(["LSTM Forecast", "News Sentiment", "Chatbot"])

with tab1:
    st.subheader(f"LSTM Forecast for {ticker}")
    df = pd.read_csv(os.path.join(prediction_dir, f"{ticker}_predictions.csv"))
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    model, scaler = load_lstm_model_and_scaler(ticker)
    last_prices = df["close"].values[-30:]
    scaled = scaler.transform(last_prices.reshape(-1, 1))
    X_input = scaled.reshape(1, 1, 30)
    next_scaled = model.predict(X_input)
    next_price = scaler.inverse_transform(next_scaled)[0][0]

    last_date = df["timestamp"].max()
    next_date = last_date + pd.DateOffset(months=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["timestamp"], df["close"], label="Actual", marker='o', alpha=0.8, markersize=3)
    ax.plot(df["timestamp"], df["lstm_predictions"], label="Predicted", marker='o', alpha=0.8, markersize=3)
    ax.annotate(
        f"${next_price:.2f}",
        xy=(next_date, next_price),
        xytext=(0, 10),
        textcoords='offset points',
        ha='center',
        fontsize=10,
        color='red',
        weight='bold'
    )
    ax.scatter(next_date, next_price, color="red", label="Next Forecast", zorder=5, s=100, marker='X')
    ax.axvline(next_date, color='red', linestyle=':', alpha=0.3)
    ax.set_xlabel("Date")
    ax.set_ylabel("Closing Price")
    ax.set_title(f"{ticker} Price Forecast")
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.subheader(f"News-Based Stock Sentiment: {ticker}")
    news_path = os.path.join("news_data", f"{ticker}_llm_analysis.csv")
    if os.path.exists(news_path):
        nd = pd.read_csv(news_path)
        nd["time_published"] = pd.to_datetime(nd["time_published"])
        nd = nd.sort_values("time_published", ascending=False)

        min_d = nd["time_published"].min().to_pydatetime()
        max_d = nd["time_published"].max().to_pydatetime()
        sel = st.slider("Select date range:", min_value=min_d, max_value=max_d, value=(min_d, max_d))
        filt = nd[(nd["time_published"] >= sel[0]) & (nd["time_published"] <= sel[1])]

        for _, row in filt.iterrows():
            st.markdown(f"### 🗞️ {row['title']}")
            st.markdown(
                f"**Published:** {row['time_published'].date()} "
                f"| **Sentiment:** {row['overall_sentiment_label']} "
                f"| **LLM Prediction:** *{row['llm_predicted_movement']}*"
            )
            st.markdown(f"**LLM Explanation:** {row['llm_explanation']}")
            st.markdown(f"> {row['summary']}")
            st.markdown("---")
    else:
        st.warning(f"No news sentiment data found for {ticker}.")

with tab3:
    st.subheader("Stock Market Chatbot")

    # API key input only in Tab 3
    api_key_input = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        help="This key is used only for the chatbot.",
        key="api_key_input"
    )

    if api_key_input:
        # no more import AuthenticationError
        try:
            qa_chain = build_qa_chain(api_key_input)
        except Exception:
            st.error("❌ Invalid API key or network error. Please check and try again.")
        else:
            def handle_chat():
                try:
                    st.session_state.chat_response = rag_chatbot_response(
                        st.session_state.chat_query,
                        stock_analysis,
                        lstm_predictions,
                        qa_chain
                    )
                except Exception as e:
                    st.session_state.chat_response = None
                    st.error(f"⚠️ Chatbot error: {e}")

            st.text_input(
                "Ask the chatbot about stock recommendations:",
                key="chat_query",
                on_change=handle_chat
            )

            if "chat_response" in st.session_state and st.session_state.chat_response:
                st.markdown(f"**Chatbot Response:** {st.session_state.chat_response}")
    else:
        st.info("Please enter your OpenAI API key to enable the chatbot.")

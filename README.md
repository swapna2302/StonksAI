# StonksAI
StonksAI is an AI-powered stock forecasting tool that combines LSTM-based time-series prediction with LLM-driven financial news sentiment analysis.

You can explore the app here: <a href="https://stonksai.streamlit.app" target="_blank">StonksAI</a>
### Project structure
```app.py```: Main Streamlit application that powers the interactive dashboard <br>
```Stock_News_Data.ipynb```: Notebook for fetching and organizing stock and news data <br>
```LSTM.ipynb```: Notebook for training LSTM models on historical stock data <br>
```LLM_prediction.ipynb```: Notebook for analyzing news articles and generating LLM-based sentiment forecasts<br>
```models/``` : Contains saved LSTM models and scalers <br>
```news_data/``` : 	Processed news articles and LLM predictions  <br>
`stock_data/`	: Monthwise closing price stock CSVs per company <br>
`outputs/` : Contains prediction outputs, plots, and metrics <br>
`requirements.txt` : 	List of Python dependencies for the project <br>

### Setup instructions to run locally
Clone the Repository
```
git clone https://github.com/swapna2302/StonksAI.git
cd StonksAI
```
Install dependencies:
```
pip install -r requirements.txt
```

Run the streamlit app using:
```
streamlit run app.py
```
### License
This project is licensed under the MIT License. See LICENSE for details.


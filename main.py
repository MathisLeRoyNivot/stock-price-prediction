import streamlit as st
from datetime import date
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go

START_DATE = "2015-01-01"
TODAY_DATE = date.today().strftime("%Y-%m-%d")

st.title("Stock Price Prediction App")

stocks = ("BTC-USD", "AAPL", "FB", "MSFT", "GOOG", "TSLA")
selected_stocks = st.selectbox("Select stock for prediction", stocks)

n_year = st.slider("Years of prediction:", 1, 4)
period = n_year * 365


@st.cache
def load_stock_data(ticker):
    data = yf.download(ticker, START_DATE, TODAY_DATE)
    data.reset_index(inplace=True)

    return data


data_load_state = st.text("Load stock data...")
data = load_stock_data(selected_stocks)
data_load_state.text("Stock data loaded!")

st.subheader("Raw data")
st.write(data.tail())


def plot_raw_stock_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)

    st.plotly_chart(fig)


plot_raw_stock_data()

# Forecast
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast stock data")
st.write(forecast.tail())

st.write("Forecast stock data")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast component")
fig2 = m.plot_components(forecast)
st.write(fig2)


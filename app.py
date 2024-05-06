import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))

st.title("Crypto Currency Price Prediction")

var = st.selectbox("Select the Coin", ["BTC", "ETH", "LTC"])


df = pd.read_csv(f'Dataset/{var}-USD.csv')
st.write(df)

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
ma100 = df['Close'].rolling(100).mean()

plot_data = pd.DataFrame({
    'Close Price': df['Close'],
    'MA 100 Days': ma100,
})

st.line_chart(plot_data)

col1, col2 = st.columns(2)
model_selection = col1.radio("Select the Model", ["Base", "Novel"])

days = col2.number_input("Enter the number of days to forecast",
                       min_value=1, max_value=365, value=10)


def create_dataset(window):
    if len(df) < window:raise ValueError("DataFrame is smaller than the specified window size.")

    scaled_data = scaler.transform(df[['Close']])
    n = len(scaled_data)
    X = [scaled_data[n - window: n]]
    return np.array(X)


def prepare_data(model):
    scaler.fit(df[['Close']])
    window = 0
    if model == "Base":
        window = 3
    else:
        window = 40
    X = create_dataset(window)
    return X[-1:]


def predict_trend(input_seq, days, model):
    model = load_model(f'./Models/{model_selection}_model_{var}.h5')
    
    predictions = []
    for _ in range(days):
        pred = model.predict(input_seq)

        next_input_seq = np.append(
            input_seq[:, 1:, :], pred.reshape(1, 1, 1), axis=1)

        scaled_pred = scaler.inverse_transform(pred)
        input_seq = next_input_seq

        predictions.append(scaled_pred[0][0])

    return predictions

if col2.button("Predict"):
    input_seq = prepare_data(model_selection)
    
    predictions = predict_trend(input_seq, days, model_selection)
    net = predictions[-1] - df['Close'].iloc[-1]

    st.write(f"Predicted values for the next {days} days for {var}:")
    st.line_chart(predictions)
    st.write("Intial Value : ", df['Close'].iloc[-1])
    st.write("Final Value : ", predictions[-1])
    st.write("Net : ", net)


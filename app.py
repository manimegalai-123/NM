import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Set page configuration
st.set_page_config(page_title="Stock Price Prediction", page_icon="📈")

# Load the model and scaler
@st.cache_resource
def load_resources():
    model = load_model('stock_price_model.h5')
    scaler = joblib.load('scaler.pkl')  # Ensure this is the same scaler used during training
    return model, scaler

model, scaler = load_resources()

def predict_next_price(manual_input):
    if len(manual_input) != 60:
        return {"error": "Input must contain exactly 60 closing prices."}
    if not all(isinstance(x, (int, float)) for x in manual_input):
        return {"error": "All prices must be numbers."}
    try:
        manual_input = np.array(manual_input).reshape(-1, 1)

        # Directly scale using the scaler from training
        scaled_input = scaler.transform(manual_input)

        # Reshape to match model input shape: (1, 60, 1)
        scaled_input = np.reshape(scaled_input, (1, 60, 1))

        # Predict and inverse transform the result
        pred_scaled = model.predict(scaled_input, verbose=0)
        st.write(f"🔍 Scaled Prediction: {pred_scaled}")  # Debugging

        pred_price = scaler.inverse_transform(pred_scaled)
        return max(0, float(pred_price[0][0]))  # Ensure price is non-negative
    except Exception as e:
        return {"error": str(e)}

# Streamlit UI
st.title("📈 Stock Price Prediction")
st.write("Enter **60 closing prices** (comma-separated) to predict the next price.")

# Input field
prices_input = st.text_area("Closing Prices", placeholder="e.g., 101.5,102.1,100.7,... (60 prices)")

if st.button("Predict"):
    if prices_input:
        try:
            # Parse input
            prices = [float(x.strip()) for x in prices_input.split(",")]
            result = predict_next_price(prices)
            if isinstance(result, dict) and "error" in result:
                st.error(result["error"])
            else:
                st.success(f"📊 Predicted Next Price")
        except ValueError:
            st.error("Please enter valid numbers separated by commas.")
    else:
        st.error("Please enter 60 closing prices.")

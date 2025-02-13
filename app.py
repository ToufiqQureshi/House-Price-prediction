import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open('regressor.pkl', 'rb'))

# Streamlit UI
st.title("California House Price Prediction")

st.sidebar.header("Enter House Features:")
medinc = st.sidebar.number_input("Median Income", min_value=0.0, format="%.4f")
house_age = st.sidebar.number_input("House Age", min_value=0.0, format="%.1f")
ave_rooms = st.sidebar.number_input("Average Rooms", min_value=0.0, format="%.6f")
ave_bedrms = st.sidebar.number_input("Average Bedrooms", min_value=0.0, format="%.6f")
population = st.sidebar.number_input("Population", min_value=0.0, format="%.1f")
ave_occup = st.sidebar.number_input("Average Occupancy", min_value=0.0, format="%.6f")
latitude = st.sidebar.number_input("Latitude", format="%.2f")
longitude = st.sidebar.number_input("Longitude", format="%.2f")

# Prediction button
if st.sidebar.button("Predict Price"):
    # Prepare input features
    features = np.array([[medinc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]])
    
    # Predict
    prediction = model.predict(features)[0]
    
    st.success(f"Estimated House Price: ${prediction*100000:.2f}")

# Footer
st.markdown("---")
st.markdown("Developed by **Toufiq Qureshi** | Powered by Streamlit & Machine Learning")

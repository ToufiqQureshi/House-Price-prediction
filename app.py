import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Model Load
try:
    model = pickle.load(open("regressor.pkl", "rb"))
except Exception as e:
    st.error("⚠️ Model load nahi ho raha! 'regressor.pkl' file check karo.")
    st.stop()

# Title
st.title("🏠 House Price Prediction - India 🇮🇳")
st.write("👇 Apni details bhariye aur ghar ka andaza daam janein!")

# **Mobile aur PC ke hisaab se Responsive Columns**
if st.sidebar.checkbox("📱 Mobile View (Compact Mode)"):
    col1 = st.container()
    col2 = st.container()
    col3 = st.container()
else:
    col1, col2, col3 = st.columns(3)

with col1:
    medinc = st.number_input("💰 Aamdaani (Lakh ₹/ Saal)", min_value=0.0, max_value=15.0, step=0.1, format="%.1f")
    population = st.number_input("👨‍👩‍👧‍👦 Area Me Kitne Log?", min_value=100, max_value=5000, step=50)
    latitude = st.number_input("🌍 Latitude (37-38)", min_value=37.0, max_value=38.0, step=0.01, format="%.2f")

with col2:
    house_age = st.number_input("🏡 Ghar Ki Umar (Saal)", min_value=0, max_value=100, step=1)
    ave_occup = st.number_input("🏠 Har Ghar Me Kitne Log?", min_value=1, max_value=10, step=1)
    longitude = st.number_input("🌎 Longitude (-123 to -121)", min_value=-123.0, max_value=-121.0, step=0.01, format="%.2f")

with col3:
    ave_rooms = st.number_input("🛏️ Ek Ghar Me Kitne Kamre?", min_value=1, max_value=10, step=1)
    ave_bedrms = st.number_input("🛌 Ek Ghar Me Kitne Bedroom?", min_value=1, max_value=5, step=1)

# Prediction Button
st.markdown("---")
if st.button("💰 Daam Janein", use_container_width=True):
    try:
        features = np.array([[medinc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude]])
        prediction = model.predict(features)[0]
        price = prediction * 1000000
        st.success(f"🏠 Aapke Ghar Ka Andaza Daam: **₹{price:,.2f}**")

        # Chart: House Price vs Income
        st.subheader("📊 Price vs Income Chart")
        fig, ax = plt.subplots()
        incomes = np.linspace(0, 15, 50)
        prices = model.predict([[inc, house_age, ave_rooms, ave_bedrms, population, ave_occup, latitude, longitude] for inc in incomes]) * 1000000

        ax.plot(incomes, prices, color="blue", marker="o", linestyle="-")
        ax.set_xlabel("💰 Median Income (Lakh ₹)")
        ax.set_ylabel("₹ Price (in Lakhs)")
        ax.set_title("📈 Income vs House Price")
        st.pyplot(fig)

    except Exception as e:
        st.error("❌ Prediction issue! Sahi data bhariye.")

# Footer
st.markdown("---")
st.markdown("🔹 **Banaya gaya India ke liye!** 🇮🇳 | **Powered by Machine Learning & Streamlit**")

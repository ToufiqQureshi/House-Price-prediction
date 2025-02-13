![alt text](image.png)

# 🏡 California House Price Prediction

This is a **Machine Learning Web App** built using **Streamlit**. It predicts **California House Prices** based on user input features using a trained regression model.

## 🚀 Features
✅ Load a pre-trained **California housing price model**  
✅ Accept user inputs like **Median Income, House Age, Number of Rooms, Population, Location**, etc.  
✅ Predict the **house price** in real-time  
✅ **Interactive UI** built using Streamlit  
✅ **Easy to install and run**  

---

## 📂 Project Structure
📁 House-Price-Prediction │-- 📄 app.py # Main Streamlit app │-- 📄 regressor.pkl # Trained Machine Learning model │-- 📄 README.md # Project documentation │-- 📄 requirements.txt # List of dependencies

yaml
Copy
Edit

---

## 📥 Installation Guide

1️⃣ **Clone the Repository**  
```bash
git clone https://github.com/ToufiqQureshi/House-Price-prediction.git
cd House-Price-Prediction
2️⃣ Create & Activate Virtual Environment

bash
Copy
Edit
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
3️⃣ Install Dependencies

bash
Copy
Edit
pip install -r requirements.txt
4️⃣ Run the App

bash
Copy
Edit
streamlit run app.py
🏗️ Model Details
The model is trained on California housing dataset

Features used:

MedInc: Median income in the area
HouseAge: Age of the house
AveRooms: Average number of rooms
AveBedrms: Average number of bedrooms
Population: Total population in the area
AveOccup: Average number of occupants per household
Latitude, Longitude: Location coordinates
The trained model is stored as regressor.pkl using pickle.
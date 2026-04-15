import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("House Price Prediction App")
st.write("Enter house details to predict the sale price.")

# Load model and data
model = joblib.load("model.pkl")
data = pd.read_csv("AmesHousing.csv")

# User inputs
user_input = {
    "Overall Qual": st.slider("Overall Quality", 1, 10, 5),
    "Gr Liv Area": st.number_input("Ground Living Area", min_value=300, max_value=6000, value=1500),
    "Garage Cars": st.slider("Garage Capacity (Cars)", 0, 5, 2),
    "Garage Area": st.number_input("Garage Area", min_value=0, max_value=2000, value=500),
    "Total Bsmt SF": st.number_input("Total Basement Area", min_value=0, max_value=4000, value=800),
    "1st Flr SF": st.number_input("1st Floor Area", min_value=300, max_value=3000, value=1000),
    "2nd Flr SF": st.number_input("2nd Floor Area", min_value=0, max_value=3000, value=0),
    "Full Bath": st.slider("Full Bathrooms", 0, 5, 2),
    "Half Bath": st.slider("Half Bathrooms", 0, 3, 1),
    "Bsmt Full Bath": st.slider("Basement Full Bathrooms", 0, 3, 0),
    "Bsmt Half Bath": st.slider("Basement Half Bathrooms", 0, 2, 0),
    "Year Built": st.number_input("Year Built", min_value=1800, max_value=2025, value=2000),
    "Year Remod/Add": st.number_input("Year Remodeled", min_value=1800, max_value=2025, value=2005),
    "Yr Sold": st.number_input("Year Sold", min_value=2006, max_value=2025, value=2010),
    "Lot Area": st.number_input("Lot Area", min_value=1000, max_value=100000, value=10000),
    "Overall Cond": st.slider("Overall Condition", 1, 10, 5),
    "Fireplaces": st.slider("Number of Fireplaces", 0, 4, 1),
    "Neighborhood": st.selectbox("Neighborhood", sorted(data["Neighborhood"].dropna().unique())),
    "House Style": st.selectbox("House Style", sorted(data["House Style"].dropna().unique())),
    "MS Zoning": st.selectbox("MS Zoning", sorted(data["MS Zoning"].dropna().unique())),
    "Central Air": st.selectbox("Central Air", sorted(data["Central Air"].dropna().unique())),
    "Kitchen Qual": st.selectbox("Kitchen Quality", sorted(data["Kitchen Qual"].dropna().unique())),
    "Garage Finish": st.selectbox("Garage Finish", sorted(data["Garage Finish"].dropna().astype(str).unique())),
    "Paved Drive": st.selectbox("Paved Drive", sorted(data["Paved Drive"].dropna().unique())),
    "Sale Type": st.selectbox("Sale Type", sorted(data["Sale Type"].dropna().unique())),
    "Sale Condition": st.selectbox("Sale Condition", sorted(data["Sale Condition"].dropna().unique()))
}

input_df = pd.DataFrame([user_input])

# Add missing columns with default values from original dataset
for col in data.columns:
    if col == "SalePrice":
        continue
    if col not in input_df.columns:
        if data[col].dtype == "object":
            input_df[col] = data[col].mode()[0]
        else:
            input_df[col] = data[col].median()

# Feature engineering to match training
input_df["TotalBath"] = (
    input_df["Full Bath"]
    + 0.5 * input_df["Half Bath"]
    + input_df["Bsmt Full Bath"]
    + 0.5 * input_df["Bsmt Half Bath"]
)

input_df["GarageScore"] = input_df["Garage Cars"] * input_df["Garage Area"]
input_df["HouseAgeAtSale"] = input_df["Yr Sold"] - input_df["Year Built"]
input_df["YearsSinceRemodel"] = input_df["Yr Sold"] - input_df["Year Remod/Add"]

# If your training used this engineered feature, include it too
input_df["TotalLivableSF"] = (
    input_df["Total Bsmt SF"] + input_df["1st Flr SF"] + input_df["2nd Flr SF"]
)

# Ensure all engineered columns exist
for engineered_col in [
    "TotalBath",
    "GarageScore",
    "HouseAgeAtSale",
    "YearsSinceRemodel",
    "TotalLivableSF",
]:
    if engineered_col not in input_df.columns:
        input_df[engineered_col] = 0

# Predict
if st.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted House Price: ${prediction:,.2f}")

import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("House Price Prediction App")
st.write("Enter house details to predict the sale price.")

# Load model and dataset
model = joblib.load("model.pkl")
data = pd.read_csv("AmesHousing.csv")

# ================= USER INPUT =================
user_input = {
    "Overall Qual": st.slider("Overall Quality", 1, 10, 5),
    "Gr Liv Area": st.number_input("Ground Living Area", 300, 6000, 1500),
    "Garage Cars": st.slider("Garage Capacity (Cars)", 0, 5, 2),
    "Garage Area": st.number_input("Garage Area", 0, 2000, 500),
    "Total Bsmt SF": st.number_input("Total Basement Area", 0, 4000, 800),
    "1st Flr SF": st.number_input("1st Floor Area", 300, 3000, 1000),
    "2nd Flr SF": st.number_input("2nd Floor Area", 0, 3000, 0),
    "Full Bath": st.slider("Full Bathrooms", 0, 5, 2),
    "Half Bath": st.slider("Half Bathrooms", 0, 3, 1),
    "Bsmt Full Bath": st.slider("Basement Full Bathrooms", 0, 3, 0),
    "Bsmt Half Bath": st.slider("Basement Half Bathrooms", 0, 2, 0),
    "Year Built": st.number_input("Year Built", 1800, 2025, 2000),
    "Year Remod/Add": st.number_input("Year Remodeled", 1800, 2025, 2005),
    "Yr Sold": st.number_input("Year Sold", 2006, 2025, 2010),
    "Lot Area": st.number_input("Lot Area", 1000, 100000, 10000),
    "Overall Cond": st.slider("Overall Condition", 1, 10, 5),
    "Fireplaces": st.slider("Number of Fireplaces", 0, 4, 1),

    # Categorical inputs
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

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# ================= FIX MISSING COLUMNS =================
for col in data.columns:
    if col == "SalePrice":
        continue
    if col not in input_df.columns:
        if data[col].dtype == "object":
            input_df[col] = data[col].mode()[0]   # categorical
        else:
            input_df[col] = data[col].median()   # numeric

# ================= FEATURE ENGINEERING =================
input_df["TotalBath"] = (
    input_df["Full Bath"]
    + 0.5 * input_df["Half Bath"]
    + input_df["Bsmt Full Bath"]
    + 0.5 * input_df["Bsmt Half Bath"]
)

input_df["GarageScore"] = input_df["Garage Cars"] * input_df["Garage Area"]
input_df["HouseAgeAtSale"] = input_df["Yr Sold"] - input_df["Year Built"]
input_df["YearsSinceRemodel"] = input_df["Yr Sold"] - input_df["Year Remod/Add"]

input_df["TotalLivableSF"] = (
    input_df["Total Bsmt SF"]
    + input_df["1st Flr SF"]
    + input_df["2nd Flr SF"]
)

# ================= PREDICTION =================
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted House Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Error: {e}")

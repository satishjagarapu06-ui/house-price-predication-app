import streamlit as st
import pandas as pd
import joblib

# ================= PAGE CONFIG =================
st.set_page_config(page_title="House Price Prediction", layout="centered")

st.title("House Price Prediction App")
st.write("Enter house details to predict the sale price.")

# ================= LOAD MODEL + DATA =================
model = joblib.load("model.pkl")
data = pd.read_csv("AmesHousing.csv")
data.columns = data.columns.str.strip()

# ================= USER INPUT =================
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

    # Categorical inputs
    "Neighborhood": st.selectbox(
        "Neighborhood",
        sorted(data["Neighborhood"].dropna().astype(str).unique())
    ),
    "House Style": st.selectbox(
        "House Style",
        sorted(data["House Style"].dropna().astype(str).unique())
    ),
    "MS Zoning": st.selectbox(
        "MS Zoning",
        sorted(data["MS Zoning"].dropna().astype(str).unique())
    ),
    "Central Air": st.selectbox(
        "Central Air",
        sorted(data["Central Air"].dropna().astype(str).unique())
    ),
    "Kitchen Qual": st.selectbox(
        "Kitchen Quality",
        sorted(data["Kitchen Qual"].dropna().astype(str).unique())
    ),
    "Garage Finish": st.selectbox(
        "Garage Finish",
        sorted(data["Garage Finish"].dropna().astype(str).unique())
    ),
    "Paved Drive": st.selectbox(
        "Paved Drive",
        sorted(data["Paved Drive"].dropna().astype(str).unique())
    ),
    "Sale Type": st.selectbox(
        "Sale Type",
        sorted(data["Sale Type"].dropna().astype(str).unique())
    ),
    "Sale Condition": st.selectbox(
        "Sale Condition",
        sorted(data["Sale Condition"].dropna().astype(str).unique())
    ),
}

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

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

# ================= EXPECTED MODEL COLUMNS =================
# Best case: model exposes the exact training columns
if hasattr(model, "feature_names_in_"):
    expected_cols = list(model.feature_names_in_)
else:
    # Fallback: use current columns if model doesn't expose names
    expected_cols = list(input_df.columns)

# ================= FILL MISSING COLUMNS =================
for col in expected_cols:
    if col not in input_df.columns:
        if col in data.columns:
            if pd.api.types.is_numeric_dtype(data[col]):
                input_df[col] = pd.to_numeric(data[col], errors="coerce").median()
            else:
                non_null = data[col].dropna().astype(str)
                input_df[col] = non_null.mode()[0] if not non_null.empty else ""
        else:
            # Engineered or unknown feature missing from raw dataset
            input_df[col] = 0

# ================= TYPE ALIGNMENT =================
for col in input_df.columns:
    if col in data.columns:
        if pd.api.types.is_numeric_dtype(data[col]):
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
        else:
            input_df[col] = input_df[col].astype(str)

# ================= REMOVE EXTRA COLUMNS + ORDER MATCH =================
input_df = input_df.reindex(columns=expected_cols, fill_value=0)

# ================= PREDICTION =================
if st.button("Predict Price"):
    try:
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted House Price: ${prediction:,.2f}")

        with st.expander("Show processed input used for prediction"):
            st.dataframe(input_df)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

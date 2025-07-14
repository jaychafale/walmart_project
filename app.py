import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.express as px

st.set_page_config(page_title="Demand Forecast", layout="wide")
st.title("🛍 Demand Forecasting and Redistribution Alert System")

# Load model
model = xgb.XGBRegressor()
model.load_model("xgb_model.json")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Rename columns if needed
    rename_map = {
        'datetime': 'dt',
        'sku_id': 'product_id',
        'weather_temperature': 'avg_temperature',
        'weather_rainfall': 'avg_humidity'
    }
    df.rename(columns=rename_map, inplace=True)

    # Ensure datetime is parsed
    if 'dt' not in df.columns:
        st.error("Missing 'dt' column (datetime). Check your file.")
        st.stop()
    df['dt'] = pd.to_datetime(df['dt'])

    # Time-based features
    df['hour'] = df['dt'].dt.hour
    df['day'] = df['dt'].dt.day
    df['day_of_week'] = df['dt'].dt.dayofweek
    df['week'] = df['dt'].dt.isocalendar().week
    df['month'] = df['dt'].dt.month

    # Fill defaults for missing values
    df['sale_amount'] = df.get('sale_amount', 0)
    df['holiday_flag'] = df.get('holiday_flag', 0).astype(int)
    df['activity_flag'] = df.get('activity_flag', 0).astype(int)
    df['avg_wind_level'] = df.get('avg_wind_level', 0)

    for lag in [1, 2, 3, 6, 12, 24]:
        col = f'demand_lag_{lag}'
        if col not in df.columns:
            df[col] = 0

    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filter Data")
        store_options = df['store_id'].unique()
        selected_store = st.selectbox("Select Store", store_options)
        product_options = df[df['store_id'] == selected_store]['product_id'].unique()
        selected_product = st.selectbox("Select Product", product_options)

        min_date, max_date = df['dt'].min(), df['dt'].max()
        selected_range = st.date_input("Date Range", [min_date, max_date])

    df = df[(df['store_id'] == selected_store) & (df['product_id'] == selected_product)]
    df = df[(df['dt'] >= pd.to_datetime(selected_range[0])) & (df['dt'] <= pd.to_datetime(selected_range[1]))]

    # Prediction
    features = [
        'hour', 'day', 'day_of_week', 'week', 'month',
        'avg_temperature', 'avg_humidity', 'avg_wind_level',
        'holiday_flag', 'activity_flag'] + [f'demand_lag_{l}' for l in [1, 2, 3, 6, 12, 24]]

    X = df[features]
    df['forecasted_demand'] = model.predict(X)

    if 'inventory_level' not in df.columns:
        inv = st.slider("Set default inventory level", 0, 100, 50)
        df['inventory_level'] = inv

    df['needs_redistribution'] = df['forecasted_demand'] > df['inventory_level']
    df['shortfall'] = df['forecasted_demand'] - df['inventory_level']
    df['shortfall'] = df['shortfall'].apply(lambda x: max(x, 0))

    # KPI Summary
    st.markdown("### ⚖️ Forecast Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Flagged Items", int(df['needs_redistribution'].sum()))
    col2.metric("Total Shortfall (units)", int(df['shortfall'].sum()))
    col3.metric("Avg Forecasted Demand", round(df['forecasted_demand'].mean(), 2))

    # Forecast Chart
    st.markdown("### 📊 Forecast vs Inventory Over Time")
    st.plotly_chart(px.line(df, x='dt', y=['forecasted_demand', 'inventory_level'], title="Forecast vs Inventory"))

    # Highlighted Table
    st.subheader("🔍 Detailed Forecast Table")
    st.dataframe(df.style.applymap(lambda v: 'background-color: #ffcccc' if isinstance(v, (int, float)) and v > 0 else ''))

    st.subheader("📦 Items Needing Redistribution")
    flagged = df[df['needs_redistribution']]
    st.dataframe(flagged[['store_id', 'product_id', 'dt', 'forecasted_demand', 'inventory_level', 'shortfall']])

    st.download_button("⬇️ Download Forecasts", df.to_csv(index=False), file_name="forecasted_results.csv")

else:
    st.info("Upload a CSV with columns like 'dt', 'store_id', 'product_id', 'avg_temperature', etc.")

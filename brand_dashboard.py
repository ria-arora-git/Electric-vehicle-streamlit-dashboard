import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import random

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_data.csv")
    df['cargo_volume_l'] = pd.to_numeric(df['cargo_volume_l'], errors='coerce')
    return df

df = load_data()

st.markdown(
    """
    <div style="text-align: center;">
        <h1>EV Model Comparison Dashboard (2025)</h1>
        <p style="font-size:1.2em;">Compare electric vehicle models side by side to help you make informed decisions.</p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Top Filters in Columns ---
st.markdown("### Filter Models")

col1, col2, col3 = st.columns(3)

with col1:
    brands = sorted(df['brand'].dropna().unique())
    selected_brands = st.multiselect("Select Brands", brands, default=random.sample(brands, min(5, len(brands))))

with col2:
    drivetrains = sorted(df[df['brand'].isin(selected_brands)]['drivetrain'].dropna().unique())
    selected_drivetrains = st.multiselect("Select Drivetrains", drivetrains, default=drivetrains)

with col3:
    models = sorted(df[(df['brand'].isin(selected_brands)) &
                       (df['drivetrain'].isin(selected_drivetrains))]['model'].dropna().unique())
    selected_models = st.multiselect("Select Models to Compare", models, default=random.sample(models, min(5, len(models))))

df_filtered = df[(df['brand'].isin(selected_brands)) &
                 (df['drivetrain'].isin(selected_drivetrains)) &
                 (df['model'].isin(selected_models))]

# --- Key Metrics Table ---
st.subheader("Key Specifications of Selected Models")
display_df = df_filtered.rename(columns={
    "brand": "Brand",
    "model": "Model",
    "top_speed_kmh": "Top Speed (km/h)",
    "battery_capacity_kWh": "Battery Capacity (kWh)",
    "battery_type": "Battery Type",
    "torque_nm": "Torque (Nm)",
    "efficiency_wh_per_km": "Efficiency (Wh/km)",
    "range_km": "Range (km)",
    "acceleration_0_100_s": "Acceleration 0–100 km/h (s)",
    "fast_charging_power_kw_dc": "Fast Charging Power (kW DC)",
    "fast_charge_port": "Fast Charge Port",
    "towing_capacity_kg": "Towing Capacity (kg)",
    "cargo_volume_l": "Cargo Volume (L)",
    "seats": "Number of Seats",
    "drivetrain": "Drivetrain",
    "segment": "Segment",
    "length_mm": "Length (mm)",
    "width_mm": "Width (mm)",
    "height_mm": "Height (mm)",
    "car_body_type": "Car Body Type"
})
st.dataframe(display_df.reset_index(drop=True))

# --- Visualizations ---
if not df_filtered.empty:
    st.subheader("Visual Comparison of Models")

    fig_bubble = px.scatter(
        df_filtered,
        x="battery_capacity_kWh",
        y="top_speed_kmh",
        size="seats",
        color="brand",
        hover_name="model",
        title="Battery Capacity vs Top Speed (Bubble Size = Seats)",
        labels={
            "battery_capacity_kWh": "Battery (kWh)",
            "top_speed_kmh": "Top Speed (km/h)",
            "seats": "Seats"
        },
        height=600
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

    fig_line = px.line(
        df_filtered.sort_values("acceleration_0_100_s"),
        x="model",
        y="acceleration_0_100_s",
        markers=True,
        color="brand",
        title="Acceleration (0–100 km/h) Comparison",
        labels={"acceleration_0_100_s": "0–100 km/h (s)", "model": "Model"}
    )
    st.plotly_chart(fig_line, use_container_width=True)

    fig_bar = px.bar(
        df_filtered,
        x="model",
        y="battery_capacity_kWh",
        color="brand",
        title="Battery Capacity by Model",
        labels={"battery_capacity_kWh": "Battery (kWh)", "model": "Model"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Radar Chart: Multi-Feature Model Comparison")

    radar_cols = [
        "top_speed_kmh",
        "acceleration_0_100_s",
        "range_km",
        "battery_capacity_kWh",
        "cargo_volume_l"
    ]
    radar_df = df_filtered[["model"] + radar_cols].dropna()

    if not radar_df.empty:
        radar_df_norm = radar_df.copy()
        radar_df_norm["acceleration_0_100_s"] = -radar_df_norm["acceleration_0_100_s"]
        scaler = MinMaxScaler()
        radar_df_norm[radar_cols] = scaler.fit_transform(radar_df_norm[radar_cols])

        categories = radar_cols
        fig_radar = go.Figure()
        for i, row in radar_df_norm.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=row[categories].values.tolist(),
                theta=categories,
                fill='toself',
                name=row["model"]
            ))

        fig_radar.update_layout(
            polar=dict(
                bgcolor="#193450",
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            title="Radar Chart of Performance and Utility Specs",
            showlegend=True
        )
        st.plotly_chart(fig_radar, use_container_width=True)

else:
    st.warning("Please select at least one EV model to compare.")

# --- Raw Data ---
with st.expander("📄 Show Raw Data"):
    st.dataframe(df_filtered.reset_index(drop=True))

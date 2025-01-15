import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Page configuration
st.set_page_config(
    page_title="Traffic Prediction Dashboard",
    page_icon="🚦",
    layout="wide",
)

# Font settings
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# API URLs
DATA_API_URL = "http://127.0.0.1:8000/data"
LOCATION_API_BASE_URL = "http://127.0.0.1:8000/location"

# Fetch data
data_response = requests.get(DATA_API_URL)

st.title("Predicted vs Actual Traffic Volume")

if data_response.status_code == 200:
    # Create DataFrame
    data = pd.DataFrame(data_response.json())
    data['datetime'] = pd.to_datetime(data['datetime'])

    # Sidebar filters
    st.sidebar.title("Filters")
    
    # Create location name mapping
    location_names = {}
    for crossroad_id in data['location_id'].unique():
        location_api_url = f"{LOCATION_API_BASE_URL}/{crossroad_id}/"
        location_response = requests.get(location_api_url)
        if location_response.status_code == 200:
            location_names[location_response.json().get("name")] = crossroad_id

    # Create multiselect with location names
    selected_names = st.sidebar.multiselect(
        "Select Locations",
        options=list(location_names.keys()),
        default=list(location_names.keys())[:2]
    )
    
    # Convert selected names to location IDs
    selected_ids = [location_names[name] for name in selected_names]
    
    threshold = st.sidebar.slider("Error Threshold (%)", 0, 100, 20, step=5) / 100

    # Rest of the code remains the same...
    filtered_data = data[data['location_id'].isin(selected_ids)]

    # Model metrics in sidebar
    st.sidebar.title("Model Metrics")
    if not filtered_data.empty:
        mae = abs(filtered_data['actual_value'] - filtered_data['predicted_value']).mean()
        rmse = ((filtered_data['actual_value'] - filtered_data['predicted_value']) ** 2).mean() ** 0.5
        mape = (abs(filtered_data['actual_value'] - filtered_data['predicted_value']) / filtered_data['actual_value']).mean() * 100
        st.sidebar.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
        st.sidebar.metric("Root Mean Square Error (RMSE)", f"{rmse:.2f}")
        st.sidebar.metric("Mean Absolute Percentage Error (MAPE)", f"{mape:.2f}%")

    # Collapsible data table
    with st.expander("📊 Show Data Table"):
        st.dataframe(filtered_data, use_container_width=True)

    # Graphs section
    st.subheader("📈 Predicted vs Actual Graphs")
    for crossroad_id in selected_ids:
        location_api_url = f"{LOCATION_API_BASE_URL}/{crossroad_id}/"
        location_response = requests.get(location_api_url)
        
        selected_location_name = location_response.json().get("name", "Unknown") if location_response.status_code == 200 else "Unknown"
        
        st.markdown(f"#### Location: {selected_location_name}")
        loc_data = filtered_data[filtered_data['location_id'] == crossroad_id]

        if not loc_data.empty:
            loc_data['relative_error'] = abs(
                loc_data['actual_value'] - loc_data['predicted_value']
            ) / loc_data['actual_value']

            fig, ax = plt.subplots(figsize=(15, 8))
            ax.plot(loc_data['datetime'], loc_data['predicted_value'], 
                   label="Predicted", color="#3498DB", linewidth=2)
            ax.plot(loc_data['datetime'], loc_data['actual_value'], 
                   label="Actual", color="#E57373", linestyle="--", linewidth=2)
            ax.fill_between(
                loc_data['datetime'],
                loc_data['actual_value'],
                loc_data['predicted_value'],
                where=loc_data['relative_error'] > threshold,
                color="#FFE0B2",
                alpha=0.3,
                label="Significant Deviation"
            )
            ax.set_title(f"Predicted vs Actual for Location: {selected_location_name}", 
                        fontsize=18, pad=20)
            ax.set_xlabel("Time", fontsize=14)
            ax.set_ylabel("Traffic Volume", fontsize=14)
            ax.legend(fontsize=12)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            plt.xticks(rotation=45)
            ax.grid(visible=True, linestyle="--", alpha=0.5)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info(f"No data available for Location: {selected_location_name}")

else:
    st.error("Failed to fetch data from API.")

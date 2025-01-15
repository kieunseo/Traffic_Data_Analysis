import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime

st.set_page_config(
    page_title="Traffic Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_traffic_data():
    try:
        # 교통 데이터 가져오기
        response = requests.get("http://localhost:8000/traffic-data")
        if response.status_code != 200:
            st.error(f"Failed to fetch traffic data: {response.status_code}")
            return pd.DataFrame(), pd.DataFrame()
        
        df = pd.DataFrame(response.json())

        # 각 location_id에 대한 교차로 정보 가져오기
        locations_data = []
        for location_id in df['location_id'].unique():
            loc_response = requests.get(f"http://localhost:8000/location/{location_id}")
            if loc_response.status_code == 200:
                locations_data.append(loc_response.json())
                
        locations_df = pd.DataFrame(locations_data)

        # datetime 컬럼 처리
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['date'] = df['datetime'].dt.date  # 날짜만 추출
        df['time'] = df['datetime'].dt.strftime('%H:%M')  # 시간만 추출
        return df, locations_df

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame(), pd.DataFrame()



# 데이터 로드
df, locations_df = load_traffic_data()

if not df.empty and not locations_df.empty:
    st.title("Historical Traffic Analysis")
    
    # Sidebar filters
    st.sidebar.title("Filters")
    
    locations_df = locations_df.sort_values(by='id').reset_index(drop=True)

    # Location filter 부분 수정
    if not df.empty and not locations_df.empty:
        selected_location = st.sidebar.selectbox(
        "Select Location",
        options=locations_df['id'].tolist(),
        format_func=lambda x: f"{x}. {locations_df[locations_df['id']==x]['name'].iloc[0]}"
    )
    
    # Date filter
    available_dates = sorted(df['datetime'].dt.date.unique())
    selected_date = st.sidebar.selectbox(
        "Select Date",
        options=available_dates,
        format_func=lambda x: x.strftime('%Y-%m-%d')
    )

    # Filter data for selected location and date
    daily_data = df[
        (df['datetime'].dt.date == selected_date) & 
        (df['location_id'] == selected_location)
    ]

    # Calculate metrics by vehicle type
    lane_columns = [f'lane_{i}' for i in range(1, 13)]
    
    # 시간대별로 데이터 그룹화 및 합계 계산
    hourly_traffic = daily_data.groupby('time')[lane_columns].sum()

    # 시간대별 총합 계산
    time_traffic = hourly_traffic.sum(axis=1)  # 시간대별 합계

    # 총합, 최대값, 평균 계산
    total_traffic = time_traffic.sum()        # 전체 교통량 합계
    max_traffic = time_traffic.max()          # 피크 시간대 최대 교통량
    avg_traffic = time_traffic.mean()         # 시간대별 평균 교통량
    
    # Top row - Key metrics
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.metric("Total Volume", f"{int(total_traffic):,}", "All Lanes")
    with metric_cols[1]:
        st.metric("Peak Hour Volume", f"{int(max_traffic):,}", "Maximum")
    with metric_cols[2]:
        st.metric("Average Volume", f"{int(avg_traffic):,}", "Per Hour")
    with metric_cols[3]:
        st.metric("Vehicle Types", "3", "Bus, Truck, Car")

    # Main content - 2x2 grid
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.subheader("Hourly Traffic Pattern")
        # Time series with moving average
        hourly_avg_data = daily_data.groupby('time')[lane_columns].sum().mean(axis=1).reset_index()
        hourly_avg_data.columns = ['time', 'average_volume']  # 컬럼명 지정
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hourly_avg_data['time'],
            y=hourly_avg_data['average_volume'],
            name='Traffic Volume',
            line=dict(color='#3498DB', width=2)
        ))
        # Add moving average
        fig.add_trace(go.Scatter(
            x=hourly_avg_data['time'],
            y=hourly_avg_data['average_volume'].rolling(3).mean(),
            name='Moving Average (3-period)',
            line=dict(color='#E57373', dash='dash')
        ))
        fig.update_layout(
            title="Hourly Traffic Pattern Analysis",
            xaxis=dict(
            title="Time",
            tickangle=45),
            yaxis=dict(
            title="Volume"),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Vehicle Type Distribution")
        # Enhanced pie chart with percentage labels
        vehicle_data = daily_data.groupby('vehicle_type')[lane_columns].sum().sum(axis=1)
        total = vehicle_data.sum()
        vehicle_pct = (vehicle_data / total * 100).round(1)
        
        # 세련된 파스텔톤 색상 조합
        custom_colors = ['#EC8305',   
                        '#DBD3D3',       
                        '#024CAA']     
        
        fig = px.pie(
            values=vehicle_data,
            names=vehicle_data.index,
            title="Traffic Composition",
            hover_data=[vehicle_pct],
            labels={'value': 'Volume', 'label': 'Vehicle Type'}
        )
        
        fig.update_traces(
            textinfo='percent+label',
            marker=dict(colors=custom_colors)
        )
        
        st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.subheader("Peak Hour Analysis")
        
        # 시간대별 총 교통량 계산
        hourly_by_type = daily_data.groupby('time')[lane_columns].sum().sum(axis=1).reset_index()
        hourly_by_type.columns = ['time', 'volume']
        
        # 피크시간 표시를 위한 데이터 처리
        max_volume = hourly_by_type['volume'].max()
        avg_volume = hourly_by_type['volume'].mean()
        
        # 피크시간 시각화
        fig = px.bar(
            hourly_by_type,
            x='time',
            y='volume',
            title="Peak Hour Distribution",
            labels={'volume': 'Traffic Volume', 'time': 'Time'},
            text='volume'
        )
        
        # HEX 코드를 사용한 막대 색상 설정
        fig.update_traces(
            marker_color=[
                '#FF8343' if x >= max_volume * 0.9  # 피크시간 (최대값의 90% 이상)
                else '#179BAE' if x >= avg_volume  # 평균 이상
                else '#F1DEC6'  # 평균 미만
                for x in hourly_by_type['volume']
            ],
            textposition='outside',
            texttemplate='%{text:,.0f}'
        )
        
        # 평균선 추가
        fig.add_hline(
            y=avg_volume,
            line_dash="dash",
            line_color="#4158A6",
            line_width=1.5
        )
        
        # 레이아웃 설정
        fig.update_layout(
            xaxis=dict(
                tickangle=45,
                tickmode='array',
                ticktext=hourly_by_type['time'],
                tickvals=hourly_by_type['time']
            ),
            yaxis_title="Traffic Volume",
            showlegend=False,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)


    with col4:
        st.subheader("Trend Analysis")
        # Date comparison with trend line
        trend_data = df[
            df['location_id'] == selected_location
        ].groupby('date')[lane_columns].sum().sum(axis=1).reset_index()
        
        fig = px.line(
            trend_data,
            x='date',
            y=0,
            title=f"Traffic Volume Trend at {locations_df[locations_df['id']==selected_location]['name'].iloc[0]}",
            labels={'0': 'Total Volume', 'date': 'Date'}
        )
        # Remove rolling average since we only have 4 specific dates
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Total Traffic Volume",
            showlegend=True,
            hovermode='x unified'
        )
        # Add markers to show specific data points clearly
        fig.update_traces(
        mode='lines+markers',
        line=dict(color='#3498DB', width=2),     
        marker=dict(
            color='#3498DB',                      
            size=8,                               # 마커 크기
            line=dict(color='#FFFFFF', width=1)   # 마커 테두리
        )
    )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.warning("No data available. Please check the API connection.")

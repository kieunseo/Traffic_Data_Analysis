
import streamlit as st
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

# 데이터 생성
data = {
    "교차로명": ["공촌3교4거리", "봉수교3거리", "송도1교4거리", "송도2교4거리", 
               "송도3교4거리", "송도4교4거리", "송도5교4거리", "제1청라4거리", 
               "제2청라4거리", "제3청라4거리"],
    "latitude": [37.53601947, 37.52992557, 37.39543406, 37.39930082, 37.40863427, 
                 37.38716997, 37.38479745, 37.53420767, 37.53605851, 37.53421653],
    "longitude": [126.6319093, 126.6558686, 126.6593195, 126.6569358, 126.6464149, 
                  126.6840014, 126.6981975, 126.649533, 126.6429517, 126.6498169]
}
df = pd.DataFrame(data)

st.set_page_config(page_title="Location Dashboard", page_icon="🌐", layout="wide")


col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("교차로 선택")
    # 교차로 선택 (셀렉트박스로 변경)
    selected_crossroad = st.selectbox(
        "교차로를 선택하세요:",
        options=df["교차로명"],
        key="crossroad_select"
    )
    
    selected_data = df[df["교차로명"] == selected_crossroad]
    
    # 선택된 교차로 정보를 카드 형태로 표시
    st.info(f"📍 선택된 교차로: {selected_crossroad}")
    st.markdown(
        f"""
        #### 위치 정보
        * 위도: `{selected_data['latitude'].values[0]:.6f}`
        * 경도: `{selected_data['longitude'].values[0]:.6f}`
        """
    )
    
    # 지도 스타일 선택
    map_style = st.radio(
        "지도 스타일",
        ["기본", "지도", "야간"],
        horizontal=True
    )
    
    # 마커 색상 선택
    marker_color = st.color_picker("마커 색상 선택", "#1f77b4")

with col2:
    # Folium 지도 생성
    # Folium 지도 생성 부분만 수정
    tiles_dict = {
    "지도": {
        'tiles': 'OpenStreetMap'
    },
    "기본": {
        'tiles': 'CartoDB positron',
        'attr': '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="http://cartodb.com/attributions">CartoDB</a>'
    },
    "야간": {
        'tiles': 'CartoDB dark_matter',
        'attr': '&copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="http://cartodb.com/attributions">CartoDB</a>'
    }
}

    m = folium.Map(
        location=[selected_data['latitude'].values[0], 
                selected_data['longitude'].values[0]],
        zoom_start=13,
        **tiles_dict[map_style]
    )

    
    # 모든 교차로 마커 추가
    for idx, row in df.iterrows():
        color = marker_color if row["교차로명"] == selected_crossroad else 'gray'
        size = 15 if row["교차로명"] == selected_crossroad else 10
        
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=size,
            popup=row['교차로명'],
            color=color,
            fill=True,
            fill_color=color
        ).add_to(m)
        
        # 선택된 교차로에 레이블 추가
        if row["교차로명"] == selected_crossroad:
            folium.Popup(
                row['교차로명'],
                parse_html=True
            ).add_to(folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=0
            ).add_to(m))
    
    st_folium(m, width=800, height=600)



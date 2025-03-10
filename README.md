# 4지 교차로 12방향 교통량 데이터 신뢰도 검증 시스템

이 프로젝트는 4지 교차로 12방향으로 수집된 교통량 데이터를 기반으로 **CNN-LSTM 모델**을 활용하여 미래 교통량을 예측하고, 예측 결과를 통해 **데이터 신뢰도를 검증**하는 시스템을 구축하는 데 목적이 있습니다.  
**오차율이 20%** 이상 발생하는 구간을 자동으로 식별하고, 해당 구간에 대해 보정 필요성을 안내하여 **계수 인력 투입**을 최소화하고 업무 효율성을 높일 수 있습니다.

## Historical Analysis
<div align="center">
  <img src="https://github.com/user-attachments/assets/d4a86f70-88a0-40f4-aeb4-11b444f9a708" 
       alt="Historical Analysis" width="800" />
</div>
<p align="center">
  과거 교통량 데이터 추이를 시각화한 화면입니다.
</p>

<br/>

## Prediction Dashboard
<div align="center">
  <img src="https://github.com/user-attachments/assets/efc48bbc-e049-4b58-9c7c-609df0c30576" 
       alt="Prediction Dashboard" width="800" />
</div>
<p align="center">
  CNN-LSTM 모델의 교통량 예측값과 실제값 비교를 보여줍니다.
</p>

<br/>

## Location Dashboard
<div align="center">
  <img src="https://github.com/user-attachments/assets/639ae506-70d3-4bdb-959d-77dffd06616a" 
       alt="Location Dashboard" width="800" />
</div>
<p align="center">
  지도 기반으로 교차로 위치를 한눈에 확인할 수 있습니다.
</p>

---

## 목차
1. [프로젝트 개요](#프로젝트-개요)  
2. [분석 설계](#분석-설계)  
3. [프로젝트 구조](#프로젝트-구조)  
4. [데이터 구성](#데이터-구성)  
5. [핵심 기능](#핵심-기능)   

---
## 프로젝트 개요

### 1. 추진 배경
- **AI 검지 정확도 문제**  
  - 당사 CCTV 영상 AI 분석 시스템으로 교통량 데이터를 수집·분석 중, 특정 시간대/환경에서 부정확한 결과가 관찰됨  
- **계수 인력 추가 투입**  
  - 잘못된 검지 데이터를 보완하기 위해 계수 인력을 추가 투입 → 비용 증가 및 업무 효율성 저하

### 2. 필요성 및 목적
- **보정 모델 도입**  
  - 야간, 카메라 이물질, 날씨 등으로 인해 발생하는 검지 오류를 자동으로 보정  
- **데이터 신뢰도 검증**  
  - 예측 결과와 실제 검지 데이터 비교  
  - 오차율 20% 초과 구간 자동 식별  
  - 계수 인력 투입 최소화 및 업무 효율성 향상

### 3. 기대 효과
- **오차율 감소**: 교통량 데이터 품질 제고  
- **업무 효율성 제고**: 자동화된 데이터 검증 → 계수 인력 최소화

---

## 2. 분석 설계

### 1. 요구사항
1. **데이터 신뢰도 검증 시스템 구축**  
   - CNN-LSTM 예측 모델 활용  
   - 오차율 20% 초과 구간 자동 식별 및 보정 필요성 판단  
   - 자동화된 검증 체계로 계수 인력 투입 최소화  
2. **성능 검증 및 모니터링**  
   - 예측값과 실제 검지값 간 오차율 분석  
   - 시간대별 데이터 신뢰도 평가

### 2. 분석 목록
- **데이터 전처리 및 가공**  
  - 시간대 특성(출근/퇴근 등 주기) 반영  
  - 교차로별 위치(위도, 경도) 변수 추가  
  - 데이터 스케일링(StandardScaler)
- **예측 모델 개발**  
  - 3층 CNN → 공간적 특성 추출, LSTM → 시계열 패턴 학습  
  - CNN-LSTM 혼합형 모델
- **성능 평가 및 검증**  
  - 평가 지표: MAE, RMSE, MAPE  
- **전략적 활용**  
  - 모니터링 대시보드 구축  
  - 오차율 모니터링

---

## 3. 프로젝트 구조

```bash
projects/
├── api #  FastAPI
│   ├── data
│       └── preprocessing.py # 데이터 전처리
│   ├── models
│       ├── 0108cnn3model.pth # CNN-LSTM Model 학습 체크포인트
│       └── model.py # CNN-LSTM Model
│   ├── sql_app # 데이터베이스 연동
│       ├── crud.py 
│       ├── database.py 
│       ├── models.py 
│       └── schemas.py 
├── models
├── notebooks         
│   ├── CNN_LSTM.ipynb
│   ├── LSTM.ipynb
│   ├── LTSF_Linear.ipynb       
│   └── ...
├── streamlit_app # streamlit 대시보드 구현
│   ├── pages
│       ├── 1_Historical_Analysis.py # 과거 데이터 분석 기능 제공
│       └── 2_Prediction_Dashboard.py # 실시간 예측 결과를 시각적으로 표출              
│   └── Location_Dashboard.py # 교차로 위치 기반 모니터링
└── main.py                  # 프로젝트 메인 실행 스크립트
```

## 4. 데이터 구성

### 1. 분석 대상
- CCTV 영상 기반 계수 데이터  
  - 4지 교차로(12방향)

### 2. 분석 범위
- **공간적 범위**  
  - 공촌 3교 4거리, 봉수교 3거리, 송도 1 ~ 5교 4거리, 제1 ~ 3 청라4거리 등  
- **시간적 범위**  
  - 2023년 3월 15일, 5월 31일, 9월 6일, 11월 29일  
  - 시간대: 7:30 ~ 9:30, 12:00 ~ 14:00, 17:30 ~ 19:30  
- **데이터 특성**  
  - 출근 및 퇴근 시간대 교통량 증가 패턴 확인  
  - 교차로별 예측 정확도 향상을 위해 위도·경도 변수를 포함

### 3. 전처리 프로세스
1. **데이터 수집 및 CSV 변환**  
   - CCTV 기반 계수 결과를 CSV 파일로 변환  
2. **스케일링(Scaling)**  
   - StandardScaler 적용  
3. **시간 특성 변수를 추가**  
   - 주기성(출·퇴근 등)을 고려한 시간대 특성  
4. **위치 정보 추가**  
   - 위도, 경도 변수를 통해 교차로별 위치 차이를 반영

---

## 핵심 기능

1. **데이터 신뢰도 검증**  
   - CNN-LSTM 예측 모델을 통해 **미래 교통량**을 예측  
   - 예측치와 실제 검지값을 비교하여 **오차율**(MAE, RMSE, MAPE) 계산  
   - **오차율 20%** 초과 구간을 자동 식별하여 보정 필요성 안내  

2. **예측 모델 (CNN-LSTM)**  
   - CNN 3개 층으로 **공간적 특성**을 추출하고,  
   - LSTM으로 **시계열 패턴**을 학습하여 교통량 변동 예측  
   - LSTM, DLinear 모델 등과 성능 비교 후 CNN-LSTM 선정

3. **API 서버 (FastAPI)**  
   - 학습된 CNN-LSTM 모델을 로드하여 실시간 예측값 반환  
   - 예측 결과 MySQL DB에 저장 및 관리  
   - REST API 형태로 대시보드/외부 시스템에 제공  

4. **대시보드 시각화 (Streamlit)**  
   - **Location Dashboard**: 지도 기반 교차로 위치
   - **Historical Analysis**: 과거 데이터 추이 분석 및 통계 그래프  
   - **Prediction Dashboard**: 예측 vs 실제 교통량 비교 시각화  
   - **오차율**: 20% 이상 오차율 발생 시 보정 필요 안내

---

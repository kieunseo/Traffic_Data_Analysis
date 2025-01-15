import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
import os
import joblib

# 데이터 전처리 함수
def preprocess(df):
    # 데이터를 Long Format으로 변환
    df = df.melt(
        id_vars=['date', 'time', 'vehicle', '교차로명', 'latitude', 'longitude'],
        value_vars=[str(i) for i in range(1, 13)],
        var_name='direction',
        value_name='traffic_volume'
    )
    
    # 날짜와 시간 병합 및 정렬
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df = df.sort_values('datetime').reset_index(drop=True)

    # 시간 특성 추가
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['hour_decimal'] = df['hour'] + df['minute'] / 60.0

    # One-hot 인코딩
    df = pd.get_dummies(df, columns=['vehicle'], prefix='vehicle')
    df = pd.get_dummies(df, columns=['direction'], prefix='dir')

    # vehicle_, dir_ 로 시작하는 더미 열만 int로 변환
    dummy_cols = [c for c in df.columns if c.startswith('vehicle_') or c.startswith('dir_')]
    df[dummy_cols] = df[dummy_cols].astype('int')

    # 불필요한 컬럼 제거
    df_pre = df.drop(columns=['date', 'time', 'year', 'hour', 'minute'])

    # 컬럼 순서 재정렬
    columns_order = [
        'datetime', 'month', 'day', 'hour_decimal', '교차로명',
        'latitude', 'longitude', 'vehicle_버스', 'vehicle_대형', 'vehicle_소형',
        'dir_1', 'dir_2', 'dir_3', 'dir_4', 'dir_5', 'dir_6', 'dir_7', 'dir_8',
        'dir_9', 'dir_10', 'dir_11', 'dir_12', 'traffic_volume'
    ]
    columns_order = [col for col in columns_order if col in df_pre.columns]
    df_pre = df_pre[columns_order]

    return df_pre


# 시퀀스 생성 함수
def create_sequences_by_intersection(df, seq_length, features, target):
    sequences = []
    targets = []

    # 교차로별 그룹화
    grouped_intersections = df.groupby('교차로명')

    for intersection, group in grouped_intersections:
        grouped = group.groupby('datetime')
        time_groups = sorted(grouped.groups.keys())

        for i in range(len(time_groups) - seq_length + 1):
            sequence = []
            for j in range(seq_length):
                current_time = time_groups[i + j]
                current_data = grouped.get_group(current_time)[features].values
                mean_data = np.mean(current_data, axis=0)
                sequence.append(mean_data)

            target_time = time_groups[i + seq_length - 1]
            target_data = grouped.get_group(target_time)[target].values
            target_mean = np.mean(target_data)

            sequences.append(sequence)
            targets.append(target_mean)

    return np.array(sequences), np.array(targets)

# Tensor 변환 함수
def make_Tensor(array, device='cpu'):
    return torch.from_numpy(array).float().to(device)

# 전처리 및 시퀀스 생성 통합
def preprocess_and_create_sequences(file_path, seq_length, train_features, train_target, input_features):
    # CSV 파일 읽기
    df = pd.read_csv(file_path)
    print("Step 1: Original DataFrame")
    print(df.head())
    print(df.info())
    
    # 데이터 전처리
    df_preprocessed = preprocess(df)
    print("\nStep 2: Preprocessed DataFrame")
    print(df_preprocessed.head())
    print(df_preprocessed.info())

    # 스케일링
    scaler_X_path = r"/Users/eunseo/Downloads/projects/api/data/datasets/scaler_X.pkl"
    if os.path.exists(scaler_X_path):
        scaler_X = joblib.load(scaler_X_path)
        df_preprocessed[input_features] = scaler_X.transform(df_preprocessed[input_features])
    else:
        scaler_X = StandardScaler()
        df_preprocessed[input_features] = scaler_X.fit_transform(df_preprocessed[input_features])

    # 타겟값 스케일링
    scaler_y = StandardScaler()
    df_preprocessed[train_target[0]] = scaler_y.fit_transform(df_preprocessed[[train_target[0]]])

    print("\nStep 3: Scaled DataFrame")
    print(df_preprocessed.head())
    print("Scaler X details:", scaler_X)
    print("Scaler y details:", scaler_y)

    # 시퀀스 생성
    X, y = create_sequences_by_intersection(df_preprocessed, seq_length, train_features, train_target)
    print("\nStep 4: Sequences and Targets")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print(X.dtype, X.shape)
    print(y.dtype, y.shape)

    # Tensor로 변환
    X_tensor = make_Tensor(X)
    y_tensor = make_Tensor(y)
    print("\nStep 5: Tensor Conversion")
    print("Tensor X shape:", X_tensor.shape)
    print("Tensor y shape:", y_tensor.shape)

    return X_tensor, y_tensor, scaler_X, scaler_y

# 교차로별 매핑 정보 및 결과 생성 함수
def process_predictions(y_test, preds, df_test, seq_length):

    # 예측값과 실제값 역스케일링
    scaler_y = joblib.load("models/scaler_y.pkl")
    y_pred_original = scaler_y.inverse_transform(preds.reshape(-1, 1))
    y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, 1))

    # 테스트 데이터프레임 정리
    df_test['datetime'] = pd.to_datetime(df_test['datetime'])
    df_test = df_test.sort_values(by=['교차로명', 'datetime'])

    # 매핑 정보 추출
    mapping_info = df_test[['datetime', '교차로명']].iloc[seq_length:].reset_index(drop=True)
    mapping_info = mapping_info.drop_duplicates(subset=['datetime', '교차로명']).reset_index(drop=True)

    # 교차로별로 데이터 분리
    unique_intersections = mapping_info['교차로명'].unique()
    num_intersections = len(unique_intersections)
    rows_per_intersection = len(y_test_original) // num_intersections

    # 결과 저장
    results = []
    start_idx = 0
    for i, intersection in enumerate(unique_intersections):
        # 마지막 교차로 처리
        end_idx = len(y_test_original) if i == num_intersections - 1 else start_idx + rows_per_intersection

        # 교차로별 데이터 슬라이싱
        y_pred_intersection = y_pred_original[start_idx:end_idx]
        y_test_intersection = y_test_original[start_idx:end_idx]

        datetime_intersection = mapping_info[mapping_info['교차로명'] == intersection]['datetime']
        datetime_intersection = datetime_intersection.iloc[-rows_per_intersection:] 
    
        # 결과 저장
        results.append({
            "intersection": intersection,
            "dates": datetime_intersection.dt.strftime('%Y-%m-%d %H:%M:%S').tolist(),
            "predicted": y_pred_intersection.flatten().tolist(),
            "actual": y_test_intersection.flatten().tolist()
        })

        # 다음 교차로로 이동
        start_idx = end_idx

    return results
from fastapi import FastAPI, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session
from sqlalchemy import text

# 데이터베이스 및 스키마 관련 모듈
from api.sql_app import crud, models, schemas
from api.sql_app.crud import process_and_store_data
from api.sql_app.database import SessionLocal, engine
from api.sql_app.schemas import DataModel

# 데이터 전처리 및 모델 관련 모듈
from api.data.preprocessing import preprocess_and_create_sequences, process_predictions, preprocess
from api.models.model import CNNLSTM_Model
from typing import List

# 파일 처리 및 유틸리티
import joblib
import pandas as pd
import numpy as np
import torch
import os

models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency Injection 
def get_db():
    db = SessionLocal()
    try : 
        yield db
    finally:
        db.close()


# 모델 로드
MODEL_PATH = r"/Users/eunseo/Downloads/projects/api/models/0108cnn3model.pth"

model = CNNLSTM_Model(
    n_features=19,
    n_hidden=32, # 은닉 노드
    seq_len=4, # 시퀀스 길이
    n_layers=1 # 레이어의 수
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()  # 모델 평가 모드 설정

@app.get("/location/{location_id}/",response_model=schemas.Location)
def get_location(location_id:int, db:Session=Depends(get_db)):
    db_location = crud.get_location(db,location_id =location_id )
    if db_location is None:
        raise HTTPException(status_code=404, detail="Location not found")
    return db_location

@app.post("/location/", response_model=schemas.Location)
def create_location(location: schemas.LocationCreate, db: Session = Depends(get_db)):
    db_location = crud.create_location(db, location)
    if db_location:
        raise HTTPException(status_code=400, detail="Location already registered")
    return crud.create_location(db=db,location=location)

@app.delete("/location/{location_id}/")
def delete_location(location_id: int, db:Session=Depends(get_db)):
    db_location = crud.get_location(db, location_id)
    if db_location is None:
        raise HTTPException(status_code=404, detail="Location not found")
    crud.delete_location(db, db_location)
    return {"message": "Location deleted successfully"}

@app.get("/actual/", response_model=schemas.Actual)
def get_actual(actual_id: int, db:Session=Depends(get_db)):
    db_actual = crud.get_actual(db,actual_id =actual_id )
    if db_actual is None:
        raise HTTPException(status_code=404, detail="Actual not found")
    return db_actual

@app.post("/actual/", response_model=schemas.Actual)
def create_actual(actual: schemas.ActualCreate, db: Session = Depends(get_db)):
    # 기존에 동일한 이름의 Actual 데이터가 있는지 확인
    db_actual = crud.get_actual(db, actual)
    if db_actual:
        raise HTTPException(status_code=400, detail="Actual already registered")

    # 새로운 Actual 데이터 생성
    return crud.create_actual(db=db, actual=actual)

@app.delete("/actual/{actual_id}/")
def delete_actual(actual_id: int, db:Session=Depends(get_db)):
    db_actual = crud.get_actual(db, actual_id)
    if db_actual is None:
        raise HTTPException(status_code=404, detail="Actual not found")
    crud.delete_actual(db, db_actual)
    return {"message": "Actual deleted successfully"}

@app.get("/predicted/", response_model=schemas.Predicted)
def get_predicted(predicted_id: int, db:Session=Depends(get_db)):
    db_predicted = crud.get_predicted(db,predicted_id =predicted_id )
    if db_predicted is None:
        raise HTTPException(status_code=404, detail="Predicted not found")
    return db_predicted

@app.post("/predicted/", response_model=schemas.Predicted)
def create_predicted(predicted: schemas.PredictedCreate, db: Session = Depends(get_db)):
    # 기존에 동일한 Predicted 데이터가 있는지 확인
    db_predicted = crud.get_predicted(db, predicted)
    if db_predicted:
        raise HTTPException(status_code=400, detail="Predicted already registered")

    # 새로운 Predicted 데이터 생성
    return crud.create_predicted(db=db, predicted=predicted)

@app.delete("/predicted/{predicted_id}/")
def delete_predicted(predicted_id: int, db:Session=Depends(get_db)):
    db_predicted = crud.get_predicted(db, predicted_id)
    if db_predicted is None:
        raise HTTPException(status_code=404, detail="Predicted not found")
    crud.delete_predicted(db, db_predicted)
    return {"message": "Predicted deleted successfully"}


@app.post("/process-data/")
async def process_data(file: UploadFile):
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported.")

        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        seq_length = 4
        train_features = ['month', 'day', 'hour_decimal', 'latitude', 'longitude',
                          'vehicle_대형', 'vehicle_소형', 'dir_1', 'dir_2', 'dir_3', 'dir_4',
                          'dir_5', 'dir_6', 'dir_7', 'dir_8', 'dir_9', 'dir_10', 'dir_11', 'dir_12']
        train_target = ['traffic_volume']
        input_features = ['month', 'day', 'hour_decimal', 'latitude', 'longitude']

        # 전처리 수행
        X_train, y_train, scaler_X, scaler_y = preprocess_and_create_sequences(
            file_path, seq_length, train_features, train_target, input_features
        )
        print(f"Shapes - X_train: {X_train.shape}, y_train: {y_train.shape}")

        # 스케일러 및 데이터 저장
        os.makedirs("models", exist_ok=True)  # 저장 폴더가 없으면 생성
        joblib.dump(scaler_X, "models/scaler_X.pkl")
        joblib.dump(scaler_y, "models/scaler_y.pkl")
        np.save("models/X_train.npy", X_train)
        np.save("models/y_train.npy", y_train)

        return {
            "message": "Data processed and saved successfully!",
            "X_train_shape": X_train.shape,
            "y_train_shape": y_train.shape,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")

    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)


# 예측 및 후처리 엔드포인트
@app.post("/predict/")
async def predict(file: UploadFile):
    try:
        # 파일 업로드 처리
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are supported.")

        # 파일 저장 (임시 저장)
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())

        # 데이터 로드
        df = pd.read_csv(file_path)
        print("Step 1: Original DataFrame Loaded")
        print(df)

        # 데이터 전처리
        df_test = preprocess(df)
        print("Step 2: Preprocessed DataFrame Created")
        print(df_test)

        # 시퀀스 길이와 테스트 데이터 설정
        seq_length = 4
        X_test = np.load("models/X_train.npy")  # 테스트 입력 데이터 로드
        y_test = np.load("models/y_train.npy")  # 테스트 실제값 로드

        # 모델 추론
        preds = []
        with torch.no_grad():
            for i in range(len(X_test)):
                model.reset_hidden_state(batch_size=1)
                y_test_pred = model(torch.unsqueeze(torch.tensor(X_test[i]), 0))
                preds.append(y_test_pred.item())

        preds = np.array(preds)

        # 결과 후처리
        results = process_predictions(y_test, preds, df_test, seq_length)
        print("Step 3: Predictions Processed")

        # 임시 파일 삭제
        import os
        os.remove(file_path)

        # 결과를 json_data 변수에 저장
        json_data = {"results": results}

        # 데이터베이스에 결과 저장
        with SessionLocal() as db:
            process_and_store_data(json_data, db)
        print("Step 4: Data stored in database")

        # 결과 반환
        return json_data

    except Exception as e:
        print(f"Error occurred: {str(e)}")  # 로그에 오류 출력
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


def fetch_data(query):
    with engine.connect() as connection:
        result = connection.execute(text(query))
        return result

@app.get("/data", response_model=List[DataModel])
def get_combined_data():
    query = """
        SELECT 
            predicted.location_id,
            predicted.predicted_value,
            predicted.datetime AS datetime,
            actual.actual_value
        FROM predicted
        JOIN actual
        ON predicted.location_id = actual.location_id
        AND predicted.datetime = actual.datetime
    """
    result = fetch_data(query)
    return [
        {
            "location_id": row._mapping["location_id"],
            "predicted_value": row._mapping["predicted_value"],
            "actual_value": row._mapping["actual_value"],
            "datetime": row._mapping["datetime"].strftime("%Y-%m-%d %H:%M:%S"),
        }
        for row in result
    ]


@app.get("/traffic-data")
async def get_traffic_data():
    try:
        query = """
        SELECT location_id, datetime, vehicle_type,
           lane_1, lane_2, lane_3, lane_4, lane_5, lane_6,
           lane_7, lane_8, lane_9, lane_10, lane_11, lane_12
        FROM traffic_data
        ORDER BY datetime;
        """
        df = pd.read_sql(query, engine)
        
        # Convert datetime to string format for JSON serialization
        df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        return df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

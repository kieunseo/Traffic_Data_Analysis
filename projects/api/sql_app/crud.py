from sqlalchemy.orm import Session
from . import models, schemas # 기존에 생성한 모델과 스키마 불러오기
from datetime import datetime

############################ Location ############################
def get_location(db: Session, location_id: int):
    return db.query(models.Location).filter(models.Location.id == location_id).first()

def create_location(db: Session, location:schemas.LocationCreate):
    db_location = models.Location(**location.model_dump())
    db.add(db_location)
    db.commit()
    db.refresh(db_location)
    return db_location

def delete_location(db: Session, location: models.Location):
    db.delete(location)
    db.commit()

############################ Actual ############################

def get_actual(db: Session, actual_id: int):
    return db.query(models.Actual).filter(models.Actual.id == actual_id).first()

def create_actual(db: Session, actual:schemas.Actual):
    db_actual = models.Actual(**actual.model_dump())
    db.add(db_actual)
    db.commit()
    db.refresh(db_actual)
    return db_actual

def delete_actual(db: Session, actual: models.Actual):
    db.delete(actual)
    db.commit()

############################ Predicted ############################

def get_predicted(db: Session, predicted_id: int):
    return db.query(models.Predicted).filter(models.Predicted.id == predicted_id).first()

def create_predicted(db: Session, predicted:schemas.PredictedCreate):
    db_predicted = models.Predicted(**predicted.model_dump())
    db.add(db_predicted)
    db.commit()
    db.refresh(db_predicted)
    return db_predicted

def delete_predicted(db: Session, predicted: models.Predicted):
    db.delete(predicted)
    db.commit()

############################ MySQL DB ############################

def process_and_store_data(json_data: dict, db: Session):
    results = json_data['results']

    for result in results:
        # Location 생성 또는 조회
        location_name = result['intersection']
        location = db.query(models.Location).filter(models.Location.name == location_name).first()
        if not location:
            location = models.Location(name=location_name)
            db.add(location)
            db.flush()

        # Actual 및 Predicted 데이터 생성 및 저장
        for date, actual, predicted in zip(result['dates'], result['actual'], result['predicted']):
            datetime_obj = datetime.strptime(date, '%Y-%m-%d %H:%M:%S')
            
            actual_data = models.Actual(
                location_id=location.id,
                actual_value=actual,
                datetime=datetime_obj
            )
            db.add(actual_data)

            predicted_data = models.Predicted(
                location_id=location.id,
                predicted_value=predicted,
                datetime=datetime_obj
            )
            db.add(predicted_data)

    db.commit()
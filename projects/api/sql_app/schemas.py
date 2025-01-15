# pydantic model은 "스키마"(유효한 데이터 모양)을 어느 정도 정의한다.
# 데이터 생성(Create) 및 API 요청 처리

from pydantic import BaseModel
from typing import List, Optional, Union
from datetime import datetime


class LocationBase(BaseModel):
    name: str
    
class LocationCreate(LocationBase):
    pass

# API에서 데이터를 읽을 때/반환할 때 사용될 모델
class Location(LocationBase):
    id: int

    class config:
        from_attributes = True

class ActualBase(BaseModel):
    actual_value: float
    datetime: datetime
    
class ActualCreate(ActualBase):
    location_id: int

class Actual(ActualBase):
    id: int
    location_id: int

    class Config:
        from_attributes = True


class PredictedBase(BaseModel):
    predicted_value: float
    datetime: datetime
    
class PredictedCreate(PredictedBase):
    location_id: int

class Predicted(PredictedBase):
    id: int
    location_id: int

    class Config:
        from_attributes = True

class DataModel(BaseModel):
    location_id: int
    predicted_value: float
    actual_value: float
    datetime: str 
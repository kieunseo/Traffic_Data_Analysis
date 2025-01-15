from sqlalchemy import Integer, String, ForeignKey, Float, Column, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from .database import Base # database.py에서 생성한 Base import
from datetime import datetime


# SQLAlchemy 모델
# SQLAlchemy는 데이터베이스의 테이블 및 열 정의를 위해 사용

class Location(Base):
    # 해당 모델이 사용할 table 이름 지정
    __tablename__ = "locations"

    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True) # 빈값 허용 x, 고유값만 받겠다.

    # 다른 테이블과의 관계 생성
    predictions = relationship("Predicted", back_populates="location")
    actuals = relationship("Actual", back_populates="location")

class Actual(Base):
    __tablename__ = "actual"

    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True) # autoincrement id 자동 생성
    location_id = Column(Integer, ForeignKey("locations.id"), nullable=False)
    actual_value = Column(Float, nullable=False)
    datetime = Column(DateTime, default=datetime.utcnow, nullable=False) 

    location = relationship("Location", back_populates="actuals")

class Predicted(Base):
    __tablename__ = "predicted"

    id = Column(Integer, primary_key=True, nullable=False, autoincrement=True) # autoincrement id 자동 생성
    location_id = Column(Integer, ForeignKey("locations.id"), nullable=False)
    predicted_value = Column(Float, nullable=False)
    datetime = Column(DateTime, default=datetime.utcnow, nullable=False) 

    location = relationship("Location", back_populates="predictions")
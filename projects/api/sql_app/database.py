from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from dotenv import load_dotenv
import os

load_dotenv()
user = os.getenv("DB_USER")     
passwd = os.getenv("DB_PASSWD") 
host = os.getenv("DB_HOST")     
port = os.getenv("DB_PORT")     
db = os.getenv("DB_NAME")       

# 1. SQLAlchemy 사용할 DB URL 생성하기
# mysql db에 연결
DB_URL = f"mysql+pymysql://{user}:{passwd}@{host}:{port}/{db}?charset=utf8mb4"
# Postgresql : f'postgresql://{user}:{passwd}@{host}:{port}/{db}'
# SQLite : f'sqlite://{/dbname.db}'


# 2. 첫 번째 단계는 SQLAlchemy "엔진"을 만드는 것입니다.
engine = create_engine(DB_URL, echo=True)

# 3. DB 세션 생성하기
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 4. Base class 생성
Base = declarative_base()

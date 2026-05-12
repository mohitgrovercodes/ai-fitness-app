from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from pathlib import Path

from app.core.config import settings
 
# Use the dynamic URL from config.py (defaults to MySQL)
SQLALCHEMY_DATABASE_URL = settings.MYSQL_URL
 
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    pool_pre_ping=True, # Recommended for MySQL to handle dropped connections
    pool_recycle=3600
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

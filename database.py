"""
Database configuration and session management for SQLAlchemy.
Handles PostgreSQL connection, session factory creation, and dependency injection for FastAPI routes.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv


# Database credentials from environment variables
# These should be defined in a .env file in the project root
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")

# SQLALCHEMY_DATABASE_URL = "sqlite:///./sql_app.db"
SQLALCHEMY_DATABASE_URL = f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:5432/{POSTGRES_DB}"

engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create session factory
# autocommit=False: Changes must be explicitly committed
# autoflush=False: Changes won't be automatically flushed to the database
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# FastAPI dependency for database sessions
# Usage: Add 'db: Session = Depends(get_db)' to route parameters
# This ensures each request gets its own database session and
# the session is properly closed when the request is complete
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

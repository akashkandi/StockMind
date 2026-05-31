import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Float, Text, DateTime, Integer, JSON
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

load_dotenv()

# --- Database connection ---
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://stockmind:stockmind123@localhost:5432/stockmind"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# --- Database Models ---
class ResearchReport(Base):
    __tablename__ = "research_reports"

    id = Column(Integer, primary_key=True, index=True)
    company = Column(String, index=True)
    ticker = Column(String)
    recommendation = Column(String)  # BUY / HOLD / SELL
    report_text = Column(Text)
    current_price = Column(Float, nullable=True)
    market_cap = Column(String, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    sentiment_label = Column(String, nullable=True)
    news_summary = Column(Text, nullable=True)
    financial_summary = Column(Text, nullable=True)
    sec_summary = Column(Text, nullable=True)
    top_risks = Column(JSON, nullable=True)
    headlines = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# --- Create all tables ---
def init_db():
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")


# --- Dependency for FastAPI ---
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    print("✅ Database initialized successfully")
import os
import json
import asyncio
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from concurrent.futures import ThreadPoolExecutor
import threading

from database import get_db, init_db, ResearchReport
from news_agent import run_news_agent
from financials_agent import run_financials_agent
from sec_agent import run_sec_agent
from sentiment_agent import run_sentiment_agent
from graph import supervisor_node, ResearchState

load_dotenv()

app = FastAPI(title="StockMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize DB on startup
@app.on_event("startup")
async def startup():
    init_db()


# --- Request/Response Models ---
class ResearchRequest(BaseModel):
    company: str


class ReportResponse(BaseModel):
    id: int
    company: str
    ticker: Optional[str]
    recommendation: str
    report_text: str
    current_price: Optional[float]
    market_cap: Optional[str]
    sentiment_score: Optional[float]
    sentiment_label: Optional[str]
    headlines: Optional[List[str]]
    top_risks: Optional[List[str]]
    created_at: datetime

    class Config:
        from_attributes = True


# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}

    async def connect(self, websocket: WebSocket, research_id: str):
        await websocket.accept()
        self.active_connections[research_id] = websocket

    def disconnect(self, research_id: str):
        if research_id in self.active_connections:
            del self.active_connections[research_id]

    async def send_update(self, research_id: str, message: dict):
        if research_id in self.active_connections:
            try:
                await self.active_connections[research_id].send_text(
                    json.dumps(message)
                )
            except Exception:
                self.disconnect(research_id)


manager = ConnectionManager()


# --- Research Pipeline with WebSocket updates ---
async def run_research_with_updates(company: str, research_id: str, db: Session):
    """Run the full research pipeline and stream updates via WebSocket"""

    async def update(agent: str, status: str, message: str = ""):
        await manager.send_update(research_id, {
            "agent": agent,
            "status": status,
            "message": message
        })

    try:
        # Notify start
        await update("system", "started", f"Starting research on {company}")

        results = {}
        errors = {}

        # Run News, Financials, SEC in parallel using threads
        await update("news", "running", "Searching for recent news...")
        await update("financials", "running", "Fetching financial data...")
        await update("sec", "running", "Reading SEC filings...")

        loop = asyncio.get_event_loop()

        with ThreadPoolExecutor(max_workers=3) as executor:
            news_future = loop.run_in_executor(executor, run_news_agent, company)
            financials_future = loop.run_in_executor(executor, run_financials_agent, company)
            sec_future = loop.run_in_executor(executor, run_sec_agent, company)

            # Wait for all three
            news_result, financials_result, sec_result = await asyncio.gather(
                news_future, financials_future, sec_future,
                return_exceptions=True
            )

        # Handle results
        results["news"] = news_result if not isinstance(news_result, Exception) else None
        results["financials"] = financials_result if not isinstance(financials_result, Exception) else None
        results["sec"] = sec_result if not isinstance(sec_result, Exception) else None

        await update("news", "complete", "News research done")
        await update("financials", "complete", "Financial data fetched")
        await update("sec", "complete", "SEC filing analyzed")

        # Run Sentiment with news headlines
        await update("sentiment", "running", "Analyzing market sentiment with FinBERT...")
        headlines = results["news"].headlines if results.get("news") else []
        sentiment_result = await loop.run_in_executor(
            None, run_sentiment_agent, company, headlines
        )
        results["sentiment"] = sentiment_result
        await update("sentiment", "complete", "Sentiment analysis done")

        # Run Supervisor
        await update("supervisor", "running", "Synthesizing research into report...")
        state = ResearchState(
            company=company,
            news=results.get("news"),
            financials=results.get("financials"),
            sec=results.get("sec"),
            sentiment=results.get("sentiment"),
            report="",
            recommendation=""
        )

        final_state = await loop.run_in_executor(None, supervisor_node, state)
        await update("supervisor", "complete", "Report generated")

        # Save to database
        report = ResearchReport(
            company=company,
            ticker=results["financials"].ticker if results.get("financials") else None,
            recommendation=final_state["recommendation"],
            report_text=final_state["report"],
            current_price=results["financials"].current_price if results.get("financials") else None,
            market_cap=results["financials"].market_cap if results.get("financials") else None,
            sentiment_score=results["sentiment"].sentiment_score if results.get("sentiment") else None,
            sentiment_label=results["sentiment"].overall_sentiment if results.get("sentiment") else None,
            news_summary=results["news"].summary if results.get("news") else None,
            financial_summary=results["financials"].financial_summary if results.get("financials") else None,
            sec_summary=results["sec"].sec_summary if results.get("sec") else None,
            top_risks=results["sec"].top_risks if results.get("sec") else None,
            headlines=results["news"].headlines if results.get("news") else None,
        )

        db.add(report)
        db.commit()
        db.refresh(report)

        # Send final report
        await manager.send_update(research_id, {
            "agent": "system",
            "status": "complete",
            "report_id": report.id,
            "recommendation": final_state["recommendation"],
            "report": final_state["report"]
        })

    except Exception as e:
        await manager.send_update(research_id, {
            "agent": "system",
            "status": "error",
            "message": str(e)
        })


# --- API Endpoints ---
@app.get("/")
def root():
    return {"status": "running", "message": "StockMind API"}


@app.post("/research")
async def start_research(request: ResearchRequest, db: Session = Depends(get_db)):
    """Start a research job — returns research_id for WebSocket connection"""
    research_id = f"{request.company}_{datetime.utcnow().timestamp()}"
    return {
        "research_id": research_id,
        "company": request.company,
        "message": f"Connect to WebSocket at /ws/{research_id} to receive updates"
    }


@app.websocket("/ws/{research_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    research_id: str,
    db: Session = Depends(get_db)
):
    """WebSocket endpoint — streams live agent progress"""
    await manager.connect(websocket, research_id)
    company = research_id.split("_")[0]

    try:
        # Start research in background
        asyncio.create_task(
            run_research_with_updates(company, research_id, db)
        )

        # Keep connection alive until research is done
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=120)
            except asyncio.TimeoutError:
                break

    except WebSocketDisconnect:
        manager.disconnect(research_id)


@app.get("/reports", response_model=List[ReportResponse])
def get_reports(db: Session = Depends(get_db)):
    """Get all research reports"""
    reports = db.query(ResearchReport).order_by(
        ResearchReport.created_at.desc()
    ).all()
    return reports


@app.get("/reports/{report_id}", response_model=ReportResponse)
def get_report(report_id: int, db: Session = Depends(get_db)):
    """Get a specific research report"""
    report = db.query(ResearchReport).filter(
        ResearchReport.id == report_id
    ).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    return report


@app.get("/reports/company/{company_name}", response_model=List[ReportResponse])
def get_reports_by_company(company_name: str, db: Session = Depends(get_db)):
    """Get all reports for a specific company"""
    reports = db.query(ResearchReport).filter(
        ResearchReport.company.ilike(f"%{company_name}%")
    ).order_by(ResearchReport.created_at.desc()).all()
    return reports
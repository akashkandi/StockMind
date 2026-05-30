import os
import requests
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv()

# --- Structured output schema ---
class SECOutput(BaseModel):
    company: str = Field(description="Company name")
    ticker: str = Field(description="Stock ticker symbol")
    filing_type: str = Field(description="Type of filing analyzed e.g. 10-K or 10-Q")
    filing_date: str = Field(description="Date of the filing")
    top_risks: List[str] = Field(description="Top 5 risk factors the company disclosed")
    recent_events: List[str] = Field(description="Key recent business events from the filing")
    sec_summary: str = Field(description="2-3 sentence summary of key SEC filing findings")


def get_cik_number(ticker: str) -> str:
    """Get the CIK number for a ticker from SEC EDGAR"""
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2024-01-01&enddt=2025-12-31&forms=10-K"
    
    # Use company tickers lookup
    headers = {"User-Agent": "financial-research-agent contact@example.com"}
    response = requests.get(
        "https://www.sec.gov/files/company_tickers.json",
        headers=headers
    )
    
    if response.status_code != 200:
        return None
    
    data = response.json()
    ticker_upper = ticker.upper()
    
    for key, company in data.items():
        if company.get("ticker") == ticker_upper:
            cik = str(company.get("cik_str")).zfill(10)
            return cik
    
    return None


def get_latest_filing(cik: str, form_type: str = "10-K") -> dict:
    """Get the latest filing of a given type for a company"""
    headers = {"User-Agent": "financial-research-agent contact@example.com"}
    
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        return None
    
    data = response.json()
    filings = data.get("filings", {}).get("recent", {})
    
    forms = filings.get("form", [])
    dates = filings.get("filingDate", [])
    accession_numbers = filings.get("accessionNumber", [])
    
    # Find the latest 10-K or 10-Q
    for i, form in enumerate(forms):
        if form == form_type:
            return {
                "form_type": form,
                "filing_date": dates[i],
                "accession_number": accession_numbers[i].replace("-", ""),
                "cik": cik
            }
    
    return None


def get_filing_text(cik: str, accession_number: str) -> str:
    """Get the actual text content of a filing"""
    headers = {"User-Agent": "financial-research-agent contact@example.com"}
    
    # Get filing index
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number}/{accession_number}-index.json"
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        # Try alternative approach — search for the filing
        return None
    
    data = response.json()
    files = data.get("directory", {}).get("item", [])
    
    # Find the main document
    for file in files:
        name = file.get("name", "")
        if name.endswith(".htm") or name.endswith(".txt"):
            if "10k" in name.lower() or "10-k" in name.lower() or name.startswith("d"):
                doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_number}/{name}"
                doc_response = requests.get(doc_url, headers=headers)
                if doc_response.status_code == 200:
                    # Return first 8000 chars — enough for risk factors
                    return doc_response.text[:8000]
    
    return None


def run_sec_agent(company: str, ticker: str = None) -> SECOutput:
    """Fetch and analyze SEC filings for a company"""
    print(f"\n📋 SEC Agent starting research on: {company}")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    # Step 1 — Get ticker if not provided
    if not ticker:
        response = llm.invoke(
            f"What is the stock ticker symbol for {company} on US stock exchanges? "
            f"Reply with ONLY the ticker symbol, nothing else."
        )
        ticker = response.content.strip().upper()

    print(f"  → Ticker: {ticker}")

    # Step 2 — Get CIK number
    cik = get_cik_number(ticker)
    if not cik:
        print(f"  ⚠️  Could not find CIK for {ticker}, using LLM knowledge")
        cik = None

    print(f"  → CIK: {cik}")

    # Step 3 — Get latest filing info
    filing_info = None
    filing_text = None

    if cik:
        filing_info = get_latest_filing(cik, "10-K")
        if not filing_info:
            filing_info = get_latest_filing(cik, "10-Q")

    if filing_info:
        print(f"  → Found {filing_info['form_type']} filed on {filing_info['filing_date']}")
        filing_text = get_filing_text(cik, filing_info["accession_number"])

    # Step 4 — Use LLM to extract insights
    llm_structured = llm.with_structured_output(SECOutput)

    if filing_text:
        prompt = f"""Analyze this SEC filing excerpt for {company} ({ticker}) and extract key information.

Filing Type: {filing_info['form_type']}
Filing Date: {filing_info['filing_date']}

Filing Content:
{filing_text[:4000]}

Extract:
1. Top 5 risk factors the company disclosed
2. Key recent business events mentioned
3. A 2-3 sentence summary of the most important findings

Focus on risks, legal issues, business challenges, and significant events."""
    else:
        # Fallback to LLM knowledge if SEC fetch fails
        print(f"  → Using LLM knowledge for SEC analysis")
        prompt = f"""Based on your knowledge of {company} ({ticker}), provide an analysis of what their recent SEC filings likely contain.

Extract typical information found in 10-K filings:
1. Top 5 common risk factors for this type of company
2. Key recent business events
3. A 2-3 sentence summary

Note: This is based on general knowledge, not a live SEC filing."""

    result = llm_structured.invoke(prompt)
    print(f"✅ SEC Agent complete for {company}")
    return result


# Test it directly
if __name__ == "__main__":
    result = run_sec_agent("Apple", "AAPL")

    print("\n" + "="*50)
    print(f"Company: {result.company} ({result.ticker})")
    print(f"Filing: {result.filing_type} — {result.filing_date}")
    print(f"\nTop Risks:")
    for i, risk in enumerate(result.top_risks, 1):
        print(f"  {i}. {risk}")
    print(f"\nRecent Events:")
    for event in result.recent_events:
        print(f"  - {event}")
    print(f"\nSEC Summary: {result.sec_summary}")
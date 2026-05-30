import os
import json
from dotenv import load_dotenv
from typing import Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import yfinance as yf
import pandas as pd

load_dotenv()

# --- Structured output schema ---
class FinancialsOutput(BaseModel):
    company: str = Field(description="Company name")
    ticker: str = Field(description="Stock ticker symbol")
    current_price: float = Field(description="Current stock price in USD")
    market_cap: str = Field(description="Market capitalization (formatted)")
    pe_ratio: Optional[float] = Field(description="Price to earnings ratio")
    revenue_ttm: str = Field(description="Revenue trailing twelve months (formatted)")
    profit_margin: Optional[float] = Field(description="Profit margin as percentage")
    debt_to_equity: Optional[float] = Field(description="Debt to equity ratio")
    week_52_high: float = Field(description="52 week high price")
    week_52_low: float = Field(description="52 week low price")
    analyst_recommendation: str = Field(description="Analyst recommendation: buy/hold/sell")
    financial_summary: str = Field(description="2-3 sentence summary of financial health")
    key_concerns: list[str] = Field(description="List of 2-3 financial concerns or risks")


def get_ticker_symbol(company_name: str) -> str:
    """Use LLM to get the correct ticker symbol for a company"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.invoke(
        f"What is the stock ticker symbol for {company_name} on US stock exchanges? "
        f"Reply with ONLY the ticker symbol, nothing else. Example: AAPL"
    )
    return response.content.strip().upper()


def format_large_number(num) -> str:
    """Format large numbers to readable format"""
    if num is None or pd.isna(num):
        return "N/A"
    if num >= 1_000_000_000_000:
        return f"${num/1_000_000_000_000:.2f}T"
    elif num >= 1_000_000_000:
        return f"${num/1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"${num/1_000_000:.2f}M"
    else:
        return f"${num:,.0f}"


def run_financials_agent(company: str) -> FinancialsOutput:
    """Fetch and analyze financial data for a company"""
    print(f"\n💰 Financials Agent starting research on: {company}")

    # Step 1 — Get ticker symbol
    ticker_symbol = get_ticker_symbol(company)
    print(f"  → Ticker symbol: {ticker_symbol}")

    # Step 2 — Fetch data from yfinance
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info

    # Step 3 — Extract key metrics
    current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
    market_cap = format_large_number(info.get("marketCap"))
    pe_ratio = info.get("trailingPE")
    revenue = format_large_number(info.get("totalRevenue"))
    profit_margin = info.get("profitMargins")
    if profit_margin:
        profit_margin = round(profit_margin * 100, 2)
    debt_to_equity = info.get("debtToEquity")
    week_52_high = info.get("fiftyTwoWeekHigh", 0)
    week_52_low = info.get("fiftyTwoWeekLow", 0)
    analyst_rec = info.get("recommendationKey", "N/A").lower()

    # Step 4 — Format raw data for LLM analysis
    raw_data = f"""
Company: {company} ({ticker_symbol})
Current Price: ${current_price}
Market Cap: {market_cap}
P/E Ratio: {pe_ratio}
Revenue (TTM): {revenue}
Profit Margin: {profit_margin}%
Debt to Equity: {debt_to_equity}
52 Week High: ${week_52_high}
52 Week Low: ${week_52_low}
Analyst Recommendation: {analyst_rec}
Business Summary: {info.get('longBusinessSummary', 'N/A')[:500]}
    """

    print(f"  → Financial data fetched successfully")

    # Step 5 — Use LLM to analyze and generate summary
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    llm_structured = llm.with_structured_output(FinancialsOutput)

    result = llm_structured.invoke(
        f"""Analyze this financial data and provide a structured assessment.

{raw_data}

Based on this data:
1. Write a 2-3 sentence summary of the company's financial health
2. Identify 2-3 key financial concerns or risks
3. Fill in all the structured fields with the data provided

Be specific and factual. Use the actual numbers provided."""
    )

    print(f"✅ Financials Agent complete for {company}")
    return result


# Test it directly
if __name__ == "__main__":
    result = run_financials_agent("Microsoft")

    print("\n" + "="*50)
    print(f"Company: {result.company} ({result.ticker})")
    print(f"Current Price: ${result.current_price}")
    print(f"Market Cap: {result.market_cap}")
    print(f"P/E Ratio: {result.pe_ratio}")
    print(f"Revenue (TTM): {result.revenue_ttm}")
    print(f"Profit Margin: {result.profit_margin}%")
    print(f"Debt/Equity: {result.debt_to_equity}")
    print(f"52W High/Low: ${result.week_52_high} / ${result.week_52_low}")
    print(f"Analyst Rec: {result.analyst_recommendation}")
    print(f"\nFinancial Summary: {result.financial_summary}")
    print(f"\nKey Concerns:")
    for c in result.key_concerns:
        print(f"  - {c}")
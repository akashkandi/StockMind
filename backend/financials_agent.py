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
    current_price: float = Field(description="Current stock price or index level")
    market_cap: str = Field(description="Market capitalization or total market cap of constituents")
    pe_ratio: Optional[float] = Field(description="Price to earnings ratio")
    revenue_ttm: str = Field(description="Revenue TTM for companies, YTD performance for indices")
    profit_margin: Optional[float] = Field(description="Profit margin as percentage, or dividend yield for indices")
    debt_to_equity: Optional[float] = Field(description="Debt to equity ratio")
    week_52_high: float = Field(description="52 week high price or index level")
    week_52_low: float = Field(description="52 week low price or index level")
    analyst_recommendation: str = Field(description="Analyst recommendation: buy/hold/sell for companies, index for indices")
    financial_summary: str = Field(description="2-3 sentence summary of financial health")
    key_concerns: list[str] = Field(description="List of 2-3 financial concerns or risks")


# Known index keywords
INDEX_KEYWORDS = ["s&p", "s&p500", "sp500", "dow jones", "nasdaq", "ftse",
                  "nikkei", "russell", "etf", "vix", "index"]


def is_market_index(company: str) -> bool:
    """Check if input is a market index rather than a company"""
    return any(kw in company.lower() for kw in INDEX_KEYWORDS)


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
    """Fetch and analyze financial data for a company or index"""
    print(f"\n💰 Financials Agent starting research on: {company}")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    llm_structured = llm.with_structured_output(FinancialsOutput)

    # --- Handle market indices differently ---
    if is_market_index(company):
        print(f"  → Detected market index — using index-specific analysis")

        result = llm_structured.invoke(
            f"""Provide current financial data for the market index: {company}

Use your knowledge of current index levels and key metrics.
For indices provide:
- current_price: current index level (e.g. 5200 for S&P 500)
- market_cap: total market cap of all constituents (e.g. "$40T")
- pe_ratio: average P/E ratio of constituents
- revenue_ttm: YTD performance formatted as "YTD: +X%" 
- profit_margin: average dividend yield of constituents
- debt_to_equity: null (not applicable)
- week_52_high: 52 week high level
- week_52_low: 52 week low level
- analyst_recommendation: "index"
- financial_summary: 2-3 sentences on current index health and performance
- key_concerns: 2-3 current macro risks affecting the index
- ticker: the index ticker (e.g. ^GSPC for S&P 500)

Be specific with real current numbers. This is a market index, not a company."""
        )

        print(f"✅ Financials Agent complete for {company}")
        return result

    # --- Handle regular companies ---
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
    # Test with a company
    print("Testing with Microsoft...")
    result = run_financials_agent("Microsoft")
    print(f"Company: {result.company} ({result.ticker})")
    print(f"Current Price: ${result.current_price}")
    print(f"Market Cap: {result.market_cap}")
    print(f"Financial Summary: {result.financial_summary}")

    print("\n" + "="*50)

    # Test with an index
    print("Testing with S&P 500...")
    result2 = run_financials_agent("S&P 500")
    print(f"Index: {result2.company} ({result2.ticker})")
    print(f"Level: {result2.current_price}")
    print(f"YTD: {result2.revenue_ttm}")
    print(f"Avg P/E: {result2.pe_ratio}")
    print(f"Financial Summary: {result2.financial_summary}")
import os
from dotenv import load_dotenv
from typing import TypedDict, List, Annotated
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.types import Send
import operator

load_dotenv()

# Import all agents
from news_agent import run_news_agent, NewsOutput
from financials_agent import run_financials_agent, FinancialsOutput
from sec_agent import run_sec_agent, SECOutput
from sentiment_agent import run_sentiment_agent, SentimentOutput


# --- Define the shared state ---
# This is the object that flows through the entire graph
class ResearchState(TypedDict):
    company: str                          # Input: company to research
    news: NewsOutput                      # Output from News Agent
    financials: FinancialsOutput          # Output from Financials Agent
    sec: SECOutput                        # Output from SEC Agent
    sentiment: SentimentOutput            # Output from Sentiment Agent
    report: str                           # Final synthesized report
    recommendation: str                   # BUY / HOLD / SELL


# --- Define the final report schema ---
class ResearchReport(BaseModel):
    executive_summary: str = Field(description="3-4 sentence overview of the company")
    financial_analysis: str = Field(description="Analysis of financial health and metrics")
    news_analysis: str = Field(description="Summary of recent news and its implications")
    risk_assessment: str = Field(description="Key risks from SEC filings and analysis")
    sentiment_analysis: str = Field(description="Market sentiment analysis summary")
    recommendation: str = Field(description="Investment recommendation: BUY, HOLD, SELL for companies — or BULLISH, NEUTRAL, BEARISH for market indices")
    recommendation_reasoning: str = Field(description="2-3 sentences explaining the recommendation")


# --- Node functions ---
# Each node receives the state and returns updates to the state

def news_node(state: ResearchState) -> dict:
    """News Agent node"""
    print(f"  📰 News Agent running...")
    result = run_news_agent(state["company"])
    return {"news": result}


def financials_node(state: ResearchState) -> dict:
    """Financials Agent node"""
    print(f"  💰 Financials Agent running...")
    result = run_financials_agent(state["company"])
    return {"financials": result}


def sec_node(state: ResearchState) -> dict:
    """SEC Agent node"""
    print(f"  📋 SEC Agent running...")
    result = run_sec_agent(state["company"])
    return {"sec": result}


def sentiment_node(state: ResearchState) -> dict:
    """Sentiment Agent node"""
    print(f"  💭 Sentiment Agent running...")
    # Pass news headlines to sentiment agent if available
    headlines = []
    if "news" in state and state["news"]:
        headlines = state["news"].headlines
    result = run_sentiment_agent(state["company"], headlines)
    return {"sentiment": result}


def supervisor_node(state: ResearchState) -> dict:
    """Supervisor node — synthesizes all agent outputs into final report"""
    print(f"\n🧠 Supervisor Agent synthesizing results...")

    company = state["company"]
    news = state.get("news")
    financials = state.get("financials")
    sec = state.get("sec")
    sentiment = state.get("sentiment")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
    llm_structured = llm.with_structured_output(ResearchReport)

    
    # Detect if input is an index, ETF, or asset class
    index_keywords = ["s&p", "dow jones", "nasdaq", "ftse", "nikkei", "russell", "etf", "index", "vix"]
    is_index = any(kw in company.lower() for kw in index_keywords)

    index_note = ""
    if is_index:
        index_note = f"""
    IMPORTANT NOTE: {company} is a market index or ETF, not an individual company. 
    - Do NOT provide a stock-specific BUY/HOLD/SELL recommendation
    - Instead provide a MARKET OUTLOOK recommendation: BULLISH, NEUTRAL, or BEARISH
    - Clearly state at the start of the executive summary that this is a market index analysis
    - Focus on macroeconomic factors, constituent performance, and market trends
    """

    prompt = f"""You are a senior financial analyst. Synthesize the following research into a professional investment report for {company}.
    {index_note}

NEWS RESEARCH:
- Headlines: {news.headlines if news else 'N/A'}
- Summary: {news.summary if news else 'N/A'}
- News Sentiment: {news.sentiment_hint if news else 'N/A'}

FINANCIAL DATA:
- Current Price: ${financials.current_price if financials else 'N/A'}
- Market Cap: {financials.market_cap if financials else 'N/A'}
- P/E Ratio: {financials.pe_ratio if financials else 'N/A'}
- Revenue: {financials.revenue_ttm if financials else 'N/A'}
- Profit Margin: {financials.profit_margin if financials else 'N/A'}%
- Analyst Recommendation: {financials.analyst_recommendation if financials else 'N/A'}
- Financial Summary: {financials.financial_summary if financials else 'N/A'}
- Key Concerns: {financials.key_concerns if financials else 'N/A'}

SEC FILING ANALYSIS:
- Filing: {sec.filing_type if sec else 'N/A'} ({sec.filing_date if sec else 'N/A'})
- Top Risks: {sec.top_risks if sec else 'N/A'}
- SEC Summary: {sec.sec_summary if sec else 'N/A'}

SENTIMENT ANALYSIS (FinBERT):
- Overall Sentiment: {sentiment.overall_sentiment if sentiment else 'N/A'}
- Sentiment Score: {sentiment.sentiment_score if sentiment else 'N/A'} (-1 to 1)
- Positive Signals: {sentiment.positive_signals if sentiment else 'N/A'}
- Negative Signals: {sentiment.negative_signals if sentiment else 'N/A'}

Based on ALL of the above data, provide:
1. A professional executive summary
2. Financial analysis
3. News analysis
4. Risk assessment
5. Sentiment analysis
6. A clear BUY, HOLD, or SELL recommendation with reasoning

Be specific, cite actual numbers, and be direct about the recommendation."""

    report = llm_structured.invoke(prompt)

    # Format the final report as text
    formatted_report = f"""
# {company} Investment Research Report

## Executive Summary
{report.executive_summary}

## Financial Analysis
{report.financial_analysis}

## Recent News
{report.news_analysis}

## Risk Assessment
{report.risk_assessment}

## Market Sentiment
{report.sentiment_analysis}

## Recommendation: {report.recommendation}
{report.recommendation_reasoning}
"""

    print(f"✅ Research complete — Recommendation: {report.recommendation}")
    return {
        "report": formatted_report,
        "recommendation": report.recommendation
    }


# --- Build the graph ---
def build_research_graph():
    graph = StateGraph(ResearchState)

    # Add all nodes
    graph.add_node("news", news_node)
    graph.add_node("financials", financials_node)
    graph.add_node("sec", sec_node)
    graph.add_node("sentiment", sentiment_node)
    graph.add_node("supervisor", supervisor_node)

    # Set entry point — supervisor starts first and dispatches to all agents
    graph.set_entry_point("news")

    # All 4 agents run, then supervisor synthesizes
    # Since LangGraph runs nodes added to the same level in parallel
    graph.add_edge("news", "supervisor")
    graph.add_edge("financials", "supervisor")
    graph.add_edge("sec", "supervisor")
    graph.add_edge("sentiment", "supervisor")
    graph.add_edge("supervisor", END)

    return graph.compile()


# --- Run the full research pipeline ---
def run_research(company: str) -> ResearchState:
    """Run complete research on a company"""
    print(f"\n{'='*60}")
    print(f"🔬 Starting research on: {company}")
    print(f"{'='*60}")

    graph = build_research_graph()

    # Run news, financials, and sec in parallel first
    # Then sentiment uses news headlines
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}

    print("\n⚡ Running News, Financials, and SEC agents in parallel...")

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(run_news_agent, company): "news",
            executor.submit(run_financials_agent, company): "financials",
            executor.submit(run_sec_agent, company): "sec",
        }

        for future in as_completed(futures):
            agent_name = futures[future]
            try:
                results[agent_name] = future.result()
                print(f"  ✅ {agent_name.capitalize()} Agent done")
            except Exception as e:
                print(f"  ❌ {agent_name.capitalize()} Agent failed: {e}")
                results[agent_name] = None

    # Run sentiment with news headlines
    print("\n⚡ Running Sentiment Agent with news headlines...")
    headlines = results.get("news").headlines if results.get("news") else []
    results["sentiment"] = run_sentiment_agent(company, headlines)
    print(f"  ✅ Sentiment Agent done")

    # Build state and run supervisor
    state = ResearchState(
        company=company,
        news=results.get("news"),
        financials=results.get("financials"),
        sec=results.get("sec"),
        sentiment=results.get("sentiment"),
        report="",
        recommendation=""
    )

    # Run supervisor
    final_state = supervisor_node(state)
    state["report"] = final_state["report"]
    state["recommendation"] = final_state["recommendation"]

    return state


# Test it
if __name__ == "__main__":
    result = run_research("Tesla")

    print("\n" + "="*60)
    print("FINAL RESEARCH REPORT")
    print("="*60)
    print(result["report"])
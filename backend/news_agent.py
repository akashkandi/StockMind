import os
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent


load_dotenv()

# --- Structured output schema ---
# This tells the LLM exactly what format to return
class NewsOutput(BaseModel):
    company: str = Field(description="Company name that was researched")
    headlines: List[str] = Field(description="Top 5 most important recent headlines")
    summary: str = Field(description="2-3 sentence summary of recent news")
    sentiment_hint: str = Field(description="Overall news tone: positive, negative, or neutral")
    sources: List[str] = Field(description="URLs of the news sources found")

# --- Tools ---
search_tool = TavilySearch(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
)

# --- LLM ---
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1
)

# --- System prompt for the News Agent ---
NEWS_AGENT_PROMPT = """You are a financial news researcher. Your job is to find the most recent and relevant news about a company.

When given a company name:
1. Search for recent news about the company (last 30 days)
2. Search for any recent earnings, partnerships, or major announcements
3. Identify the overall sentiment of recent news coverage

Always search at least twice with different queries to get comprehensive coverage.
Focus on factual, financial, and business news — not opinion pieces.

Return your findings in the exact structured format requested."""

# --- Create the agent ---
news_agent = create_react_agent(
    model=llm,
    tools=[search_tool],
    prompt=NEWS_AGENT_PROMPT
)

def run_news_agent(company: str) -> NewsOutput:
    """Run the news agent for a given company"""
    print(f"\n📰 News Agent starting research on: {company}")
    
    result = news_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"""Research recent news about {company}.
                
Return a JSON object with exactly these fields:
- company: the company name
- headlines: list of 5 most important recent headlines
- summary: 2-3 sentence summary of recent news  
- sentiment_hint: overall tone (positive/negative/neutral)
- sources: list of source URLs

Company to research: {company}"""
            }
        ]
    })
    
    # Extract the final message
    final_message = result["messages"][-1].content
    print(f"✅ News Agent complete for {company}")
    
    # Parse into structured output
    llm_structured = llm.with_structured_output(NewsOutput)
    structured_result = llm_structured.invoke(
        f"Convert this research into the required format:\n\n{final_message}\n\nCompany: {company}"
    )
    
    return structured_result


# Test it directly
if __name__ == "__main__":
    result = run_news_agent("Apple")
    
    print("\n" + "="*50)
    print(f"Company: {result.company}")
    print(f"\nHeadlines:")
    for i, h in enumerate(result.headlines, 1):
        print(f"  {i}. {h}")
    print(f"\nSummary: {result.summary}")
    print(f"\nSentiment: {result.sentiment_hint}")
    print(f"\nSources:")
    for s in result.sources:
        print(f"  - {s}")
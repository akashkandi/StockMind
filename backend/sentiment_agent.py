import os
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv()

# --- Structured output schema ---
class SentimentOutput(BaseModel):
    company: str = Field(description="Company name")
    overall_sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    sentiment_score: float = Field(description="Sentiment score from -1.0 (very negative) to 1.0 (very positive)")
    positive_signals: List[str] = Field(description="List of positive sentiment signals found")
    negative_signals: List[str] = Field(description="List of negative sentiment signals found")
    sentiment_summary: str = Field(description="2-3 sentence summary of overall market sentiment")


def run_sentiment_agent(company: str, news_headlines: List[str] = None) -> SentimentOutput:
    """
    Run sentiment analysis.
    
    Locally: uses FinBERT (real ML model inference)
    Production/Render: uses GPT-4o-mini via API (same results, no memory issues)
    
    The USE_API_SENTIMENT env var controls which mode is used.
    Set USE_API_SENTIMENT=true on Render, leave unset locally to use FinBERT.
    """
    use_api = os.getenv("USE_API_SENTIMENT", "false").lower() == "true"

    if use_api:
        return _run_via_api(company, news_headlines)
    else:
        return _run_via_finbert(company, news_headlines)


def _run_via_api(company: str, news_headlines: List[str] = None) -> SentimentOutput:
    """GPT-4o-mini based sentiment — used in production deployment"""
    print(f"\n💭 Sentiment Agent analyzing: {company} (API mode)")

    if not news_headlines:
        news_headlines = [
            f"{company} stock performance",
            f"{company} earnings and revenue",
            f"{company} market outlook"
        ]

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    llm_structured = llm.with_structured_output(SentimentOutput)

    result = llm_structured.invoke(
        f"""Analyze the market sentiment for {company} based on these recent headlines:

{chr(10).join(f'- {h}' for h in news_headlines)}

Provide a financial sentiment analysis:
- overall_sentiment: positive, negative, or neutral
- sentiment_score: number from -1.0 (very negative) to 1.0 (very positive)
- positive_signals: 2-3 positive factors from the headlines
- negative_signals: 2-3 negative/risk factors from the headlines
- sentiment_summary: 2-3 sentence summary of market sentiment

Be specific and reference the actual headlines provided."""
    )

    print(f"✅ Sentiment Agent complete for {company}")
    return result


def _run_via_finbert(company: str, news_headlines: List[str] = None) -> SentimentOutput:
    """FinBERT based sentiment — used locally for real ML inference"""
    print(f"\n💭 Sentiment Agent analyzing: {company} (FinBERT mode)")

    # Lazy import — only load FinBERT when actually needed locally
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

    if not news_headlines:
        news_headlines = [
            f"{company} stock performance this quarter",
            f"{company} revenue growth and earnings",
            f"{company} market position and competition",
            f"{company} investor confidence and outlook",
            f"{company} product launches and innovation"
        ]

    print(f"  → Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    finbert = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
    print(f"  → Analyzing {len(news_headlines)} samples with FinBERT")

    results = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for text in news_headlines:
        try:
            pred = finbert(text[:512])[0]
            label = pred["label"].lower()
            results[label] = results.get(label, 0) + pred["score"]
        except Exception:
            continue

    total = len(news_headlines)
    score = round((results["positive"] - results["negative"]) / total, 3)

    if score > 0.2:
        overall = "positive"
    elif score < -0.2:
        overall = "negative"
    else:
        overall = "neutral"

    print(f"  → FinBERT score: {score} ({overall})")

    # Use LLM to generate structured output
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    llm_structured = llm.with_structured_output(SentimentOutput)

    result = llm_structured.invoke(
        f"""Based on FinBERT ML sentiment analysis for {company}:

FinBERT Results:
- Texts analyzed: {total}
- Positive score: {results['positive']:.3f}
- Negative score: {results['negative']:.3f}
- Neutral score: {results['neutral']:.3f}
- Overall sentiment score: {score} (-1 to 1)
- Overall sentiment: {overall}

Headlines analyzed:
{chr(10).join(f'- {h}' for h in news_headlines)}

Provide structured sentiment analysis with positive signals, negative signals, and summary."""
    )

    print(f"✅ Sentiment Agent complete for {company}")
    return result


# Test it directly
if __name__ == "__main__":
    # Test API mode
    os.environ["USE_API_SENTIMENT"] = "true"
    result = run_sentiment_agent("Apple", [
        "Apple reports record quarterly revenue",
        "Apple faces competition from Chinese manufacturers",
        "Apple AI features drive iPhone upgrade cycle"
    ])
    print(f"Company: {result.company}")
    print(f"Sentiment: {result.overall_sentiment} ({result.sentiment_score})")
    print(f"Summary: {result.sentiment_summary}")
import os
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

load_dotenv()

# --- Structured output schema ---
class SentimentOutput(BaseModel):
    company: str = Field(description="Company name")
    overall_sentiment: str = Field(description="Overall sentiment: positive, negative, or neutral")
    sentiment_score: float = Field(description="Sentiment score from -1.0 (very negative) to 1.0 (very positive)")
    positive_signals: List[str] = Field(description="List of positive sentiment signals found")
    negative_signals: List[str] = Field(description="List of negative sentiment signals found")
    sentiment_summary: str = Field(description="2-3 sentence summary of overall market sentiment")


# --- Load FinBERT model ---
print("🤖 Loading FinBERT model from HuggingFace...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
finbert_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=-1  # CPU
)
print("✅ FinBERT model loaded")


def analyze_sentiment(texts: List[str]) -> dict:
    """Run FinBERT on a list of texts and return aggregated sentiment"""
    if not texts:
        return {"positive": 0, "negative": 0, "neutral": 0, "score": 0}

    results = {"positive": 0, "negative": 0, "neutral": 0}

    for text in texts:
        # Truncate to 512 tokens max (FinBERT limit)
        truncated = text[:512]
        try:
            prediction = finbert_pipeline(truncated)[0]
            label = prediction["label"].lower()
            results[label] = results.get(label, 0) + prediction["score"]
        except Exception as e:
            continue

    # Calculate overall score (-1 to 1)
    total = len(texts)
    if total == 0:
        return results

    # Normalize
    score = (results["positive"] - results["negative"]) / total
    results["score"] = round(score, 3)

    return results


def run_sentiment_agent(company: str, news_headlines: List[str] = None) -> SentimentOutput:
    """Run FinBERT sentiment analysis on company news"""
    print(f"\n💭 Sentiment Agent analyzing: {company}")

    # If no headlines provided use default financial phrases about the company
    if not news_headlines:
        news_headlines = [
            f"{company} stock performance this quarter",
            f"{company} revenue growth and earnings",
            f"{company} market position and competition",
            f"{company} investor confidence and outlook",
            f"{company} product launches and innovation"
        ]

    print(f"  → Analyzing {len(news_headlines)} text samples with FinBERT")

    # Run FinBERT on all headlines
    sentiment_results = analyze_sentiment(news_headlines)

    print(f"  → FinBERT results: positive={sentiment_results['positive']:.2f}, "
          f"negative={sentiment_results['negative']:.2f}, "
          f"neutral={sentiment_results['neutral']:.2f}")
    print(f"  → Overall score: {sentiment_results['score']}")

    # Determine overall sentiment label
    if sentiment_results["score"] > 0.2:
        overall = "positive"
    elif sentiment_results["score"] < -0.2:
        overall = "negative"
    else:
        overall = "neutral"

    # Use LLM to generate structured output with interpretation
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
    llm_structured = llm.with_structured_output(SentimentOutput)

    result = llm_structured.invoke(
        f"""Based on FinBERT sentiment analysis results for {company}, provide a structured sentiment report.

FinBERT Analysis Results:
- Texts analyzed: {len(news_headlines)}
- Positive score: {sentiment_results['positive']:.3f}
- Negative score: {sentiment_results['negative']:.3f}
- Neutral score: {sentiment_results['neutral']:.3f}
- Overall sentiment score: {sentiment_results['score']} (scale: -1.0 to 1.0)
- Overall sentiment: {overall}

Headlines analyzed:
{chr(10).join(f'- {h}' for h in news_headlines)}

Based on these FinBERT scores:
1. Identify 2-3 positive sentiment signals
2. Identify 2-3 negative sentiment signals  
3. Write a 2-3 sentence sentiment summary
4. Fill in all structured fields"""
    )

    print(f"✅ Sentiment Agent complete for {company}")
    return result


# Test it directly
if __name__ == "__main__":
    # Test with real financial headlines
    test_headlines = [
        "Apple reports record quarterly revenue beating analyst expectations",
        "Apple faces increasing competition from Chinese smartphone makers",
        "Apple announces major AI features for iPhone driving upgrade cycle",
        "Apple supply chain concerns amid geopolitical tensions with China",
        "Apple services revenue continues strong double digit growth"
    ]

    result = run_sentiment_agent("Apple", test_headlines)

    print("\n" + "="*50)
    print(f"Company: {result.company}")
    print(f"Overall Sentiment: {result.overall_sentiment}")
    print(f"Sentiment Score: {result.sentiment_score} (-1 to 1)")
    print(f"\nPositive Signals:")
    for s in result.positive_signals:
        print(f"  + {s}")
    print(f"\nNegative Signals:")
    for s in result.negative_signals:
        print(f"  - {s}")
    print(f"\nSentiment Summary: {result.sentiment_summary}")
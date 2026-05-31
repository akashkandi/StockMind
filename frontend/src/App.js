import { useState, useEffect, useRef } from "react";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import { RadialBarChart, RadialBar, PolarAngleAxis } from "recharts";
import "./App.css";

const API_URL = "https://stockmind-2.onrender.com";
const WS_URL = "wss://stockmind-2.onrender.com";

const AGENT_INFO = {
  news: { label: "News Agent", icon: "📰", desc: "Searching recent news..." },
  financials: { label: "Financials Agent", icon: "💰", desc: "Fetching live financial data..." },
  sec: { label: "SEC Agent", icon: "📋", desc: "Reading SEC filings..." },
  sentiment: { label: "Sentiment Agent", icon: "🧠", desc: "Running FinBERT analysis..." },
  supervisor: { label: "Supervisor Agent", icon: "⚡", desc: "Synthesizing report..." },
};

function AgentCard({ name, status }) {
  const info = AGENT_INFO[name] || { label: name, icon: "🤖", desc: "" };
  return (
    <div className={`agent-card ${status}`}>
      <div className="agent-icon">{info.icon}</div>
      <div className="agent-info">
        <p className="agent-name">{info.label}</p>
        <p className="agent-status-text">
          {status === "idle" && "Waiting..."}
          {status === "running" && info.desc}
          {status === "complete" && "Complete ✓"}
        </p>
      </div>
      <div className="agent-indicator">
        {status === "running" && <div className="spinner"></div>}
        {status === "complete" && <div className="check">✓</div>}
        {status === "idle" && <div className="idle-dot"></div>}
      </div>
    </div>
  );
}

function SentimentGauge({ score }) {
  const normalized = Math.round((score + 1) * 50);
  const color = score > 0.2 ? "#4ade80" : score < -0.2 ? "#f87171" : "#facc15";
  const data = [{ value: normalized, fill: color }];

  return (
    <div className="gauge-container">
      <RadialBarChart
        width={120} height={70}
        cx={60} cy={65}
        innerRadius={40} outerRadius={60}
        startAngle={180} endAngle={0}
        data={data}
      >
        <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
        <RadialBar dataKey="value" cornerRadius={4} background={{ fill: "#1a1a1a" }} />
      </RadialBarChart>
      <p className="gauge-score" style={{ color }}>{score > 0 ? "+" : ""}{score?.toFixed(2)}</p>
    </div>
  );
}

function App() {
  const [company, setCompany] = useState("");
  const [researching, setResearching] = useState(false);
  const [agentStatus, setAgentStatus] = useState({
    news: "idle", financials: "idle", sec: "idle",
    sentiment: "idle", supervisor: "idle"
  });
  const [report, setReport] = useState(null);
  const [pastReports, setPastReports] = useState([]);
  const [activeTab, setActiveTab] = useState("research");
  const [error, setError] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
    fetchPastReports();
  }, []);

  const fetchPastReports = async () => {
    try {
      const res = await axios.get(`${API_URL}/reports`);
      setPastReports(res.data);
    } catch (e) {
      console.error("Failed to fetch reports");
    }
  };

  const resetAgents = () => {
    setAgentStatus({
      news: "idle", financials: "idle", sec: "idle",
      sentiment: "idle", supervisor: "idle"
    });
  };

  const startResearch = async () => {
    if (!company.trim()) return;
    setResearching(true);
    setReport(null);
    setError(null);
    resetAgents();
    setActiveTab("research");

    try {
      // Get research ID
      const res = await axios.post(`${API_URL}/research`, { company });
      const { research_id } = res.data;

      // Connect WebSocket
      const ws = new WebSocket(`${WS_URL}/ws/${research_id}`);
      wsRef.current = ws;

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.agent && data.agent !== "system") {
          setAgentStatus(prev => ({
            ...prev,
            [data.agent]: data.status
          }));
        }

        if (data.status === "complete" && data.report) {
          setReport({
            report: data.report,
            recommendation: data.recommendation,
            report_id: data.report_id
          });
          setResearching(false);
          fetchPastReports();
          ws.close();
        }

        if (data.status === "error") {
          setError(data.message);
          setResearching(false);
          ws.close();
        }
      };

      ws.onerror = () => {
        setError("Connection error. Make sure the backend is running.");
        setResearching(false);
      };

    } catch (e) {
      setError("Failed to start research. Make sure the backend is running.");
      setResearching(false);
    }
  };

  const loadReport = async (reportId) => {
    try {
      const res = await axios.get(`${API_URL}/reports/${reportId}`);
      const r = res.data;
      setReport({
        report: r.report_text,
        recommendation: r.recommendation,
        report_id: r.id,
        sentiment_score: r.sentiment_score,
        current_price: r.current_price,
        market_cap: r.market_cap,
        ticker: r.ticker
      });
      setActiveTab("research");
    } catch (e) {
      console.error("Failed to load report");
    }
  };

  const recColor = (rec) => {
    if (!rec) return "#666";
    if (rec === "BUY" || rec === "BULLISH") return "#4ade80";
    if (rec === "SELL" || rec === "BEARISH") return "#f87171";
    return "#facc15";
  };
  // const recColor = (rec) => {
  //   if (!rec) return "#666";
  //   if (rec === "BUY") return "#4ade80";
  //   if (rec === "SELL") return "#f87171";
  //   return "#facc15";
  // };

  return (
    <div className="app">
      <header className="header">
        <div className="header-left">
          <div className="logo">📈</div>
          <div>
            <h1>StockMind</h1>
            <p>Multi-Agent Investment Research System</p>
          </div>
        </div>
        <div className="header-right">
          <span className="badge-tech">LangGraph</span>
          <span className="badge-tech">FinBERT</span>
          <span className="badge-tech">GPT-4o</span>
        </div>
      </header>

      <div className="search-bar">
        <input
          className="company-input"
          placeholder="Enter company name — e.g. Apple, Tesla, Microsoft, Google..."
          value={company}
          onChange={(e) => setCompany(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && !researching && startResearch()}
          disabled={researching}
        />
        <button
          className="research-btn"
          onClick={startResearch}
          disabled={researching || !company.trim()}
        >
          {researching ? "Researching..." : "Generate Report →"}
        </button>
      </div>

      {error && <div className="error-bar">❌ {error}</div>}

      <nav className="nav">
        <button className={activeTab === "research" ? "nav-btn active" : "nav-btn"} onClick={() => setActiveTab("research")}>Research</button>
        <button className={activeTab === "history" ? "nav-btn active" : "nav-btn"} onClick={() => setActiveTab("history")}>
          History {pastReports.length > 0 && <span className="count">{pastReports.length}</span>}
        </button>
      </nav>

      <main className="main">
        {activeTab === "research" && (
          <div className="research-tab">

            {/* Agent Status Grid */}
            {(researching || report) && (
              <div className="agents-section">
                <p className="section-label">
                  {researching ? "⚡ Agents running in parallel..." : "✅ Research complete"}
                </p>
                <div className="agents-grid">
                  {Object.entries(agentStatus).map(([name, status]) => (
                    <AgentCard key={name} name={name} status={status} />
                  ))}
                </div>
              </div>
            )}

            {/* Empty state */}
            {!researching && !report && (
              <div className="empty-state">
                <div className="empty-icon">🔬</div>
                <h2>Research any company instantly</h2>
                <p>Enter a company name above. 4 AI agents will research news, financials, SEC filings, and sentiment simultaneously — then synthesize everything into a professional investment report.</p>
                <div className="example-chips">
                  {["Apple", "Tesla", "Microsoft", "Google", "Amazon", "Nvidia"].map(c => (
                    <button key={c} className="example-chip" onClick={() => setCompany(c)}>{c}</button>
                  ))}
                </div>
              </div>
            )}

            {/* Report */}
            {report && (
              <div className="report-section">
                <div className="report-header">
                  <div>
                    <h2 className="report-company">{company || "Research Report"}</h2>
                    {report.ticker && <span className="ticker-badge">{report.ticker}</span>}
                  </div>
                  <div className="rec-badge" style={{ background: recColor(report.recommendation) + "22", border: `1px solid ${recColor(report.recommendation)}`, color: recColor(report.recommendation) }}>
                    {report.recommendation}
                  </div>
                </div>

                {(report.current_price || report.sentiment_score !== undefined) && (
                  <div className="report-metrics">
                    {report.current_price && (
                      <div className="metric">
                        <span className="metric-label">Price</span>
                        <span className="metric-value">${report.current_price}</span>
                      </div>
                    )}
                    {report.market_cap && (
                      <div className="metric">
                        <span className="metric-label">Market Cap</span>
                        <span className="metric-value">{report.market_cap}</span>
                      </div>
                    )}
                    {report.sentiment_score !== undefined && (
                      <div className="metric">
                        <span className="metric-label">Sentiment</span>
                        <SentimentGauge score={report.sentiment_score} />
                      </div>
                    )}
                  </div>
                )}

                <div className="report-body">
                  <ReactMarkdown>{report.report}</ReactMarkdown>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === "history" && (
          <div className="history-tab">
            <h2>Past Research Reports</h2>
            {pastReports.length === 0 ? (
              <p className="empty-history">No reports yet. Research a company to get started.</p>
            ) : (
              <div className="history-list">
                {pastReports.map(r => (
                  <div key={r.id} className="history-card" onClick={() => loadReport(r.id)}>
                    <div className="history-left">
                      <p className="history-company">{r.company}</p>
                      <p className="history-ticker">{r.ticker} · {new Date(r.created_at).toLocaleDateString()}</p>
                    </div>
                    <div className="history-right">
                      <span className="rec-pill" style={{ background: recColor(r.recommendation) + "22", color: recColor(r.recommendation) }}>
                        {r.recommendation}
                      </span>
                      {r.current_price && <span className="history-price">${r.current_price}</span>}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
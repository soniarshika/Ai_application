import './Dashboard.css';

export default function Dashboard({ navigate }) {
  return (
    <div className="dashboard">
      {/* Hero */}
      <div className="dashboard-hero">
        <div className="hero-badge">
          <span className="hero-badge-dot" />
          AI-Powered · RAG · OpenAI
        </div>
        <h1 className="dashboard-title">Logistics Document AI</h1>
        <p className="dashboard-subtitle">
          Upload, index, and query logistics documents using retrieval-augmented generation.
          Extract structured data instantly — no manual parsing.
        </p>
      </div>

      {/* Cards */}
      <div className="dashboard-cards">
        {/* RAG Chatbot */}
        <div className="dash-card">
          <div className="dash-card-header">
            <div className="dash-card-icon rag">📚</div>
            <span className="dash-card-tag">RAG Pipeline</span>
          </div>
          <div>
            <h2 className="dash-card-title">RAG Chatbot</h2>
          </div>
          <p className="dash-card-desc">
            Index logistics documents into a persistent knowledge base and ask natural-language questions.
            Answers are grounded strictly in your documents with confidence scoring.
          </p>
          <div className="dash-card-divider" />
          <div className="dash-card-actions">
            <button className="btn btn-primary" onClick={() => navigate('train')}>
              Train Documents
            </button>
            <button className="btn btn-secondary" onClick={() => navigate('ask')}>
              Ask Queries
            </button>
          </div>
        </div>

        {/* Direct Extract */}
        <div className="dash-card">
          <div className="dash-card-header">
            <div className="dash-card-icon extract">⚡</div>
            <span className="dash-card-tag">Instant</span>
          </div>
          <div>
            <h2 className="dash-card-title">Direct Extract</h2>
          </div>
          <p className="dash-card-desc">
            Upload any logistics document and instantly extract structured shipment fields as JSON —
            shipper, consignee, rate, dates, carrier, and more. Nothing is stored or indexed.
          </p>
          <div className="dash-card-divider" />
          <div className="dash-card-actions">
            <button className="btn btn-primary" onClick={() => navigate('extract')}>
              Extract Fields
            </button>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="dashboard-footer">
        <span>GPT-4o</span>
        <span className="footer-dot" />
        <span>text-embedding-3-small</span>
        <span className="footer-dot" />
        <span>ChromaDB</span>
        <span className="footer-dot" />
        <span>FastAPI</span>
      </div>
    </div>
  );
}

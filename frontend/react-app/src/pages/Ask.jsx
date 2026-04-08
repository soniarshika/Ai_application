import { useState, useEffect, useRef } from 'react';
import { api } from '../api';
import './Ask.css';

function ConfidenceBadge({ score }) {
  const pct = (score * 100).toFixed(1);
  const cls = score >= 0.75 ? 'high' : score >= 0.5 ? 'medium' : 'low';
  const label = score >= 0.75 ? 'High' : score >= 0.5 ? 'Medium' : 'Low';
  return <span className={`badge badge-${cls}`}>● {pct}% {label}</span>;
}

function ChunkBadge({ type }) {
  const map = { kv_block: ['🟩','#3dd68c'], table_row: ['🟨','#f5c842'], narrative: ['🟦','#6c8ef5'] };
  const [icon] = map[type] || ['⬜', '#7880a0'];
  return <span className="chunk-badge">{icon} {type}</span>;
}

export default function Ask({ navigate }) {
  const [docs, setDocs]         = useState([]);
  const [selectedId, setSelId]  = useState('');
  const [question, setQuestion] = useState('');
  const [loading, setLoading]   = useState(false);
  const [answer, setAnswer]     = useState(null);
  const [error, setError]       = useState('');
  const inputRef = useRef();

  useEffect(() => {
    api.listDocs().then(({ data }) => {
      const list = data || [];
      setDocs(list);
      if (list.length) setSelId(list[0].doc_id);
    });
  }, []);

  async function submit(e) {
    e?.preventDefault();
    if (!question.trim() || !selectedId) return;
    setLoading(true);
    setAnswer(null);
    setError('');
    const { data, error } = await api.ask(selectedId, question);
    setLoading(false);
    if (error) { setError(error); return; }
    setAnswer(data);
  }

  if (docs.length === 0 && !loading) {
    return (
      <div className="ask-empty">
        <div style={{ fontSize: 40 }}>📭</div>
        <p>No documents trained yet.</p>
        <button className="btn btn-primary" onClick={() => navigate('train')}>
          🗂️ Go to Train
        </button>
      </div>
    );
  }

  return (
    <div className="ask-layout">
      {/* Left: doc selector */}
      <aside className="ask-sidebar">
        <div className="sidebar-label">Documents</div>
        {docs.map(doc => (
          <button
            key={doc.doc_id}
            className={`doc-btn ${doc.doc_id === selectedId ? 'active' : ''}`}
            onClick={() => { setSelId(doc.doc_id); setAnswer(null); }}
          >
            <span className="doc-btn-icon">📄</span>
            <span className="doc-btn-info">
              <span className="doc-btn-name">{doc.filename}</span>
              <span className="doc-btn-meta">{doc.chunk_count} chunks</span>
            </span>
          </button>
        ))}
      </aside>

      {/* Right: Q&A */}
      <div className="ask-main">
        <div className="ask-header">
          <h2 className="page-title">💬 Ask Queries</h2>
          <p className="page-subtitle">Questions are answered strictly from the selected document.</p>
        </div>

        <form className="ask-form" onSubmit={submit}>
          <input
            ref={inputRef}
            className="ask-input"
            value={question}
            onChange={e => setQuestion(e.target.value)}
            placeholder="e.g. What is the carrier rate?  Who is the consignee?  When is pickup?"
            disabled={loading}
          />
          <button className="btn btn-primary" type="submit" disabled={loading || !question.trim()}>
            {loading ? <span className="spinner" /> : 'Ask'}
          </button>
        </form>

        {error && <div className="alert alert-error">{error}</div>}

        {loading && (
          <div className="ask-loading">
            <span className="spinner spinner-lg" style={{ color: 'var(--accent)' }} />
            <span style={{ color: 'var(--text-dim)' }}>Searching document…</span>
          </div>
        )}

        {answer && (
          <div className="answer-card card">
            <div className="answer-meta">
              <span className="answer-label">Answer</span>
              <ConfidenceBadge score={answer.confidence} />
            </div>

            {answer.guardrail_triggered ? (
              <div className="alert alert-warning" style={{ marginTop: 12 }}>
                ⚠️ {answer.answer}
              </div>
            ) : (
              <div className="answer-text">{answer.answer}</div>
            )}

            <div className="confidence-bar">
              <div
                className="confidence-fill"
                style={{
                  width: `${answer.confidence * 100}%`,
                  background: answer.confidence >= 0.75
                    ? 'var(--green)' : answer.confidence >= 0.5
                    ? 'var(--yellow)' : 'var(--red)',
                }}
              />
            </div>

            {answer.source_chunks?.length > 0 && (
              <details className="sources">
                <summary>📎 {answer.source_chunks.length} source chunk(s)</summary>
                {answer.source_chunks.map((c, i) => (
                  <div key={i} className="source-chunk">
                    <div className="source-meta">
                      <span>#{i + 1}</span>
                      <ChunkBadge type={c.chunk_type} />
                      <span>Page {c.page_number}</span>
                      <span>{(c.similarity * 100).toFixed(1)}% similarity</span>
                    </div>
                    <pre className="source-text">{c.text}</pre>
                  </div>
                ))}
              </details>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

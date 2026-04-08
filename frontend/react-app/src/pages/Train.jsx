import { useState, useEffect, useRef } from 'react';
import { api } from '../api';
import './Train.css';

export default function Train({ navigate }) {
  const [docs, setDocs]           = useState([]);
  const [loading, setLoading]     = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress]   = useState([]);
  const [dragOver, setDragOver]   = useState(false);
  const fileRef = useRef();

  useEffect(() => { fetchDocs(); }, []);

  async function fetchDocs() {
    setLoading(true);
    const { data } = await api.listDocs();
    setDocs(data || []);
    setLoading(false);
  }

  async function handleFiles(files) {
    if (!files?.length) return;
    setUploading(true);
    const items = Array.from(files).map(f => ({ name: f.name, status: 'pending', msg: '' }));
    setProgress(items);

    for (let i = 0; i < files.length; i++) {
      const f = Array.from(files)[i];
      setProgress(prev => prev.map((p, idx) => idx === i ? { ...p, status: 'uploading' } : p));
      const { data, error } = await api.upload(f);
      setProgress(prev => prev.map((p, idx) =>
        idx === i
          ? { ...p, status: error ? 'error' : 'done', msg: error || `${data.chunk_count} chunks · ${data.page_count}p` }
          : p
      ));
    }

    setUploading(false);
    fetchDocs();
  }

  async function deleteDoc(id) {
    await api.deleteDoc(id);
    fetchDocs();
  }

  function onDrop(e) {
    e.preventDefault();
    setDragOver(false);
    handleFiles(e.dataTransfer.files);
  }

  return (
    <div className="page-container">
      <div className="page-header">
        <h2 className="page-title">Train Documents</h2>
        <p className="page-subtitle">
          Index PDF, DOCX, or TXT files into the knowledge base. Documents persist across sessions.
        </p>
      </div>

      {/* Drop zone */}
      <div
        className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
        onClick={() => fileRef.current.click()}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
      >
        <div className="drop-icon-wrap">📂</div>
        <p className="drop-label"><strong>Click to browse</strong> or drag &amp; drop files here</p>
        <p className="drop-hint">PDF · DOCX · TXT &nbsp;·&nbsp; Multiple files supported</p>
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.docx,.txt"
          multiple
          style={{ display: 'none' }}
          onChange={e => handleFiles(e.target.files)}
        />
      </div>

      {/* Upload progress */}
      {progress.length > 0 && (
        <div className="card upload-progress">
          <div className="progress-title">Upload Progress</div>
          {progress.map((p, i) => (
            <div key={i} className={`progress-row status-${p.status}`}>
              <span className="progress-icon">
                {p.status === 'pending'   && '○'}
                {p.status === 'uploading' && <span className="spinner" style={{ color: 'var(--accent)' }} />}
                {p.status === 'done'      && '✓'}
                {p.status === 'error'     && '✕'}
              </span>
              <span className="progress-name">{p.name}</span>
              {p.msg && <span className="progress-msg">{p.msg}</span>}
            </div>
          ))}
        </div>
      )}

      {/* Indexed documents */}
      <div className="section-header">
        <span className="section-title">
          Indexed Documents {!loading && `(${docs.length})`}
        </span>
        {docs.length > 0 && (
          <button className="btn btn-secondary btn-sm" onClick={() => navigate('ask')}>
            Ask Queries →
          </button>
        )}
      </div>

      {loading && (
        <div className="docs-loading">
          <span className="spinner" style={{ color: 'var(--accent)' }} />
          Loading…
        </div>
      )}

      {!loading && docs.length === 0 && (
        <div className="empty-state">
          <div className="empty-icon">📭</div>
          <p>No documents indexed yet.<br />Upload your first file above.</p>
        </div>
      )}

      {docs.map(doc => (
        <div key={doc.doc_id} className="doc-row">
          <div className="doc-icon-wrap">📄</div>
          <div className="doc-info">
            <div className="doc-name">{doc.filename}</div>
            <div className="doc-meta">
              <span className="doc-meta-chip">{doc.chunk_count} chunks</span>
              <span className="doc-meta-chip">{doc.page_count} page{doc.page_count !== 1 ? 's' : ''}</span>
              {doc.upload_timestamp && (
                <span className="doc-meta-chip">{doc.upload_timestamp.slice(0, 10)}</span>
              )}
            </div>
          </div>
          <button
            className="btn btn-danger btn-sm"
            onClick={() => deleteDoc(doc.doc_id)}
            title="Delete document"
          >
            Delete
          </button>
        </div>
      ))}
    </div>
  );
}

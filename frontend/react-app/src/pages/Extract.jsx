import { useState, useRef } from 'react';
import { api } from '../api';
import './Extract.css';

const FIELDS = [
  ['Shipment ID',        d => d.shipment_id],
  ['Shipper Name',       d => d.shipper?.name],
  ['Shipper Address',    d => d.shipper?.address],
  ['Consignee Name',     d => d.consignee?.name],
  ['Consignee Address',  d => d.consignee?.address],
  ['Pickup Date/Time',   d => d.pickup_datetime],
  ['Delivery Date/Time', d => d.delivery_datetime],
  ['Equipment Type',     d => d.equipment_type],
  ['Mode',               d => d.mode],
  ['Rate',               d => d.rate],
  ['Currency',           d => d.currency],
  ['Weight',             d => d.weight],
  ['Carrier Name',       d => d.carrier_name],
];

export default function Extract() {
  const [file, setFile]       = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult]   = useState(null);
  const [error, setError]     = useState('');
  const [view, setView]       = useState('table');
  const [dragOver, setDragOver] = useState(false);
  const fileRef = useRef();

  function pickFile(files) {
    if (!files?.length) return;
    setFile(files[0]);
    setResult(null);
    setError('');
  }

  async function extract() {
    if (!file) return;
    setLoading(true);
    setError('');
    const { data, error } = await api.extractFile(file);
    setLoading(false);
    if (error) { setError(error); return; }
    setResult(data.data);
  }

  const found   = result ? FIELDS.filter(([, fn]) => fn(result) != null) : [];
  const missing = result ? FIELDS.filter(([, fn]) => fn(result) == null) : [];

  return (
    <div className="extract-container">
      <div className="page-header">
        <h2 className="page-title">📤 Direct Extraction</h2>
        <p className="page-subtitle">
          Upload a logistics document to extract structured shipment fields instantly.
          Nothing is stored or indexed.
        </p>
      </div>

      {/* Drop zone */}
      <div
        className={`drop-zone ${dragOver ? 'drag-over' : ''} ${file ? 'has-file' : ''}`}
        onClick={() => fileRef.current.click()}
        onDragOver={e => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={e => { e.preventDefault(); setDragOver(false); pickFile(e.dataTransfer.files); }}
      >
        {file ? (
          <>
            <div className="drop-icon">📄</div>
            <p className="drop-label"><strong>{file.name}</strong></p>
            <p className="drop-hint">Click to change file</p>
          </>
        ) : (
          <>
            <div className="drop-icon">📂</div>
            <p className="drop-label"><strong>Click to browse</strong> or drag &amp; drop</p>
            <p className="drop-hint">PDF · DOCX · TXT</p>
          </>
        )}
        <input
          ref={fileRef}
          type="file"
          accept=".pdf,.docx,.txt"
          style={{ display: 'none' }}
          onChange={e => pickFile(e.target.files)}
        />
      </div>

      <div className="extract-actions">
        <button className="btn btn-primary" onClick={extract} disabled={!file || loading}>
          {loading ? <><span className="spinner" /> Extracting…</> : 'Extract Fields'}
        </button>
        {result && (
          <button className="btn btn-ghost btn-sm" onClick={() => { setResult(null); setFile(null); }}>
            Clear
          </button>
        )}
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {result && (
        <div className="card result-card">
          <div className="result-header">
            <span className="result-title">Extracted Fields</span>
            <div className="view-toggle">
              <button className={`toggle-btn ${view === 'table' ? 'active' : ''}`} onClick={() => setView('table')}>Table</button>
              <button className={`toggle-btn ${view === 'json'  ? 'active' : ''}`} onClick={() => setView('json')}>JSON</button>
            </div>
          </div>

          {view === 'json' ? (
            <pre className="json-output">{JSON.stringify(result, null, 2)}</pre>
          ) : (
            <div className="field-table">
              {found.map(([label, fn]) => (
                <div key={label} className="field-row">
                  <span className="field-label">{label}</span>
                  <span className="field-value">{String(fn(result))}</span>
                </div>
              ))}
              {missing.length > 0 && (
                <details className="missing-fields">
                  <summary>⬜ {missing.length} field(s) not found</summary>
                  {missing.map(([label]) => (
                    <div key={label} className="field-row missing">
                      <span className="field-label">{label}</span>
                      <span className="field-value null-val">—</span>
                    </div>
                  ))}
                </details>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

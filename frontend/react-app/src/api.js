// ── Token storage ─────────────────────────────────────────────────────────
export const token = {
  get:    ()      => localStorage.getItem('auth_token'),
  set:    (t)     => localStorage.setItem('auth_token', t),
  clear:  ()      => localStorage.removeItem('auth_token'),
};

// ── Core fetch wrapper ────────────────────────────────────────────────────
async function request(method, path, opts = {}) {
  const headers = { ...opts.headers };

  // Attach JWT if present (skip for login — no token yet)
  const t = token.get();
  if (t) headers['Authorization'] = `Bearer ${t}`;

  // Don't set Content-Type for FormData — browser sets it with boundary
  try {
    const res  = await fetch(path, { method, ...opts, headers });
    const ct   = res.headers.get('content-type') || '';
    const data = ct.includes('application/json') ? await res.json() : await res.text();

    if (res.status === 401) {
      // Token expired or invalid — clear and reload to login screen
      token.clear();
      window.location.reload();
      return { data: null, error: 'Session expired. Please log in again.' };
    }

    if (!res.ok) throw new Error(data?.detail || data || `HTTP ${res.status}`);
    return { data, error: null };
  } catch (e) {
    return { data: null, error: e.message };
  }
}

// ── Auth ──────────────────────────────────────────────────────────────────
export const auth = {
  login: (username, password) => {
    // OAuth2 form login — must be application/x-www-form-urlencoded
    const body = new URLSearchParams({ username, password });
    return request('POST', '/auth/login', {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body,
    });
  },
  me: () => request('GET', '/auth/me'),
};

// ── API ───────────────────────────────────────────────────────────────────
export const api = {
  listDocs:    ()       => request('GET',    '/docs'),
  deleteDoc:   (id)     => request('DELETE', `/docs/${id}`),
  upload:      (file)   => { const fd = new FormData(); fd.append('file', file); return request('POST', '/upload', { body: fd }); },
  ask:         (doc_id, q) => request('POST', '/ask',   { headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ doc_id, question: q }) }),
  extractFile: (file)   => { const fd = new FormData(); fd.append('file', file); return request('POST', '/extract-file', { body: fd }); },
};

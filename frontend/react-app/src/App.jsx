import { useState, useEffect } from 'react';
import Dashboard from './pages/Dashboard';
import Train     from './pages/Train';
import Ask       from './pages/Ask';
import Extract   from './pages/Extract';
import Login     from './pages/Login';
import { token, auth } from './api';
import './App.css';

const PAGE_LABELS = {
  dashboard: null,
  train:     'Train',
  ask:       'Ask Queries',
  extract:   'Direct Extract',
};

export default function App() {
  const [page, setPage]   = useState('dashboard');
  const [user, setUser]   = useState(null);   // null = not logged in
  const [checking, setChecking] = useState(true); // validating stored token

  // On mount: if a token is stored, verify it's still valid
  useEffect(() => {
    if (!token.get()) { setChecking(false); return; }
    auth.me()
      .then(({ data }) => {
        if (data?.username) setUser(data);
        else token.clear();
      })
      .catch(() => token.clear())
      .finally(() => setChecking(false));
  }, []);

  function handleLogin(userData) {
    setUser(userData);
    setPage('dashboard');
  }

  function handleLogout() {
    token.clear();
    setUser(null);
    setPage('dashboard');
  }

  const navigate = (p) => setPage(p);

  // Still verifying stored token — show nothing to avoid flash
  if (checking) return null;

  // Not authenticated — show login
  if (!user) return <Login onLogin={handleLogin} />;

  return (
    <div className="app-shell">
      <header className="app-header">
        <div
          className="app-logo"
          style={{ cursor: page !== 'dashboard' ? 'pointer' : 'default' }}
          onClick={() => navigate('dashboard')}
        >
          <div className="logo-icon">✈</div>
          <span className="logo-name">Logistics AI</span>
        </div>

        <div className="app-header-right">
          {page !== 'dashboard' && (
            <div className="breadcrumb">
              <span>Dashboard</span>
              <span className="breadcrumb-sep">›</span>
              <span className="breadcrumb-current">{PAGE_LABELS[page]}</span>
            </div>
          )}
          {page !== 'dashboard' && (
            <button className="btn btn-ghost btn-sm" onClick={() => navigate('dashboard')}>
              ← Back
            </button>
          )}
          <div className="user-chip">
            <span className="user-avatar">{user.display_name?.[0]?.toUpperCase() ?? '?'}</span>
            <span className="user-name">{user.display_name}</span>
          </div>
          <button className="btn btn-ghost btn-sm" onClick={handleLogout}>
            Sign out
          </button>
        </div>
      </header>

      <main className="app-main">
        {page === 'dashboard' && <Dashboard navigate={navigate} />}
        {page === 'train'     && <Train     navigate={navigate} />}
        {page === 'ask'       && <Ask       navigate={navigate} />}
        {page === 'extract'   && <Extract   navigate={navigate} />}
      </main>
    </div>
  );
}

import { useEffect, useState } from 'react';

interface HealthResponse {
  status: string;
  message: string;
  system: string;
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetch('/api/health')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP error! status: ${res.status}`);
        return res.json();
      })
      .then((data: HealthResponse) => {
        setHealth(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return (
    <div className="app-container">
      <header className="header">
        <div className="title-group">
          <h1>Smart Multi-Agent Traffic System</h1>
          <p>Phase 1 Foundation & Dashboard Dashboard</p>
        </div>
        <div className="status-badge">
          <span
            className={`status-dot ${
              loading ? 'loading' : health ? 'ok' : 'error'
            }`}
          />
          {loading
            ? 'Checking Backend...'
            : health
            ? 'Backend Connected'
            : 'Backend Offline'}
        </div>
      </header>

      <main className="dashboard-grid">
        <div className="card">
          <h3>Backend Health Check</h3>
          <div className="card-content">
            {loading && <p>Fetching API status from <code>/api/health</code>...</p>}
            {error && (
              <p style={{ color: 'var(--accent-rose)' }}>
                Connection failed: {error}. Launch the backend server to connect.
              </p>
            )}
            {health && (
              <div>
                <p><strong>Status:</strong> {health.status.toUpperCase()}</p>
                <p><strong>System:</strong> {health.system}</p>
                <p><strong>Response Message:</strong> {health.message}</p>
              </div>
            )}
            <div className="endpoint-box">GET /api/health</div>
          </div>
        </div>

        <div className="card">
          <h3>Architecture Progress</h3>
          <div className="card-content">
            <p>
              Phase 1 foundation is initialized with directory structures, FastAPI service, React + Vite dashboard, and Living Architecture documentation.
            </p>
          </div>
        </div>

        <div className="card">
          <h3>Simulation Engine Status</h3>
          <div className="card-content">
            <p>
              Ready for Phase 2: Agent-based grid simulation, traffic lights controller, and vehicle dynamics.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}

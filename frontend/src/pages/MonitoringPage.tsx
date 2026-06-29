import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { getMonitoring } from '../services/api';

export default function MonitoringPage() {
  const { id } = useParams<{ id: string }>();
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    if (!id) return;
    const poll = () => getMonitoring(id).then(setData).catch(() => {});
    poll();
    const interval = setInterval(poll, 5000);
    return () => clearInterval(interval);
  }, [id]);

  if (!data) return <p className="text-forge-muted">Loading...</p>;

  if (!data.deployed) {
    return (
      <div>
        <Link to={`/experiments/${id}/deploy`} className="text-sm text-forge-muted hover:text-white">← Deploy first</Link>
        <div className="card mt-4"><p>Model not deployed yet.</p></div>
      </div>
    );
  }

  const perf = data.performance || {};

  return (
    <div className="space-y-6">
      <Link to={`/experiments/${id}/deploy`} className="text-sm text-forge-muted hover:text-white">← Back to deployment</Link>
      <h1 className="text-3xl font-bold">Monitoring</h1>

      <div className="grid md:grid-cols-3 gap-4">
        <Stat label="Total Requests" value={String(perf.total_requests ?? 0)} />
        <Stat label="Error Rate" value={`${((perf.error_rate ?? 0) * 100).toFixed(1)}%`} />
        <Stat label="P95 Latency" value={`${(perf.latency_p95_ms ?? 0).toFixed(1)} ms`} />
      </div>

      <div className="card">
        <h2 className="text-lg font-semibold mb-4">Latency</h2>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div><p className="text-xs text-forge-muted">P50</p><p className="text-xl font-mono">{(perf.latency_p50_ms ?? 0).toFixed(1)} ms</p></div>
          <div><p className="text-xs text-forge-muted">P95</p><p className="text-xl font-mono">{(perf.latency_p95_ms ?? 0).toFixed(1)} ms</p></div>
          <div><p className="text-xs text-forge-muted">P99</p><p className="text-xl font-mono">{(perf.latency_p99_ms ?? 0).toFixed(1)} ms</p></div>
        </div>
      </div>

      {perf.prediction_distribution && (
        <div className="card">
          <h2 className="text-lg font-semibold mb-4">Prediction Distribution</h2>
          {Object.entries(perf.prediction_distribution).map(([k, v]) => (
            <div key={k} className="flex justify-between py-1 text-sm">
              <span>{k}</span>
              <span className="font-mono">{String(v)}</span>
            </div>
          ))}
        </div>
      )}

      <div className="card">
        <h2 className="text-lg font-semibold mb-2">Drift Monitoring</h2>
        <p className="text-sm text-forge-muted">
          Baseline set from training data. PSI threshold: 0.25. Upload new data via API to trigger drift checks.
        </p>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="card">
      <p className="text-xs text-forge-muted uppercase">{label}</p>
      <p className="text-2xl font-bold mt-1">{value}</p>
    </div>
  );
}

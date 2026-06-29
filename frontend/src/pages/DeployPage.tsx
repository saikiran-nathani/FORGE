import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { deployExperiment, getExperiment, predict, getMonitoring } from '../services/api';

export default function DeployPage() {
  const { id } = useParams<{ id: string }>();
  const [exp, setExp] = useState<any>(null);
  const [deploying, setDeploying] = useState(false);
  const [deployed, setDeployed] = useState(false);
  const [deployment, setDeployment] = useState<any>(null);
  const [monitoring, setMonitoring] = useState<any>(null);

  useEffect(() => {
    if (!id) return;
    getExperiment(id).then(setExp);
  }, [id]);

  async function handleDeploy() {
    if (!id) return;
    setDeploying(true);
    try {
      const result = await deployExperiment(id);
      setDeployment(result);
      setDeployed(true);
      const mon = await getMonitoring(id);
      setMonitoring(mon);
    } finally {
      setDeploying(false);
    }
  }

  if (!exp) return <p className="text-forge-muted">Loading...</p>;

  return (
    <div className="space-y-6">
      <Link to={`/experiments/${id}`} className="text-sm text-forge-muted hover:text-white">← Back to experiment</Link>
      <h1 className="text-3xl font-bold">Deployment</h1>

      {exp.status !== 'completed' ? (
        <div className="card"><p>Experiment must complete before deployment.</p></div>
      ) : !deployed ? (
        <div className="card space-y-4">
          <p className="text-forge-muted">Deploy the best model as a production API with monitoring.</p>
          <ul className="text-sm space-y-1 text-forge-muted">
            <li>• FastAPI serving endpoint</li>
            <li>• Docker container + docker-compose</li>
            <li>• Model card documentation</li>
            <li>• Drift monitoring baseline</li>
            <li>• Airflow batch scoring DAG</li>
          </ul>
          <button onClick={handleDeploy} disabled={deploying} className="btn-primary">
            {deploying ? 'Deploying...' : 'Deploy Model'}
          </button>
        </div>
      ) : (
        <>
          <div className="card">
            <h2 className="text-lg font-semibold mb-3">Deployment Info</h2>
            <p>ID: <code className="text-forge-accent">{deployment?.deployment_id}</code></p>
            <p className="mt-2">API: <code>{deployment?.api_url}</code></p>
            <div className="flex gap-4 mt-4">
              <Link to={`/experiments/${id}/playground`} className="btn-primary">Prediction Playground</Link>
              <Link to={`/experiments/${id}/monitoring`} className="text-forge-accent hover:underline">Monitoring →</Link>
              <a href={`/api/v1/experiments/${id}/model-card`} target="_blank" rel="noreferrer" className="text-forge-accent hover:underline">Model Card</a>
            </div>
          </div>
          {monitoring?.performance && (
            <div className="card">
              <h2 className="text-lg font-semibold mb-3">Performance</h2>
              <p>Requests: {monitoring.performance.total_requests ?? 0}</p>
              <p>P50 latency: {monitoring.performance.latency_p50_ms?.toFixed(1) ?? '—'} ms</p>
            </div>
          )}
        </>
      )}
    </div>
  );
}

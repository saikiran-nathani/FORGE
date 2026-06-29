import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Experiment, listExperiments } from '../services/api';

export default function HomePage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);

  useEffect(() => {
    listExperiments().then(setExperiments).catch(() => setExperiments([]));
    const interval = setInterval(() => {
      listExperiments().then(setExperiments).catch(() => {});
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">Experiments</h1>
          <p className="text-forge-muted mt-1">LLM-powered automated ML pipelines</p>
        </div>
        <Link to="/new" className="btn-primary">New Experiment</Link>
      </div>

      {experiments.length === 0 ? (
        <div className="card text-center py-16">
          <p className="text-forge-muted mb-4">No experiments yet</p>
          <Link to="/new" className="btn-primary inline-block">Upload a dataset</Link>
        </div>
      ) : (
        <div className="grid gap-4">
          {experiments.map((exp) => (
            <Link key={exp.id} to={`/experiments/${exp.id}`} className="card hover:border-forge-accent/50 transition block">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="font-semibold text-lg">{exp.name}</h2>
                  <p className="text-sm text-forge-muted">Target: {exp.target_column}</p>
                </div>
                <StatusBadge status={exp.status} />
              </div>
              {exp.status === 'running' && (
                <p className="text-sm text-forge-muted mt-2">{exp.progress}</p>
              )}
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const colors: Record<string, string> = {
    pending: 'bg-slate-600',
    running: 'bg-amber-500 text-black',
    completed: 'bg-forge-success text-black',
    failed: 'bg-red-500',
  };
  return (
    <span className={`px-3 py-1 rounded-full text-xs font-medium ${colors[status] || 'bg-slate-600'}`}>
      {status}
    </span>
  );
}

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
      <div className="flex items-end justify-between mb-10 flex-wrap gap-4">
        <div>
          <p className="kicker">The yard</p>
          <h1 className="font-display text-4xl font-bold mt-3">Experiments</h1>
          <p className="text-forge-steel mt-2">Every model FORGE has built.</p>
        </div>
        <Link to="/new" className="btn-primary">Forge a model →</Link>
      </div>

      {experiments.length === 0 ? (
        <div className="card text-center py-20">
          <div className="h-12 w-12 rounded-xl border border-forge-accent/30 flex items-center justify-center mx-auto mb-5">
            <span className="h-3 w-3 rounded-sm bg-forge-accent shadow-[0_0_12px_2px_rgba(255,90,30,0.7)]" />
          </div>
          <p className="text-forge-steel mb-5 font-mono text-sm">the yard is empty</p>
          <Link to="/new" className="btn-primary inline-flex">Forge your first model</Link>
        </div>
      ) : (
        <div className="grid gap-4">
          {experiments.map((exp) => (
            <Link
              key={exp.id}
              to={`/experiments/${exp.id}`}
              className="card flex items-center justify-between gap-4 group"
            >
              <div className="min-w-0">
                <div className="flex items-center gap-3">
                  <h2 className="font-display font-semibold text-lg text-forge-hot truncate">{exp.name}</h2>
                  {exp.id === 'demo' && (
                    <span className="font-mono text-[0.6rem] tracking-widest uppercase px-2 py-0.5 rounded border border-forge-accent/40 text-forge-amber">demo</span>
                  )}
                </div>
                <p className="font-mono text-xs text-forge-steel/80 mt-1">target · {exp.target_column}</p>
                {exp.status === 'running' && exp.progress && (
                  <p className="text-sm text-forge-amber/90 mt-2">{exp.progress}</p>
                )}
              </div>
              <StatusBadge status={exp.status} />
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

function StatusBadge({ status }: { status: string }) {
  const map: Record<string, string> = {
    pending: 'border-forge-line text-forge-steel',
    running: 'border-forge-amber/50 text-forge-amber',
    completed: 'border-forge-success/50 text-forge-success',
    failed: 'border-red-500/50 text-red-400',
  };
  return (
    <span className={`shrink-0 font-mono text-[0.65rem] tracking-widest uppercase px-3 py-1.5 rounded-full border ${map[status] || map.pending}`}>
      {status}
    </span>
  );
}

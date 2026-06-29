import { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Experiment, getExperiment, reportUrl } from '../services/api';

export default function ExperimentPage() {
  const { id } = useParams<{ id: string }>();
  const [exp, setExp] = useState<Experiment | null>(null);

  useEffect(() => {
    if (!id) return;
    const poll = () => getExperiment(id).then(setExp).catch(() => {});
    poll();
    const interval = setInterval(poll, 2000);
    return () => clearInterval(interval);
  }, [id]);

  if (!exp) return <p className="text-forge-muted">Loading...</p>;

  const result = exp.result || {};
  const models = (result.model_results as Array<{ model_name: string; cv_score: number }>) || [];
  const metrics = (result.best_metrics as Record<string, number>) || {};
  const shap = (result.shap_summary as { top_features?: Array<{ feature: string; mean_abs_shap: number }> }) || {};
  const features = (result.generated_features as Array<{ source_column: string; new_columns: string[] }>) || [];

  return (
    <div className="space-y-8">
      <div className="flex items-start justify-between">
        <div>
          <Link to="/" className="text-sm text-forge-muted hover:text-white">← Back</Link>
          <h1 className="text-3xl font-bold mt-2">{exp.name}</h1>
          <p className="text-forge-muted">{exp.task_description || 'No description'}</p>
        </div>
        <span className={`px-3 py-1 rounded-full text-sm font-medium ${
          exp.status === 'completed' ? 'bg-forge-success text-black' :
          exp.status === 'running' ? 'bg-amber-500 text-black' :
          exp.status === 'failed' ? 'bg-red-500' : 'bg-slate-600'
        }`}>{exp.status}</span>
      </div>

      {exp.status === 'running' && (
        <div className="card">
          <div className="animate-pulse text-forge-accent font-medium">{exp.progress || 'Running pipeline...'}</div>
        </div>
      )}

      {exp.status === 'failed' && (
        <div className="card border-red-500/50">
          <p className="text-red-400">{exp.error}</p>
        </div>
      )}

      {exp.status === 'completed' && (
        <>
          <div className="grid md:grid-cols-3 gap-4">
            <StatCard label="Best Model" value={String(result.best_model_name || '—')} />
            <StatCard label="Quality Score" value={`${result.quality_score || '—'}/100`} />
            <StatCard label="Task Type" value={String(result.task_type || '—')} />
          </div>

          {id && (
            <div className="card flex items-center justify-between">
              <div>
                <h2 className="text-lg font-semibold">Deploy Model</h2>
                <p className="text-sm text-forge-muted">One-click production deployment with monitoring</p>
              </div>
              <Link to={`/experiments/${id}/deploy`} className="btn-primary">Deploy →</Link>
            </div>
          )}

          {id && (
            <div className="card">
              <h2 className="text-lg font-semibold mb-3">EDA Report</h2>
              <a href={reportUrl(id)} target="_blank" rel="noreferrer" className="text-forge-accent hover:underline">
                Open interactive EDA report →
              </a>
            </div>
          )}

          <div className="card">
            <h2 className="text-lg font-semibold mb-4">Test Metrics</h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              {Object.entries(metrics).filter(([k]) => k !== 'confusion_matrix').map(([k, v]) => (
                <div key={k}>
                  <p className="text-xs text-forge-muted uppercase">{k.replace(/_/g, ' ')}</p>
                  <p className="text-xl font-mono">{typeof v === 'number' ? v.toFixed(4) : String(v)}</p>
                </div>
              ))}
            </div>
          </div>

          <div className="card">
            <h2 className="text-lg font-semibold mb-4">Model Comparison</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-forge-muted border-b border-slate-700">
                    <th className="text-left py-2">Model</th>
                    <th className="text-right py-2">CV Score</th>
                  </tr>
                </thead>
                <tbody>
                  {models.sort((a, b) => b.cv_score - a.cv_score).map((m) => (
                    <tr key={m.model_name} className="border-b border-slate-800">
                      <td className="py-2">{m.model_name}{m.model_name === result.best_model_name ? ' ★' : ''}</td>
                      <td className="text-right font-mono">{m.cv_score.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {features.length > 0 && (
            <div className="card">
              <h2 className="text-lg font-semibold mb-4">LLM-Generated Features</h2>
              <ul className="space-y-2 text-sm">
                {features.map((f, i) => (
                  <li key={i} className="text-forge-muted">
                    <span className="text-white">{f.source_column}</span> → {f.new_columns.join(', ')}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {shap.top_features && (
            <div className="card">
              <h2 className="text-lg font-semibold mb-4">SHAP Feature Importance</h2>
              <div className="space-y-2">
                {shap.top_features.slice(0, 10).map((f) => (
                  <div key={f.feature} className="flex items-center gap-3">
                    <span className="text-sm w-48 truncate">{f.feature}</span>
                    <div className="flex-1 bg-slate-800 rounded-full h-2">
                      <div
                        className="bg-forge-accent h-2 rounded-full"
                        style={{ width: `${Math.min(100, f.mean_abs_shap * 500)}%` }}
                      />
                    </div>
                    <span className="text-xs font-mono text-forge-muted">{f.mean_abs_shap.toFixed(4)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {(result.pareto_frontier as Array<{ model_name: string; cv_score: number; latency_ms: number; is_pareto_optimal: boolean }>)?.length > 0 && (
            <div className="card">
              <h2 className="text-lg font-semibold mb-4">Pareto Frontier (Accuracy vs Latency)</h2>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-forge-muted border-b border-slate-700">
                    <th className="text-left py-2">Model</th>
                    <th className="text-right py-2">CV Score</th>
                    <th className="text-right py-2">Latency (ms)</th>
                    <th className="text-right py-2">Pareto</th>
                  </tr>
                </thead>
                <tbody>
                  {(result.pareto_frontier as Array<{ model_name: string; cv_score: number; latency_ms: number; is_pareto_optimal: boolean }>).map((p) => (
                    <tr key={p.model_name} className="border-b border-slate-800">
                      <td className="py-2">{p.model_name}</td>
                      <td className="text-right font-mono">{p.cv_score.toFixed(4)}</td>
                      <td className="text-right font-mono">{p.latency_ms.toFixed(1)}</td>
                      <td className="text-right">{p.is_pareto_optimal ? '★' : ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {(result.error_analysis as { worst_predictions?: unknown[] })?.worst_predictions && (
            <div className="card">
              <h2 className="text-lg font-semibold mb-4">Error Analysis</h2>
              <p className="text-sm text-forge-muted">
                {(result.error_analysis as { worst_predictions: unknown[] }).worst_predictions.length} worst predictions analyzed
              </p>
            </div>
          )}

          {id && (
            <div className="card">
              <h2 className="text-lg font-semibold mb-3">Analysis Report</h2>
              <a href={`/api/v1/experiments/${id}/report?format=analysis`} target="_blank" rel="noreferrer" className="text-forge-accent hover:underline">
                Open LLM analysis report →
              </a>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="card">
      <p className="text-xs text-forge-muted uppercase">{label}</p>
      <p className="text-2xl font-bold mt-1">{value}</p>
    </div>
  );
}

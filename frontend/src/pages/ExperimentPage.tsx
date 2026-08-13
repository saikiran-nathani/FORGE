import { useEffect, useRef, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { Experiment, getExperiment, reportUrl } from '../services/api';

const PIPELINE = [
  { key: 'profiling', label: 'Profile', desc: 'reading & profiling the data' },
  { key: 'feature_engineering', label: 'Engineer', desc: 'features + selection' },
  { key: 'training', label: 'Train & tune', desc: 'models + Bayesian HPO' },
  { key: 'evaluation', label: 'Evaluate', desc: 'SHAP · errors · fairness' },
  { key: 'finalizing', label: 'Package', desc: 'bundling the model' },
];

function fmtElapsed(sec: number) {
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

// Display order for the metrics table; unknown metrics fall to the end.
const METRIC_ORDER = [
  'balanced_accuracy', 'accuracy', 'f1', 'f1_macro', 'f1_weighted', 'mcc', 'cohen_kappa',
  'roc_auc', 'pr_auc', 'precision_macro', 'recall_macro', 'brier_score', 'log_loss', 'ece',
  'rmse', 'mae', 'r2', 'adjusted_r2', 'mape', 'max_error',
];
const fmtNum = (v: unknown) => (typeof v === 'number' ? v.toFixed(4) : v == null ? '—' : String(v));

export default function ExperimentPage() {
  const { id } = useParams<{ id: string }>();
  const [exp, setExp] = useState<Experiment | null>(null);
  const [notFound, setNotFound] = useState(false);

  useEffect(() => {
    if (!id) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const e = await getExperiment(id);
        setExp(e);
        if (e.status === 'completed' || e.status === 'failed') return; // terminal — stop polling
      } catch {
        setNotFound(true);
        return;
      }
      if (!stopped) timer = setTimeout(poll, 2000);
    };
    poll();
    return () => { stopped = true; clearTimeout(timer); };
  }, [id]);

  if (notFound) {
    return (
      <div className="card text-center py-16">
        <p className="kicker text-forge-amber/90">404 · cooled down</p>
        <h1 className="font-display text-2xl font-bold mt-3">This experiment isn’t in the yard</h1>
        <p className="text-forge-steel mt-2 font-mono text-sm">It may have been cleared when the backend restarted.</p>
        <Link to="/experiments" className="btn-primary inline-flex mt-6">← Back to the yard</Link>
      </div>
    );
  }

  if (!exp) return <p className="text-forge-steel font-mono text-sm">Loading…</p>;

  const result = exp.result || {};
  const models = (result.model_results as Array<{ model_name: string; cv_score: number }>) || [];
  const metrics = (result.best_metrics as Record<string, number>) || {};
  type SpecField = { value: unknown; source?: string; rationale?: string };
  const plan = (result.task_plan as {
    selection_metric?: string; metric_rationale?: string; data_metric?: string;
    requested_metric?: string; unoptimizable_reason?: string | null;
    decision_threshold?: number; task_spec?: Record<string, SpecField>;
  }) || {};
  const specFields: Array<[string, SpecField]> = Object.entries(plan.task_spec || {})
    .filter(([, f]) => f && typeof f === 'object' && 'value' in f)
    .filter(([, f]) => f.source === 'stated');
  const fmtSpec = (v: unknown) =>
    v === null || v === undefined ? '—'
      : Array.isArray(v) ? (v.length ? v.join(', ') : '—')
      : typeof v === 'object' ? Object.entries(v as Record<string, unknown>).map(([k, x]) => `${k}:${x}`).join(' / ')
      : String(v);

  type ParetoRow = {
    model_name: string; cv_score: number; latency_ms: number;
    is_pareto_optimal: boolean; tied_with_leader?: boolean;
  };
  const sig = (result.significance as {
    status?: string;
    reason?: string;
    metric?: string;
    leader?: string;
    alpha?: number;
    n_units?: number;
    tied_with_leader?: string[];
    fastest_tied_model?: string;
    fastest_tied_latency_ms?: number;
    per_model?: Record<string, {
      mean: number; diff_vs_leader: number; ci_lo: number; ci_hi: number;
      distinguishable_from_leader: boolean;
    }>;
  }) || {};
  const shap = (result.shap_summary as {
    top_features?: Array<{ feature: string; mean_abs_shap: number }>;
    shap_status?: string;
    shap_error?: string;
  }) || {};
  const features = (result.generated_features as Array<{ source_column: string; new_columns: string[] }>) || [];
  const baseline = (result.baseline_metrics as Record<string, number>) || {};
  const ctx = (result.eval_context as Record<string, any>) || {};
  const warnings = (result.warnings as string[]) || [];
  const confusion = (metrics as any).confusion_matrix as number[][] | undefined;
  const metricKeys = Array.from(new Set([...Object.keys(metrics), ...Object.keys(baseline)]))
    .filter((k) => k !== 'confusion_matrix' && k !== 'error')
    .sort((a, b) => ((METRIC_ORDER.indexOf(a) + 1) || 99) - ((METRIC_ORDER.indexOf(b) + 1) || 99));
  const hasBaseline = Object.keys(baseline).length > 0 && !('error' in baseline);

  return (
    <div className="space-y-8">
      <div className="flex items-start justify-between gap-4">
        <div>
          <Link to="/experiments" className="kicker hover:text-forge-hot transition-colors">← the yard</Link>
          <h1 className="font-display text-3xl font-bold mt-3">{exp.name}</h1>
          <p className="text-forge-steel mt-1">{exp.task_description || 'No description'}</p>
        </div>
        <StatusBadge status={exp.status} />
      </div>

      {(exp.status === 'running' || exp.status === 'pending') && <RunningView exp={exp} />}

      {exp.status === 'failed' && (
        <div className="card border-red-500/40">
          <p className="kicker text-red-400/90">the forge stalled</p>
          <p className="text-red-300 mt-3 leading-relaxed">{exp.error || 'The pipeline failed unexpectedly.'}</p>
          <Link to="/new" className="btn-ghost inline-flex mt-5 text-sm">↻ Try another run</Link>
        </div>
      )}

      {exp.status === 'completed' && (
        <>
          {warnings.length > 0 && (
            <div className="flex flex-col gap-2">
              {warnings.map((w, i) => (
                <div key={i} className="flex items-start gap-2.5 rounded-lg border border-forge-amber/40 bg-forge-amber/[0.06] px-3.5 py-2.5">
                  <span className="text-forge-amber mt-0.5">⚠</span>
                  <span className="font-mono text-xs text-forge-amber/90 leading-relaxed">{w}</span>
                </div>
              ))}
            </div>
          )}

          <div className="grid md:grid-cols-3 gap-4">
            <StatCard label={`Selected model${ctx.selection_metric ? ` · by cv ${ctx.selection_metric}` : ''}`} value={String(result.best_model_name || '—')} />
            <StatCard label="Task type" value={String(result.task_type || '—')} />
            <StatCard label="Dataset" value={`${ctx.n_train ?? '?'} train · ${ctx.n_test ?? '?'} test`} />
          </div>

          <div className="card">
            <div className="flex flex-wrap gap-x-8 gap-y-2 font-mono text-xs text-forge-steel">
              {ctx.positive_class != null && <span>positive class · <span className="text-forge-hot">{String(ctx.positive_class)}</span></span>}
              {ctx.majority_fraction != null && <span>base rate · <span className="text-forge-hot">{(Number(ctx.majority_fraction) * 100).toFixed(0)}%</span> majority</span>}
              {ctx.test_class_counts && <span>test split · <span className="text-forge-hot">{Object.entries(ctx.test_class_counts).map(([k, v]) => `${v} ${k}`).join(' / ')}</span></span>}
              {ctx.cv_best_score != null && ctx.test_metric_value != null && (
                <span>{ctx.selection_metric} · CV <span className="text-forge-hot">{Number(ctx.cv_best_score).toFixed(3)}</span> → test <span className="text-forge-hot">{Number(ctx.test_metric_value).toFixed(3)}</span></span>
              )}
              {result.quality_score != null && <span>data quality · {String(result.quality_score)}/100</span>}
            </div>
          </div>

          {id && (
            <div className="card flex items-center justify-between gap-4">
              <div>
                <h2 className="font-display text-lg font-semibold text-forge-hot">Deploy model</h2>
                <p className="text-sm text-forge-steel">One-click production deployment. Available regardless of scores — you judge the numbers.</p>
              </div>
              <Link to={`/experiments/${id}/deploy`} className="btn-primary">Deploy →</Link>
            </div>
          )}

          {id && (
            <div className="card">
              <h2 className="font-display text-lg font-semibold mb-3 text-forge-hot">EDA report</h2>
              <a href={reportUrl(id)} target="_blank" rel="noreferrer" className="text-forge-accent hover:text-forge-amber transition-colors font-mono text-sm">
                Open interactive EDA report →
              </a>
            </div>
          )}

          <div className="card">
            <h2 className="font-display text-lg font-semibold mb-1 text-forge-hot">Test metrics</h2>
            <p className="font-mono text-[0.7rem] text-forge-steel/70 mb-4">
              your model vs a trivial baseline ({result.task_type === 'regression' ? 'mean prediction' : 'always predict majority class'}) on the same hold-out split — compare and judge
            </p>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="kicker text-forge-steel/70 border-b border-forge-line">
                    <th className="text-left py-2 font-normal">Metric</th>
                    <th className="text-right py-2 font-normal">Model</th>
                    {hasBaseline && <th className="text-right py-2 font-normal">Baseline</th>}
                  </tr>
                </thead>
                <tbody>
                  {metricKeys.map((k) => (
                    <tr key={k} className="border-b border-forge-line/60">
                      <td className="py-2 text-forge-steel">{k.replace(/_/g, ' ')}</td>
                      <td className="text-right py-2 font-mono text-forge-hot">{fmtNum((metrics as any)[k])}</td>
                      {hasBaseline && <td className="text-right py-2 font-mono text-forge-steel/80">{fmtNum((baseline as any)[k])}</td>}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {confusion && Array.isArray(confusion) && confusion.length > 0 && (
            <div className="card">
              <h2 className="font-display text-lg font-semibold mb-4 text-forge-hot">Confusion matrix</h2>
              <ConfusionMatrix matrix={confusion} labels={ctx.test_class_counts ? Object.keys(ctx.test_class_counts) : undefined} />
            </div>
          )}

          <div className="card">
            <h2 className="font-display text-lg font-semibold mb-4 text-forge-hot">Model comparison</h2>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="kicker text-forge-steel/70 border-b border-forge-line">
                    <th className="text-left py-2 font-normal">Model</th>
                    <th className="text-right py-2 font-normal">CV Score</th>
                  </tr>
                </thead>
                <tbody>
                  {models.sort((a, b) => b.cv_score - a.cv_score).map((m) => (
                    <tr key={m.model_name} className="border-b border-forge-line/60">
                      <td className="py-2.5">
                        {m.model_name}
                        {m.model_name === result.best_model_name && <span className="text-forge-accent"> ★</span>}
                      </td>
                      <td className="text-right font-mono text-forge-hot">{m.cv_score.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {features.length > 0 && (
            <div className="card">
              <h2 className="font-display text-lg font-semibold mb-4 text-forge-hot">Generated features</h2>
              <ul className="space-y-2 text-sm">
                {features.map((f, i) => (
                  <li key={i} className="text-forge-steel font-mono text-xs">
                    <span className="text-forge-hot">{f.source_column}</span> → {f.new_columns.join(', ')}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {(shap.top_features || shap.shap_status) && (
            <div className="card">
              <h2 className="font-display text-lg font-semibold mb-4 text-forge-hot">SHAP feature importance</h2>
              {shap.top_features && shap.top_features.length > 0 ? (
                <div className="space-y-2.5">
                  {shap.top_features.slice(0, 10).map((f) => (
                    <div key={f.feature} className="flex items-center gap-3">
                      <span className="text-sm w-48 truncate font-mono text-forge-steel">{f.feature}</span>
                      <div className="flex-1 bg-[#0e0d11] rounded-full h-2 border border-forge-line">
                        <div
                          className="h-full rounded-full"
                          style={{ width: `${Math.min(100, f.mean_abs_shap * 500)}%`, background: 'var(--ember-grad)' }}
                        />
                      </div>
                      <span className="text-xs font-mono text-forge-steel/80">{f.mean_abs_shap.toFixed(4)}</span>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-forge-steel/70">
                  SHAP explanations unavailable for this run{shap.shap_error ? ` — ${shap.shap_error}` : ''}.
                </p>
              )}
            </div>
          )}

          {plan.selection_metric && (
            <div className="card">
              <h2 className="font-display text-lg font-semibold mb-1 text-forge-hot">
                How your description shaped the run
              </h2>
              <p className="text-sm text-forge-steel/80 mb-4">
                The data decides the task type and imbalance. Only your words can state what an
                error <em>costs</em> — so FORGE reads them, shows you what it understood, and lets
                you correct it.
              </p>
              <div className="flex flex-wrap gap-x-8 gap-y-2 text-sm mb-3">
                <span>
                  <span className="kicker text-forge-steel/70">metric optimized</span>{' '}
                  <span className="font-mono text-forge-hot">{plan.selection_metric}</span>
                </span>
                {plan.data_metric && plan.data_metric !== plan.selection_metric && (
                  <span>
                    <span className="kicker text-forge-steel/70">from data alone</span>{' '}
                    <span className="font-mono text-forge-steel/70">{plan.data_metric}</span>
                  </span>
                )}
                {typeof plan.decision_threshold === 'number' && (
                  <span>
                    <span className="kicker text-forge-steel/70">decision threshold</span>{' '}
                    <span className="font-mono text-forge-hot">{plan.decision_threshold.toFixed(3)}</span>
                    {plan.decision_threshold !== 0.5 && (
                      <span className="text-forge-steel/60"> (cost-weighted, not 0.5)</span>
                    )}
                  </span>
                )}
              </div>
              {plan.metric_rationale && (
                <p className="text-sm text-forge-steel/80 mb-3">{plan.metric_rationale}</p>
              )}
              {plan.unoptimizable_reason && (
                <p className="text-sm text-forge-amber/90 mb-3">Note: {plan.unoptimizable_reason}.</p>
              )}
              {specFields.length > 0 ? (
                <table className="w-full text-sm">
                  <thead>
                    <tr className="kicker text-forge-steel/70 border-b border-forge-line">
                      <th className="text-left py-2 font-normal">Read from your description</th>
                      <th className="text-left py-2 font-normal">Value</th>
                      <th className="text-left py-2 font-normal">Why</th>
                    </tr>
                  </thead>
                  <tbody>
                    {specFields.map(([name, f]) => (
                      <tr key={name} className="border-b border-forge-line/60 align-top">
                        <td className="py-2.5 font-mono text-xs">{name}</td>
                        <td className="py-2.5 font-mono text-forge-hot">{fmtSpec(f.value)}</td>
                        <td className="py-2.5 text-forge-steel/70 text-xs">{f.rationale}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : (
                <p className="text-sm text-forge-steel/60">
                  Nothing specific was stated in the description, so every choice above came from the
                  data and documented defaults.
                </p>
              )}
            </div>
          )}

          {sig.status === 'ok' && sig.per_model && (
            <div className="card">
              <h2 className="font-display text-lg font-semibold mb-1 text-forge-hot">
                Is the winner actually better?
              </h2>
              <p className="text-sm text-forge-steel/80 mb-4">
                Paired bootstrap over {sig.n_units} held-out rows ({Math.round((1 - (sig.alpha ?? 0.05)) * 100)}% CI on the
                difference in <span className="font-mono">{sig.metric}</span> vs <span className="font-mono">{sig.leader}</span>).
                A model is <strong>tied</strong> when its interval contains zero — the gap is sampling noise, not skill.
              </p>
              <table className="w-full text-sm">
                <thead>
                  <tr className="kicker text-forge-steel/70 border-b border-forge-line">
                    <th className="text-left py-2 font-normal">Model</th>
                    <th className="text-right py-2 font-normal">{sig.metric}</th>
                    <th className="text-right py-2 font-normal">Δ vs leader</th>
                    <th className="text-right py-2 font-normal">95% CI</th>
                    <th className="text-right py-2 font-normal">Verdict</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(sig.per_model).map(([name, m]) => (
                    <tr key={name} className="border-b border-forge-line/60">
                      <td className="py-2.5">{name}{name === sig.leader && <span className="text-forge-accent"> ★</span>}</td>
                      <td className="text-right font-mono text-forge-hot">{m.mean.toFixed(4)}</td>
                      <td className="text-right font-mono text-forge-steel">{m.diff_vs_leader >= 0 ? '+' : ''}{m.diff_vs_leader.toFixed(4)}</td>
                      <td className="text-right font-mono text-forge-steel/70 text-xs">
                        [{m.ci_lo.toFixed(4)}, {m.ci_hi.toFixed(4)}]
                      </td>
                      <td className="text-right">
                        {m.distinguishable_from_leader
                          ? <span className="text-forge-steel/60">worse</span>
                          : <span className="text-forge-accent">tied</span>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {sig.fastest_tied_model && sig.fastest_tied_model !== result.best_model_name && (
                <p className="mt-4 text-sm text-forge-steel/80">
                  <span className="text-forge-accent">Worth knowing:</span>{' '}
                  <span className="font-mono">{sig.fastest_tied_model}</span> is statistically tied with the
                  selected model but runs at {sig.fastest_tied_latency_ms?.toFixed(2)}ms. FORGE reports this —
                  the choice is yours.
                </p>
              )}
            </div>
          )}

          {sig.status === 'unavailable' && sig.reason && (
            <div className="card">
              <h2 className="font-display text-lg font-semibold mb-2 text-forge-hot">Is the winner actually better?</h2>
              <p className="text-sm text-forge-steel/70">Significance testing unavailable — {sig.reason}</p>
            </div>
          )}

          {(result.pareto_frontier as ParetoRow[])?.length > 0 && (
            <div className="card">
              <h2 className="font-display text-lg font-semibold mb-4 text-forge-hot">Pareto frontier · accuracy vs latency</h2>
              <table className="w-full text-sm">
                <thead>
                  <tr className="kicker text-forge-steel/70 border-b border-forge-line">
                    <th className="text-left py-2 font-normal">Model</th>
                    <th className="text-right py-2 font-normal">CV Score</th>
                    <th className="text-right py-2 font-normal">Latency (ms)</th>
                    <th className="text-right py-2 font-normal">Tied</th>
                    <th className="text-right py-2 font-normal">Pareto</th>
                  </tr>
                </thead>
                <tbody>
                  {(result.pareto_frontier as ParetoRow[]).map((p) => (
                    <tr key={p.model_name} className="border-b border-forge-line/60">
                      <td className="py-2.5">{p.model_name}</td>
                      <td className="text-right font-mono text-forge-hot">{p.cv_score.toFixed(4)}</td>
                      <td className="text-right font-mono text-forge-steel">{p.latency_ms.toFixed(1)}</td>
                      <td className="text-right">{p.tied_with_leader ? <span className="text-forge-accent">=</span> : ''}</td>
                      <td className="text-right">{p.is_pareto_optimal ? <span className="text-forge-accent">★</span> : ''}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {(result.error_analysis as { worst_predictions?: unknown[] })?.worst_predictions && (
            <div className="card">
              <h2 className="font-display text-lg font-semibold mb-2 text-forge-hot">Error analysis</h2>
              <p className="text-sm text-forge-steel">
                {(result.error_analysis as { worst_predictions: unknown[] }).worst_predictions.length} worst predictions analyzed
              </p>
            </div>
          )}

          {id && (
            <div className="card">
              <h2 className="font-display text-lg font-semibold mb-3 text-forge-hot">Analysis report</h2>
              <a href={`${reportUrl(id)}?format=analysis`} target="_blank" rel="noreferrer" className="text-forge-accent hover:text-forge-amber transition-colors font-mono text-sm">
                Open LLM analysis report →
              </a>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function RunningView({ exp }: { exp: Experiment }) {
  const [now, setNow] = useState(Date.now());
  const feedRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);
  useEffect(() => {
    if (feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight;
  }, [exp.progress]);

  const started = Date.parse(exp.created_at) || now;
  const elapsed = Math.max(0, Math.floor((now - started) / 1000));
  const log = exp.progress_log || [];

  let current = PIPELINE.findIndex((s) => s.key === exp.stage);
  if (exp.stage === 'done') current = PIPELINE.length;
  if (current < 0) current = 0;

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-7">
        <div className="flex items-center gap-3">
          <span className="h-2.5 w-2.5 rounded-full bg-forge-accent animate-pulse shadow-[0_0_12px_2px_rgba(255,90,30,0.8)]" />
          <span className="kicker text-forge-amber/90">forging · in progress</span>
        </div>
        <span className="font-mono text-sm text-forge-steel">elapsed {fmtElapsed(elapsed)}</span>
      </div>

      <div className="relative">
        {PIPELINE.map((s, i) => {
          const done = i < current;
          const active = i === current;
          return (
            <div key={s.key} className="relative flex gap-4 pb-7 last:pb-0">
              {i < PIPELINE.length - 1 && (
                <span
                  className="absolute left-[11px] top-7 bottom-0 w-px"
                  style={{ background: done ? 'var(--ember)' : 'var(--line)' }}
                />
              )}
              <span
                className={`relative z-10 grid place-items-center h-6 w-6 rounded-full border text-[10px] font-mono shrink-0 ${
                  done
                    ? 'border-transparent text-forge-bg'
                    : active
                    ? 'border-forge-accent text-forge-hot'
                    : 'border-forge-line text-forge-steel/50'
                }`}
                style={
                  done
                    ? { background: 'var(--ember-grad)' }
                    : active
                    ? { boxShadow: '0 0 14px 2px rgba(255,90,30,0.45)' }
                    : {}
                }
              >
                {done ? '✓' : active ? <span className="h-2 w-2 rounded-full bg-forge-accent animate-pulse" /> : String(i + 1).padStart(2, '0')}
              </span>
              <div className="min-w-0 -mt-0.5">
                <div className={`font-display font-semibold ${active ? 'text-forge-hot' : done ? 'text-forge-steel' : 'text-forge-steel/50'}`}>
                  {s.label}
                </div>
                <div className="font-mono text-[0.72rem] text-forge-steel/55">{s.desc}</div>
                {active && (
                  <div className="font-mono text-xs text-forge-amber mt-1.5">{exp.progress || 'working…'}</div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {log.length > 0 && (
        <div ref={feedRef} className="mt-5 rounded-lg border border-forge-line bg-[#0b0a0d] p-3 max-h-40 overflow-y-auto">
          {log.map((line, i) => (
            <div
              key={i}
              className={`font-mono text-[0.72rem] py-0.5 ${i === log.length - 1 ? 'text-forge-hot' : 'text-forge-steel/65'}`}
            >
              <span className="text-forge-accent/60">›</span> {line}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ConfusionMatrix({ matrix, labels }: { matrix: number[][]; labels?: string[] }) {
  const n = matrix.length;
  const names = labels && labels.length === n ? labels : matrix.map((_, i) => `class ${i}`);
  const max = Math.max(1, ...matrix.flat());
  return (
    <div className="overflow-x-auto">
      <table className="text-sm font-mono border-separate border-spacing-1">
        <thead>
          <tr>
            <td className="p-2 text-forge-steel/50 text-[0.65rem]">true ↓ / pred →</td>
            {names.map((nm) => (
              <th key={nm} className="p-2 text-forge-steel font-normal text-xs">{nm}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <th className="p-2 text-left text-forge-steel font-normal text-xs">{names[i]}</th>
              {row.map((v, j) => (
                <td
                  key={j}
                  className="p-2 text-center rounded min-w-[3rem]"
                  style={{ background: `rgba(255,90,30,${(v / max) * 0.5})`, color: i === j ? 'var(--hot)' : 'var(--ink)' }}
                >
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
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

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="card">
      <p className="kicker text-forge-steel/80">{label}</p>
      <p className="font-display text-2xl font-bold mt-2 text-forge-hot">{value}</p>
    </div>
  );
}

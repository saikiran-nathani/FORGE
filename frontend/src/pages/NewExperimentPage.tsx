import { FormEvent, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { createExperiment } from '../services/api';

const NEXT = [
  ['01', 'Profile', 'columns, leakage, metric'],
  ['02', 'Engineer', 'features + selection'],
  ['03', 'Train', '16 models + HPO'],
  ['04', 'Evaluate', 'SHAP · fairness'],
  ['05', 'Deploy', 'live API'],
];

export default function NewExperimentPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [target, setTarget] = useState('');
  const [task, setTask] = useState('');
  const [trials, setTrials] = useState(10);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!file || !target) return;
    setLoading(true);
    setError('');
    try {
      const exp = await createExperiment(file, name || file.name, target, task, trials);
      navigate(`/experiments/${exp.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create experiment');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="grid lg:grid-cols-[1.4fr_0.6fr] gap-10 max-w-5xl mx-auto">
      <div>
        <p className="kicker">The anvil</p>
        <h1 className="font-display text-4xl font-bold mt-3">Forge a model</h1>
        <p className="text-forge-steel mt-2.5 max-w-lg">
          Drop in a dataset and tell FORGE what to predict. It takes over from there.
        </p>

        <form onSubmit={handleSubmit} className="card mt-8 space-y-6">
          {/* Dropzone */}
          <div>
            <label className="kicker mb-2 block text-forge-steel">Dataset · CSV / Parquet / JSON</label>
            <label
              htmlFor="dataset"
              className={`flex flex-col items-center justify-center text-center rounded-xl border border-dashed cursor-pointer px-6 py-10 transition-colors ${
                file ? 'border-forge-accent/60 bg-forge-accent/[0.04]' : 'border-forge-line hover:border-forge-accent/50'
              }`}
            >
              <span className="h-11 w-11 rounded-xl border border-forge-accent/30 flex items-center justify-center mb-3">
                <span className="h-3 w-3 rounded-sm bg-forge-accent shadow-[0_0_12px_2px_rgba(255,90,30,0.7)]" />
              </span>
              {file ? (
                <span className="font-mono text-sm text-forge-hot break-all">{file.name}</span>
              ) : (
                <>
                  <span className="text-forge-hot font-medium">Click to choose a file</span>
                  <span className="font-mono text-xs text-forge-steel/70 mt-1">or drag it onto the anvil</span>
                </>
              )}
              <input
                id="dataset"
                type="file"
                accept=".csv,.parquet,.json"
                className="hidden"
                onChange={(e) => setFile(e.target.files?.[0] || null)}
                required
              />
            </label>
          </div>

          <div className="grid sm:grid-cols-2 gap-4">
            <div>
              <label className="kicker mb-2 block text-forge-steel">Experiment name</label>
              <input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="optional" />
            </div>
            <div>
              <label className="kicker mb-2 block text-forge-steel">Target column *</label>
              <input className="input" value={target} onChange={(e) => setTarget(e.target.value)} required placeholder="e.g. churned" />
            </div>
          </div>

          <div>
            <label className="kicker mb-2 block text-forge-steel">What are you predicting?</label>
            <textarea
              className="input min-h-[90px] resize-y"
              value={task}
              onChange={(e) => setTask(e.target.value)}
              placeholder="Predict which customers will churn based on usage and billing…"
            />
            <p className="text-xs text-forge-steel/60 mt-1.5">
              Guides the LLM-powered analysis (semantic profiling &amp; feature ideas) when an
              API key is configured. Without a key, the pipeline runs its deterministic
              heuristics and this is used only for labeling.
            </p>
          </div>

          <div>
            <label className="kicker mb-2 block text-forge-steel">HPO trials per model · {trials}</label>
            <input
              type="range"
              min={5}
              max={50}
              value={trials}
              onChange={(e) => setTrials(Number(e.target.value))}
              className="w-full accent-[#ff5a1e]"
            />
            <div className="flex justify-between font-mono text-[0.65rem] text-forge-steel/60 mt-1">
              <span>faster</span>
              <span>more thorough</span>
            </div>
          </div>

          {error && (
            <p className="font-mono text-sm text-red-400 border border-red-500/30 bg-red-500/5 rounded-lg px-3 py-2">{error}</p>
          )}

          <button type="submit" className="btn-primary w-full text-base py-3" disabled={loading || !file || !target}>
            {loading ? 'Igniting the forge…' : 'Start the forge →'}
          </button>
        </form>
      </div>

      {/* What happens next */}
      <aside className="lg:pt-16">
        <div className="card sticky top-24">
          <p className="kicker text-forge-amber/90">What happens next</p>
          <ol className="mt-5 space-y-4">
            {NEXT.map(([n, t, d]) => (
              <li key={n} className="flex gap-3.5 items-start">
                <span className="font-mono text-xs text-forge-accent pt-0.5">{n}</span>
                <div>
                  <div className="font-display font-semibold text-forge-hot text-sm">{t}</div>
                  <div className="font-mono text-[0.7rem] text-forge-steel/70">{d}</div>
                </div>
              </li>
            ))}
          </ol>
          <div className="mt-6 pt-5 border-t border-forge-line">
            <Link to="/experiments/demo" className="font-mono text-xs text-forge-steel hover:text-forge-hot transition-colors">
              ↳ or explore the live demo first
            </Link>
          </div>
        </div>
      </aside>
    </div>
  );
}

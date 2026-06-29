import { FormEvent, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { createExperiment } from '../services/api';

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
      const exp = await createExperiment(
        file,
        name || file.name,
        target,
        task,
        trials,
      );
      navigate(`/experiments/${exp.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create experiment');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="max-w-xl mx-auto">
      <h1 className="text-3xl font-bold mb-2">New Experiment</h1>
      <p className="text-forge-muted mb-8">Upload a dataset and describe your prediction goal</p>

      <form onSubmit={handleSubmit} className="card space-y-5">
        <div>
          <label className="block text-sm text-forge-muted mb-1">Dataset (CSV)</label>
          <input
            type="file"
            accept=".csv,.parquet,.json"
            className="input"
            onChange={(e) => setFile(e.target.files?.[0] || null)}
            required
          />
        </div>
        <div>
          <label className="block text-sm text-forge-muted mb-1">Experiment Name</label>
          <input className="input" value={name} onChange={(e) => setName(e.target.value)} placeholder="Optional" />
        </div>
        <div>
          <label className="block text-sm text-forge-muted mb-1">Target Column *</label>
          <input className="input" value={target} onChange={(e) => setTarget(e.target.value)} required placeholder="e.g. Survived" />
        </div>
        <div>
          <label className="block text-sm text-forge-muted mb-1">Task Description</label>
          <textarea
            className="input min-h-[80px]"
            value={task}
            onChange={(e) => setTask(e.target.value)}
            placeholder="Predict customer churn based on usage patterns..."
          />
        </div>
        <div>
          <label className="block text-sm text-forge-muted mb-1">HPO Trials per Model</label>
          <input
            type="number"
            className="input"
            min={1}
            max={100}
            value={trials}
            onChange={(e) => setTrials(Number(e.target.value))}
          />
        </div>
        {error && <p className="text-red-400 text-sm">{error}</p>}
        <button type="submit" className="btn-primary w-full" disabled={loading}>
          {loading ? 'Starting...' : 'Start Pipeline'}
        </button>
      </form>
    </div>
  );
}

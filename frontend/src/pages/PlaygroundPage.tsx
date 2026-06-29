import { FormEvent, useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { getModelInfo, predict } from '../services/api';

export default function PlaygroundPage() {
  const { id } = useParams<{ id: string }>();
  const [inputColumns, setInputColumns] = useState<string[]>([]);
  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!id) return;
    getModelInfo(id)
      .then((info) => {
        const cols = (info.input_columns as string[]) || [];
        setInputColumns(cols);
      })
      .catch((err) => setError(err instanceof Error ? err.message : 'Failed to load model schema'));
  }, [id]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!id) return;
    setLoading(true);
    setError('');
    try {
      const body: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(inputs)) {
        body[k] = Number.isNaN(Number(v)) ? v : Number(v);
      }
      const res = await predict(id, body);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <Link to={`/experiments/${id}/deploy`} className="text-sm text-forge-muted hover:text-white">← Back to deployment</Link>
      <h1 className="text-3xl font-bold">Prediction Playground</h1>

      {inputColumns.length === 0 && !error && (
        <p className="text-forge-muted text-sm">Loading input schema…</p>
      )}

      <form onSubmit={handleSubmit} className="card space-y-4 max-w-lg">
        {inputColumns.map((col) => (
          <div key={col}>
            <label className="block text-sm text-forge-muted mb-1">{col}</label>
            <input
              className="input"
              value={inputs[col] || ''}
              onChange={(e) => setInputs({ ...inputs, [col]: e.target.value })}
              required
            />
          </div>
        ))}
        {error && <p className="text-red-400 text-sm">{error}</p>}
        <button type="submit" className="btn-primary w-full" disabled={loading || inputColumns.length === 0}>
          {loading ? 'Predicting...' : 'Predict'}
        </button>
      </form>

      {result && (
        <div className="card">
          <h2 className="text-lg font-semibold mb-3">Result</h2>
          <pre className="text-sm font-mono bg-slate-800 p-4 rounded-lg overflow-auto">
            {JSON.stringify(result, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

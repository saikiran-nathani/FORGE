export interface Experiment {
  id: string;
  name: string;
  target_column: string;
  task_description: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  created_at: string;
  progress: string;
  error: string;
  result: Record<string, unknown>;
}

// Backend base URL. Empty in local dev -> requests go to '/api/v1', which the
// Vite dev proxy (and the Docker nginx proxy) forward to the FastAPI server.
// On Netlify, set VITE_API_URL to the deployed backend, e.g. https://forge-api.onrender.com
const BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '');
const API = `${BASE}/api/v1`;

export async function createExperiment(
  file: File,
  name: string,
  targetColumn: string,
  taskDescription: string,
  trials: number,
): Promise<Experiment> {
  const form = new FormData();
  form.append('file', file);
  form.append('name', name);
  form.append('target_column', targetColumn);
  form.append('task_description', taskDescription);
  form.append('trials', String(trials));
  form.append('fast_mode', 'true');

  const res = await fetch(`${API}/experiments`, { method: 'POST', body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function listExperiments(): Promise<Experiment[]> {
  const res = await fetch(`${API}/experiments`);
  if (!res.ok) throw new Error('Failed to fetch experiments');
  return res.json();
}

export async function getExperiment(id: string): Promise<Experiment> {
  const res = await fetch(`${API}/experiments/${id}`);
  if (!res.ok) throw new Error('Experiment not found');
  return res.json();
}

export async function deployExperiment(id: string): Promise<Record<string, unknown>> {
  const res = await fetch(`${API}/experiments/${id}/deploy`, { method: 'POST' });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function predict(id: string, features: Record<string, unknown>): Promise<Record<string, unknown>> {
  const res = await fetch(`${API}/experiments/${id}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(features),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function reportUrl(id: string): string {
  return `${API}/experiments/${id}/report`;
}

export async function getModelInfo(id: string): Promise<Record<string, unknown>> {
  const res = await fetch(`${API}/experiments/${id}/model-info`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getMonitoring(id: string): Promise<Record<string, unknown>> {
  const res = await fetch(`${API}/experiments/${id}/monitoring`);
  if (!res.ok) throw new Error('Failed to fetch monitoring');
  return res.json();
}

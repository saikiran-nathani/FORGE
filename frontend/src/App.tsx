import { Link, Route, Routes } from 'react-router-dom';
import DeployPage from './pages/DeployPage';
import ExperimentPage from './pages/ExperimentPage';
import HomePage from './pages/HomePage';
import MonitoringPage from './pages/MonitoringPage';
import NewExperimentPage from './pages/NewExperimentPage';
import PlaygroundPage from './pages/PlaygroundPage';

export default function App() {
  return (
    <div className="min-h-screen">
      <nav className="border-b border-slate-700/50 bg-slate-900/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-6xl mx-auto px-4 py-3 flex items-center gap-6">
          <Link to="/" className="text-xl font-bold text-forge-accent tracking-tight">
            FORGE
          </Link>
          <Link to="/" className="text-forge-muted hover:text-white text-sm">
            Experiments
          </Link>
          <Link to="/new" className="text-forge-muted hover:text-white text-sm">
            New Experiment
          </Link>
        </div>
      </nav>
      <main className="max-w-6xl mx-auto px-4 py-8">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/new" element={<NewExperimentPage />} />
          <Route path="/experiments/:id" element={<ExperimentPage />} />
          <Route path="/experiments/:id/deploy" element={<DeployPage />} />
          <Route path="/experiments/:id/playground" element={<PlaygroundPage />} />
          <Route path="/experiments/:id/monitoring" element={<MonitoringPage />} />
        </Routes>
      </main>
    </div>
  );
}

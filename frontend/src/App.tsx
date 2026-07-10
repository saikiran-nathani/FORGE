import { NavLink, Route, Routes, Link } from 'react-router-dom';
import { ReactNode } from 'react';
import DeployPage from './pages/DeployPage';
import ExperimentPage from './pages/ExperimentPage';
import HomePage from './pages/HomePage';
import LandingPage from './pages/LandingPage';
import MonitoringPage from './pages/MonitoringPage';
import NewExperimentPage from './pages/NewExperimentPage';
import PlaygroundPage from './pages/PlaygroundPage';

function Contained({ children }: { children: ReactNode }) {
  return <div className="max-w-6xl mx-auto px-5 sm:px-8 py-12">{children}</div>;
}

function Nav() {
  const link = ({ isActive }: { isActive: boolean }) =>
    `text-sm font-mono tracking-wide transition-colors ${
      isActive ? 'text-forge-hot' : 'text-forge-steel hover:text-forge-hot'
    }`;
  return (
    <nav className="sticky top-0 z-40 backdrop-blur-md bg-[#0a0a0b]/70 border-b border-forge-line">
      <div className="max-w-6xl mx-auto px-5 sm:px-8 h-16 flex items-center gap-8">
        <Link to="/" className="flex items-center gap-2.5 group">
          <span className="relative h-2.5 w-2.5 rounded-full bg-forge-accent shadow-[0_0_12px_2px_rgba(255,90,30,0.8)]" />
          <span className="font-display font-bold text-xl tracking-[0.18em] gradient-text">FORGE</span>
        </Link>
        <div className="hidden sm:flex items-center gap-7 ml-2">
          <NavLink to="/" className={link} end>foundry</NavLink>
          <NavLink to="/experiments" className={link}>experiments</NavLink>
        </div>
        <div className="ml-auto">
          <Link to="/new" className="btn-primary text-sm py-2 px-4">Forge a model</Link>
        </div>
      </div>
    </nav>
  );
}

function Footer() {
  return (
    <footer className="border-t border-forge-line mt-24">
      <div className="max-w-6xl mx-auto px-5 sm:px-8 py-10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div className="flex items-center gap-2.5">
          <span className="h-2 w-2 rounded-full bg-forge-accent shadow-[0_0_10px_1px_rgba(255,90,30,0.8)]" />
          <span className="font-display font-semibold tracking-[0.18em] text-forge-steel">FORGE</span>
        </div>
        <p className="font-mono text-xs text-forge-steel/70">
          autonomous ML foundry · profiled · trained · explained · shipped
        </p>
      </div>
    </footer>
  );
}

export default function App() {
  return (
    <div className="atmos flex flex-col min-h-screen">
      <div className="grain" aria-hidden />
      <Nav />
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/experiments" element={<Contained><HomePage /></Contained>} />
          <Route path="/new" element={<Contained><NewExperimentPage /></Contained>} />
          <Route path="/experiments/:id" element={<Contained><ExperimentPage /></Contained>} />
          <Route path="/experiments/:id/deploy" element={<Contained><DeployPage /></Contained>} />
          <Route path="/experiments/:id/playground" element={<Contained><PlaygroundPage /></Contained>} />
          <Route path="/experiments/:id/monitoring" element={<Contained><MonitoringPage /></Contained>} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}

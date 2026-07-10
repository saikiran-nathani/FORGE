import { useMemo } from 'react';
import { Link } from 'react-router-dom';
import { useReveal } from '../hooks/useReveal';

const STAGES = [
  { n: '01', code: 'PROFILE', title: 'Profile', desc: 'Statistical + LLM semantic read of every column — types, leakage, quality score, and the right metric for the job.' },
  { n: '02', code: 'ENGINEER', title: 'Engineer', desc: 'The model proposes new features; a validation layer runs them safely, then selection keeps only what earns its place.' },
  { n: '03', code: 'TRAIN', title: 'Train & tune', desc: '16 architectures — linear, trees, boosting, neural — tuned with Optuna Bayesian HPO, then fused into ensembles.' },
  { n: '04', code: 'EVALUATE', title: 'Evaluate', desc: 'SHAP, LIME, partial-dependence, calibration, error analysis and a fairness audit — not just an accuracy number.' },
  { n: '05', code: 'DEPLOY', title: 'Deploy', desc: 'The Pareto-best model ships as a FastAPI endpoint with a model card, drift monitoring, and a live Playground.' },
];

const CAPS = [
  { t: '16 model architectures', d: 'Logistic, Ridge/Lasso, RF, Extra Trees, XGBoost, LightGBM, CatBoost, SVM, KNN, Naive Bayes — plus MLP & TabTransformer.' },
  { t: 'Pareto model selection', d: 'Chooses on accuracy vs. latency, not blind best-score. Production-aware by default.' },
  { t: 'Explainability built in', d: 'SHAP beeswarm, LIME local reasons, partial-dependence, and calibration (Brier, ECE).' },
  { t: 'LLM feature engineering', d: 'Model-written pandas behind an AST validation layer, with heuristic fallback when offline.' },
  { t: 'Fairness & error analysis', d: 'Subgroup metrics and the worst-case predictions surfaced automatically.' },
  { t: 'One-click deployment', d: 'Serialized bundle becomes a live API, model card, and drift monitor — instantly.' },
];

const STATS = ['16 ARCHITECTURES', 'BAYESIAN HPO', 'SHAP · LIME · PDP', '1-CLICK DEPLOY'];

function EmberField() {
  const embers = useMemo(
    () =>
      Array.from({ length: 18 }, (_, i) => ({
        left: `${(i * 53 + 7) % 100}%`,
        duration: `${7 + ((i * 7) % 9)}s`,
        delay: `${(i * 1.3) % 9}s`,
        scale: 0.6 + ((i * 13) % 10) / 10,
      })),
    [],
  );
  return (
    <div className="ember-field" aria-hidden>
      {embers.map((e, i) => (
        <span
          key={i}
          className="ember"
          style={{ left: e.left, animationDuration: e.duration, animationDelay: e.delay, transform: `scale(${e.scale})` }}
        />
      ))}
    </div>
  );
}

function ForgeCore() {
  return (
    <div className="relative mx-auto aspect-square w-[300px] sm:w-[380px] lg:w-[440px]" aria-hidden>
      {/* rotating heat ring */}
      <div
        className="absolute inset-0 rounded-full blur-[2px] opacity-70"
        style={{
          background: 'conic-gradient(from 0deg, transparent, rgba(255,90,30,0.0), rgba(255,122,31,0.55), rgba(255,230,194,0.9), rgba(255,122,31,0.55), transparent)',
          animation: 'spin-slow 14s linear infinite',
          WebkitMaskImage: 'radial-gradient(closest-side, transparent 64%, #000 66%, #000 76%, transparent 80%)',
          maskImage: 'radial-gradient(closest-side, transparent 64%, #000 66%, #000 76%, transparent 80%)',
        }}
      />
      {/* concentric steel rings */}
      <div className="absolute inset-[6%] rounded-full border border-forge-line" />
      <div className="absolute inset-[20%] rounded-full border border-forge-line/70" />
      <div className="absolute inset-[34%] rounded-full border border-forge-accent/25" />
      {/* molten core */}
      <div
        className="absolute inset-[40%] rounded-full"
        style={{
          background: 'radial-gradient(circle at 38% 32%, #ffe6c2, #ff7a1f 45%, #ff4d00 78%)',
          boxShadow: '0 0 60px 12px rgba(255,90,30,0.55), inset 0 0 30px rgba(255,230,194,0.5)',
          animation: 'floaty 6s ease-in-out infinite',
        }}
      />
      {/* orbiting tick labels */}
      <div className="absolute inset-0 font-mono text-[10px] tracking-widest text-forge-steel/70">
        <span className="absolute top-1 left-1/2 -translate-x-1/2">RAW · IN</span>
        <span className="absolute bottom-1 left-1/2 -translate-x-1/2 text-forge-amber/80">MODEL · OUT</span>
      </div>
    </div>
  );
}

export default function LandingPage() {
  const ref = useReveal<HTMLDivElement>();

  return (
    <div ref={ref}>
      {/* ============================ HERO ============================ */}
      <section className="relative overflow-hidden">
        <EmberField />
        <div className="max-w-6xl mx-auto px-5 sm:px-8 pt-20 pb-24 grid lg:grid-cols-[1.15fr_0.85fr] gap-12 items-center relative z-10">
          <div>
            <p className="kicker reveal in">Autonomous ML Foundry</p>
            <h1 className="reveal in font-display font-bold leading-[0.98] tracking-tight mt-5 text-[3.1rem] sm:text-[4.4rem]" style={{ transitionDelay: '0.08s' }}>
              Forge a <span className="gradient-text">deployed model</span> from raw data.
            </h1>
            <p className="reveal in mt-7 max-w-xl text-[1.06rem] leading-relaxed text-forge-steel" style={{ transitionDelay: '0.16s' }}>
              Upload a dataset and describe your goal. FORGE profiles it, engineers and selects features,
              trains and tunes <span className="text-forge-hot">16 model architectures</span>, explains the
              winner with SHAP, and ships it as a <span className="text-forge-hot">live prediction API</span> —
              end to end, on its own.
            </p>
            <div className="reveal in mt-9 flex flex-wrap items-center gap-3.5" style={{ transitionDelay: '0.24s' }}>
              <Link to="/new" className="btn-primary text-base px-6 py-3">Forge a model →</Link>
              <Link to="/experiments/demo" className="btn-ghost text-base px-6 py-3">Explore the live demo</Link>
            </div>
            <div className="reveal in mt-10 flex flex-wrap gap-x-5 gap-y-2 font-mono text-[0.72rem] tracking-widest text-forge-steel/80" style={{ transitionDelay: '0.32s' }}>
              {STATS.map((s, i) => (
                <span key={s} className="flex items-center gap-5">
                  {i > 0 && <span className="text-forge-accent/50">·</span>}
                  {s}
                </span>
              ))}
            </div>
          </div>
          <div className="reveal in" style={{ transitionDelay: '0.2s', animation: 'floaty 7s ease-in-out infinite' }}>
            <ForgeCore />
          </div>
        </div>
        <div className="h-px flowline mx-5 sm:mx-8" />
      </section>

      {/* ======================= WHAT IS FORGE ======================= */}
      <section className="max-w-6xl mx-auto px-5 sm:px-8 py-24">
        <p className="kicker reveal">The idea</p>
        <h2 className="reveal font-display text-3xl sm:text-[2.6rem] font-semibold leading-tight mt-4 max-w-4xl" style={{ transitionDelay: '0.05s' }}>
          Most AutoML stops at a leaderboard. <span className="text-forge-steel">FORGE runs the whole foundry</span> — from raw data to a deployed, explainable model.
        </h2>
        <div className="grid sm:grid-cols-3 gap-5 mt-12">
          {[
            { k: 'Describe, don’t configure', v: 'Say what you want to predict in plain language. FORGE infers the task, the metric, and the plan.' },
            { k: 'Decisions, not dials', v: 'It picks features, models, and hyperparameters — and shows you why each one won.' },
            { k: 'Ends at a live API', v: 'The output isn’t a notebook. It’s a deployed endpoint you can call right now.' },
          ].map((c, i) => (
            <div key={c.k} className="reveal card" style={{ transitionDelay: `${0.08 * i}s` }}>
              <h3 className="font-display font-semibold text-forge-hot text-lg">{c.k}</h3>
              <p className="text-forge-steel mt-2.5 text-[0.95rem] leading-relaxed">{c.v}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ========================= PIPELINE ========================== */}
      <section className="max-w-6xl mx-auto px-5 sm:px-8 py-12">
        <div className="flex items-end justify-between flex-wrap gap-4">
          <div>
            <p className="kicker reveal">The forge line</p>
            <h2 className="reveal font-display text-3xl sm:text-[2.6rem] font-semibold mt-4" style={{ transitionDelay: '0.05s' }}>
              Five stations. One pass.
            </h2>
          </div>
          <p className="reveal font-mono text-xs text-forge-steel/70 max-w-xs" style={{ transitionDelay: '0.1s' }}>
            // raw data enters at 01 and leaves 05 as a deployed model
          </p>
        </div>

        <div className="relative mt-12">
          <div className="hidden lg:block absolute left-0 right-0 top-[34px] h-px flowline" />
          <div className="grid lg:grid-cols-5 gap-5">
            {STAGES.map((s, i) => (
              <div key={s.code} className="reveal card group" style={{ transitionDelay: `${0.1 * i}s` }}>
                <div className="flex items-center justify-between">
                  <span className="font-display text-2xl font-bold gradient-text">{s.n}</span>
                  <span className="h-2 w-2 rounded-full bg-forge-accent/40 group-hover:bg-forge-accent transition-colors shadow-[0_0_10px_1px_rgba(255,90,30,0.6)]" />
                </div>
                <p className="kicker mt-4 text-forge-amber/90">{s.code}</p>
                <h3 className="font-display font-semibold text-forge-hot text-lg mt-1">{s.title}</h3>
                <p className="text-forge-steel mt-2.5 text-[0.88rem] leading-relaxed">{s.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ====================== CAPABILITIES ========================= */}
      <section className="max-w-6xl mx-auto px-5 sm:px-8 py-24">
        <p className="kicker reveal">Under the hood</p>
        <h2 className="reveal font-display text-3xl sm:text-[2.6rem] font-semibold mt-4" style={{ transitionDelay: '0.05s' }}>
          Built like production, not a demo.
        </h2>
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-5 mt-12">
          {CAPS.map((c, i) => (
            <div key={c.t} className="reveal card" style={{ transitionDelay: `${0.06 * i}s` }}>
              <div className="h-9 w-9 rounded-lg border border-forge-accent/30 flex items-center justify-center mb-4">
                <span className="h-2.5 w-2.5 rounded-sm bg-forge-accent shadow-[0_0_10px_1px_rgba(255,90,30,0.7)]" />
              </div>
              <h3 className="font-display font-semibold text-forge-hot text-[1.05rem]">{c.t}</h3>
              <p className="text-forge-steel mt-2.5 text-[0.92rem] leading-relaxed">{c.d}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ========================= NUMBERS =========================== */}
      <section className="max-w-6xl mx-auto px-5 sm:px-8">
        <div className="reveal grid grid-cols-2 lg:grid-cols-4 gap-px bg-forge-line rounded-2xl overflow-hidden border border-forge-line">
          {[
            ['16', 'model architectures'],
            ['5', 'stage pipeline'],
            ['0.94', 'demo accuracy / 0.98 AUC'],
            ['<1ms', 'inference latency'],
          ].map(([n, l]) => (
            <div key={l} className="bg-forge-bg2 px-6 py-9 text-center">
              <div className="font-display text-4xl sm:text-5xl font-bold gradient-text">{n}</div>
              <div className="font-mono text-[0.68rem] tracking-widest text-forge-steel/80 mt-3 uppercase">{l}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ========================== CTA ============================== */}
      <section className="max-w-6xl mx-auto px-5 sm:px-8 py-28">
        <div className="reveal relative overflow-hidden rounded-2xl border border-forge-line px-8 sm:px-16 py-20 text-center">
          <div
            className="absolute inset-0 -z-10"
            style={{ background: 'radial-gradient(60% 120% at 50% 120%, rgba(255,90,30,0.18), transparent 60%)' }}
          />
          <p className="kicker">Step up to the anvil</p>
          <h2 className="font-display text-4xl sm:text-[3.2rem] font-bold leading-tight mt-5">
            Bring a CSV and a goal.<br />Get back a model you can <span className="gradient-text">ship</span>.
          </h2>
          <div className="mt-10 flex flex-wrap justify-center gap-3.5">
            <Link to="/new" className="btn-primary text-base px-7 py-3.5">Forge a model →</Link>
            <Link to="/experiments/demo" className="btn-ghost text-base px-7 py-3.5">See the live demo first</Link>
          </div>
        </div>
      </section>
    </div>
  );
}

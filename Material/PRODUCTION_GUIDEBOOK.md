# The Production Guidebook
### How real teams take an app from Day-1 POC to a system that survives strangers at 3 a.m.

> Most tutorials teach you to *build* software. This teaches you to **operate** it — the
> disciplines, roles, gates, and safety nets that separate "it runs on my machine" from
> "it runs reliably for a million people while nobody is watching." Stack-agnostic; examples
> use Python/FastAPI/React where concrete helps.

---

## 0. The mental model (read this first)

**POC ≠ MVP ≠ Production.** These are different products with different bars:

| Stage | Question it answers | Bar | Who cares |
|---|---|---|---|
| **POC / Prototype** | "Is this *possible*?" | Works once, on your laptop, happy path | You |
| **MVP** | "Do people *want* it?" | Works for a few real users, core path solid | Early users |
| **Production / GA** | "Does it work for *everyone, always, safely*?" | Reliable, secure, observable, recoverable, compliant | The business, on-call, legal, customers |

The jump most people never make is **MVP → Production**. That jump is not more features — it's
**everything that happens when things go wrong**: the server crashes, the disk fills, a hacker
probes you, a deploy breaks, a database migration corrupts data, traffic 100×'s, a dependency
ships a vulnerability, an engineer fat-fingers a command. Production engineering is the discipline
of *assuming failure* and engineering for it.

**The three invisible truths of production software:**
1. **Code is ~20% of the work.** The other 80% is tests, infra, security, observability, docs, process, and operations.
2. **You optimize for the person debugging at 3 a.m.**, who is tired, under pressure, and might be you.
3. **Everything is a trade-off against time, money, and risk.** Maturity is knowing which corners are safe to cut *for this product* — and writing down why.

---

## 1. The team — generalized roles

Real companies have dozens of specialized titles. They collapse into **~11 generalized roles**.
In a big org each is a team; in a startup one person wears five hats; solo, *you are all of them*
and must consciously "switch hats." The value isn't the titles — it's realizing each role is a
**distinct set of concerns that must be owned by someone**, or it silently doesn't happen.

| # | Generalized role | Owns (the core question) | Real-world titles that map here | Key deliverables |
|---|---|---|---|---|
| 1 | **Product** | *What* are we building and *why*? For whom? | PM, Product Owner, Business Analyst | Roadmap, specs, acceptance criteria, prioritization |
| 2 | **Design / UX** | Is it usable, accessible, coherent? | UX/UI designer, researcher, design systems | Wireframes, prototypes, design system, a11y specs |
| 3 | **Architecture** | How do the pieces fit? What are the -ilities? | Tech Lead, Staff/Principal Eng, Architect | System design, ADRs, tech standards, API contracts |
| 4 | **Application Engineering** | Building the features (frontend + backend) | SWE, Frontend/Backend/Full-stack | Feature code, unit/integration tests, PRs |
| 5 | **Quality Engineering** | How do we *know* it works and stays working? | QA, SDET, Test Engineer | Test strategy, E2E/automation suites, quality gates |
| 6 | **Platform / DevOps / SRE** | How does it run, scale, and stay up? | DevOps, SRE, Platform, Infra | CI/CD, IaC, environments, SLOs, on-call, runbooks |
| 7 | **Security (AppSec)** | How could this be abused, and how do we prevent it? | AppSec, Security Eng, GRC | Threat models, security reviews, pen tests, policies |
| 8 | **Data** | How is data modeled, stored, moved, and governed? | Data Eng, DBA, Data/ML Eng | Schemas, pipelines, migrations, backups, governance |
| 9 | **Release / Change Mgmt** | How do changes reach users safely? | Release Manager, Build Eng | Versioning, release process, rollback plans |
| 10 | **Engineering Management** | People, delivery, unblocking, priorities | EM, Delivery Lead, Scrum Master | Planning, staffing, process, removing blockers |
| 11 | **Docs / DevEx / Support** | Can others understand and use this? | Tech Writer, DevEx, Support/CS | Docs, onboarding, API references, support runbooks |

**How they interact (the "shape" of collaboration):** Product defines *what* → Design shapes *how it feels*
→ Architecture decides *how it's built* → Engineering builds it → Quality proves it → Platform runs it
→ Security guards it → Data persists it → Release ships it → Management orchestrates it → Docs/Support
sustains it. Crucially this is **not a waterfall** — in modern teams these run **concurrently and
continuously**, with feedback loops (see §2 and §3.4).

> **Solo-founder translation:** you can't do all 11 well at once. The trick is to *timebox each hat*:
> "Right now I'm the Security engineer reviewing this endpoint," then explicitly switch. Most solo
> disasters come from *never putting on* the Security, Quality, or SRE hat at all.

---

## 2. The lifecycle — from Day 1 to steady-state operation

Software has a lifecycle with **gates** between phases. A gate is an explicit checklist that must
pass before you advance — this is what stops half-baked work from reaching users.

### Phase 0 — Discovery & framing
- **Problem statement**: the user, the pain, the "why now." (Product)
- **Success metrics**: how you'll *measure* it worked (activation, retention, latency, revenue). Define these *before* building.
- **Constraints**: budget, deadline, compliance, team skills.
- **Risks & assumptions**: what could kill this; what you're betting on.
- **Gate → "Definition of Ready":** the problem is clear, scoped, and worth doing.

### Phase 1 — Design & architecture
- **UX**: user flows, wireframes, prototype, accessibility plan.
- **Architecture**: choose the stack *with reasons*, define components, data flow, and the **-ilities** you're targeting (scalability, availability, security, maintainability, cost).
- **ADRs (Architecture Decision Records)**: 1-page docs capturing *each significant decision and why* — the single highest-ROI habit most solo devs skip. Future-you will thank present-you.
- **Threat model** (first pass): "how could this be attacked/abused?" (see §3.5).
- **API/contract design**: define interfaces before implementing (OpenAPI, schemas). Enables parallel work.
- **Gate:** design reviewed; major risks have a plan.

### Phase 2 — POC / spike
- Prove the *riskiest* technical assumption fast and throwaway. Time-boxed (days).
- **Explicitly labeled disposable.** The #1 sin: a POC quietly becoming production. If it survives, *rewrite it deliberately.*

### Phase 3 — MVP build
- Build the thin **end-to-end** slice (one real user journey working front-to-back) before breadth.
- Set up the **engineering foundation now, not later** (this is the part people defer and regret): repo + branching, CI, linting/formatting/type-checking, test harness, secrets handling, environments, a logging baseline. Retrofitting these onto a mature codebase is 10× harder.
- **Gate:** core journey works for real users; foundation in place.

### Phase 4 — Hardening (MVP → Production) ← *the phase people skip*
This is the whole point of this guidebook. In parallel:
- **Testing** brought up to target coverage + integration/E2E (§3.2).
- **Security** review, dependency audit, secrets, authz, rate limiting (§3.5).
- **Observability** wired in: logs, metrics, traces, dashboards, alerts (§3.9).
- **Reliability** patterns: timeouts, retries, graceful degradation, backups, DR (§3.8).
- **Performance/load** testing to known limits (§3.8).
- **Infra**: real environments, IaC, deployment strategy, rollback (§3.4, §3.6).
- **Docs & runbooks** for operating it (§3.11).
- **Gate → "Production Readiness Review" (§4).**

### Phase 5 — Launch / GA
- Staged rollout (canary → % → 100%), feature-flagged, with a **rollback plan rehearsed**.
- On-call coverage arranged; dashboards watched; a "war room" for big launches.

### Phase 6 — Operate & evolve (forever)
- **Monitor** SLOs, spend error budget, respond to alerts, run **on-call** and **incident response**.
- **Blameless postmortems** after incidents → action items → fix the *system*, not the person.
- **Maintain**: patch dependencies, rotate secrets, prune data, control cost, pay down tech debt.
- **Iterate**: new features re-enter at Phase 1. **Deprecate** old things with a policy.
- **Sunset**: eventually, decommission gracefully (data export, comms, archival).

---

## 3. The disciplines (the deep end)

Each of these is a *specialty*. This is where "how real apps are made" actually lives.

### 3.1 Source control & collaboration
- **Everything in version control** — code, IaC, configs, docs, DB migrations. If it's not in git, it doesn't exist.
- **Branching strategy**: trunk-based (short-lived branches, merge daily) is the modern default; GitFlow for slower release trains. Protect `main` with **branch protection** (no direct pushes; PR + green CI + review required to merge).
- **Pull Requests**: small, focused, one concern each. A 2,000-line PR gets rubber-stamped; a 200-line PR gets a real review.
- **Code review** (see below) is mandatory — even solo, review your own PR the next morning.
- **Conventions**: `CODEOWNERS` (who must review what), Conventional Commits (`feat:`, `fix:` → auto-changelogs), PR templates (what/why/testing/screenshots).

**Code review — what reviewers actually look for** (not just "does it work"):
correctness & edge cases · security holes · tests included · readability/naming · does it fit the
architecture · error handling · performance foot-guns · backward compatibility · does the PR do
*one* thing. **Culture:** review the code, not the person; ask questions, don't demand; approve fast to keep flow.

### 3.2 Testing strategy (far more than "write some tests")
The **test pyramid** — many fast/cheap tests at the bottom, few slow/expensive at the top:

```
        /\        E2E / UI         (few, slow, brittle, high-confidence)
       /  \       Integration      (some — services, DB, real deps)
      /____\      Unit             (many, fast, isolated)  ← the base
```

**Types of tests, and what each catches:**
- **Unit** — one function/class in isolation (mock deps). Fast feedback on logic.
- **Integration** — components together (API + DB, service + queue). Catches wiring bugs.
- **Contract** — verifies services agree on an interface (Pact). Critical for microservices/teams.
- **End-to-end (E2E)** — a real user flow through the whole system (Playwright, Cypress).
- **Regression** — a test added for every bug fixed, so it never returns.
- **Load / stress / soak** — behavior under expected, extreme, and *sustained* traffic (k6, Locust).
- **Chaos** — deliberately break things in prod-like envs (kill nodes, add latency) to prove resilience (Chaos Monkey).
- **Mutation** — mutate your code and check tests fail; measures *test quality*, not just quantity (Stryker, mutmut).
- **Security** — SAST/DAST/dependency scans (§3.5).
- **Accessibility** — automated (axe) + manual (screen readers, keyboard-only).
- **Property-based** — generate thousands of random inputs against invariants (Hypothesis, fast-check).

**Coverage — and why it lies.** Coverage measures lines *executed*, not lines *verified*. 100%
coverage with no assertions tests nothing. Use it as a **floor and a trend** (e.g., "≥80% on new
code, never decreasing"), not a trophy. Better signals: **mutation score**, and "are the *risky*
paths tested?" **Flaky tests** (pass/fail randomly) are worse than no tests — they train people to
ignore red; quarantine and fix them. Manage **test data & fixtures** deliberately (factories, seeds,
disposable DBs). **Gate CI on tests** — a red build cannot merge.

### 3.3 Static analysis & quality gates (catch bugs before running)
- **Formatter** (Prettier, Black, gofmt) — ends style debates; auto-applied.
- **Linter** (ESLint, Ruff, golangci-lint) — bug patterns, smells, anti-patterns.
- **Type checking** (TypeScript, mypy/pyright) — a whole class of bugs eliminated at compile time.
- **Complexity/dead-code/duplication** checks; **pre-commit hooks** to run all of it *before* a commit.
- **These run in CI as gates** — merge is blocked if any fail.

### 3.4 CI/CD (Continuous Integration / Delivery / Deployment)
- **CI** = every push automatically builds + lints + type-checks + tests. Prevents "works on my machine" and integration hell. The pipeline **is the gate.**
- **CD** = automated path to environments. *Delivery* = auto to staging, one click to prod; *Deployment* = auto to prod once green.
- **Pipeline stages** (typical): `install → lint/typecheck → unit → build → integration → security scan → package (container) → deploy staging → E2E/smoke → deploy prod → post-deploy checks`.
- **Artifacts are immutable & versioned** — build once, promote the *same* artifact through envs (never rebuild per environment).
- **Deployment strategies** (how new code reaches prod without downtime):
  - **Rolling** — replace instances gradually.
  - **Blue/green** — two identical envs; flip traffic; instant rollback.
  - **Canary** — send 1% → 5% → 50% → 100%, watching metrics; auto-rollback on error spike.
  - **Feature flags** — deploy code *dark*, enable per-user/%/region without redeploy. Decouples *deploy* from *release*.
- **Rollback** must be a rehearsed, boring, one-command action — not an emergency.
- **DORA metrics** (how you measure delivery health): deploy frequency, lead time for change, change failure rate, mean time to recovery (MTTR). Elite teams deploy many times/day with <15% failure and recover in <1h.

### 3.5 Security (AppSec) — the discipline of assuming attack
Security is a *program*, not a checkbox. It's woven through the whole SDLC ("shift left").

- **Threat modeling**: for each feature, ask *what could go wrong?* Frameworks: **STRIDE** (Spoofing, Tampering, Repudiation, Info-disclosure, Denial-of-service, Elevation-of-privilege). Do it at design time.
- **Automated scanning in CI:**
  - **SAST** — static analysis of *your* code for vulns (Semgrep, CodeQL, Bandit).
  - **DAST** — attack the *running* app (OWASP ZAP).
  - **SCA** — scan *dependencies* for known CVEs (Dependabot, Snyk, `pip-audit`, `npm audit`).
  - **Secret scanning** — block committed keys/tokens (gitleaks, trufflehog).
- **Supply-chain security** (huge lately): pin/lock dependencies, generate an **SBOM** (software bill of materials), verify artifact provenance, sign builds (Sigstore), automate dep updates (Renovate/Dependabot). Most breaches now come *through* dependencies.
- **Secrets management**: never in code or env files committed to git. Use a vault (HashiCorp Vault, AWS Secrets Manager, KMS). **Rotate** them. Least-privilege access.
- **AuthN vs AuthZ**: *authentication* = who you are (OAuth/OIDC, MFA); *authorization* = what you may do (RBAC/ABAC, least privilege). Never roll your own crypto/auth.
- **Data protection**: encrypt **in transit** (TLS everywhere) and **at rest**; classify data (public/internal/PII/secret); minimize what you collect.
- **The usual suspects — OWASP Top 10**: injection (SQL/command), broken auth, broken access control, XSS, SSRF, security misconfiguration, vulnerable dependencies, etc. Know them.
- **Runtime defenses**: input validation & output encoding, rate limiting, WAF, audit logging (who did what, when), least-privilege IAM roles.
- **Ongoing**: periodic **penetration tests**, **bug bounty** programs, a **vulnerability disclosure** policy, and a **security incident response** plan.

### 3.6 Infrastructure & environments
- **Infrastructure as Code (IaC)**: infra defined in version-controlled files (Terraform, Pulumi, CloudFormation) — reproducible, reviewable, no "click-ops."
- **Immutable infrastructure**: never SSH in and patch a server; rebuild and replace. Cattle, not pets.
- **Containers & orchestration**: package with Docker; run/scale with Kubernetes or a managed platform (ECS, Cloud Run, Fly).
- **Environments**: `local → dev → staging → prod`, kept **as identical as possible** ("dev/prod parity" — differences cause "worked in staging" failures). Prod-like data (anonymized) in staging.
- **Config & secrets** externalized (12-Factor App): config via env, never hardcoded; same artifact, different config per env.
- **Networking**: private networks, security groups/firewalls, CDN for static assets, load balancers, autoscaling policies.

### 3.7 Data & storage
- **Schema design** & normalization; indexes for query patterns.
- **Migrations**: schema changes are versioned, forward-only, reviewed, and **reversible** (Alembic, Flyway, Prisma Migrate). A bad migration can destroy a company — treat them like surgery.
- **Backups**: automated, encrypted, **and regularly test-restored** (an untested backup is a prayer).
- **Disaster Recovery**: define **RPO** (how much data you can afford to lose) and **RTO** (how fast you must be back). Replication, multi-AZ/region as needed.
- **Data governance**: PII handling, retention policies, right-to-be-forgotten (GDPR), audit trails, access controls.

### 3.8 Reliability engineering (SRE)
The discipline of *staying up.* Assume every dependency will fail.
- **SLI / SLO / SLA**: an **SLI** is a measured indicator (e.g., % requests <200ms); an **SLO** is your internal target (99.9%); an **SLA** is the contractual promise to customers. Miss the SLO → stop shipping features and fix reliability.
- **Error budgets**: 99.9% uptime = ~43 min/month of allowed downtime. Spend it deliberately — it balances velocity vs stability.
- **Resilience patterns**: **timeouts** (never wait forever), **retries with backoff + jitter**, **circuit breakers** (stop hammering a dead dependency), **idempotency** (safe to retry), **rate limiting / throttling**, **bulkheads** (isolate failures), **graceful degradation** (serve stale/partial rather than 500), **backpressure** (shed load instead of collapsing).
- **Redundancy & no single points of failure**; **capacity planning**; **load testing** to find the ceiling *before* users do.

### 3.9 Observability (you can't fix what you can't see)
The **three pillars**:
- **Logs** — structured (JSON), leveled, correlated with a request/trace ID. Not `print()`.
- **Metrics** — time-series numbers (latency, throughput, error rate, saturation — the "four golden signals"); dashboards (Grafana, Datadog).
- **Traces** — follow one request across services (OpenTelemetry, Jaeger) to find *where* time/errors go.
- **Error tracking** — aggregate exceptions with context (Sentry).
- **Alerting** — page a human *only* for actionable, user-impacting problems (alert fatigue kills). Tie alerts to SLOs.
- **On-call**: a rotation of who responds; **runbooks** (step-by-step "if X alert, do Y"); escalation policy; **incident management** (declare severity, coordinate, comms) and **blameless postmortems** (what happened, why, what we'll change — never who to blame).

### 3.10 Release & change management
- **Semantic versioning** (`MAJOR.MINOR.PATCH`) so consumers know what broke.
- **Changelogs & release notes** (often auto-generated from Conventional Commits).
- **Progressive delivery** (canary + flags), **maintenance windows** for risky changes, **deprecation policy** (announce → warn → remove, with timelines) so you don't break integrators.
- **Change Advisory** for high-risk changes in regulated shops.

### 3.11 Documentation, DevEx & knowledge
- **README** (run it in 5 min), **CONTRIBUTING**, **architecture docs + ADRs**, **API reference** (OpenAPI/Swagger, auto-generated), **runbooks**, **onboarding guide** (new hire productive in a day).
- **Docs live next to code**, are versioned, and are updated *in the same PR* as the change.
- **DevEx**: fast local setup (one command), fast CI, good error messages — internal friction is a tax on every feature.

### 3.12 Compliance, legal & cost (the non-code realities)
- **Regulatory**: GDPR/CCPA (privacy), HIPAA (health), PCI-DSS (payments), SOC 2 / ISO 27001 (enterprise trust) — these dictate architecture, not just paperwork.
- **Licensing**: know your dependencies' licenses (GPL can be a landmine for commercial code); scan with FOSSA/`license-checker`.
- **Data Processing Agreements**, terms of service, privacy policy.
- **FinOps / cloud cost**: monitor spend, set budgets/alerts, right-size resources, tag by team/feature. Cloud bills quietly become the #2 expense.

---

## 4. The gates & checklists (copy these)

### Definition of Ready (before work starts)
- [ ] Problem & user clear; acceptance criteria written
- [ ] Designs/API contract available; dependencies identified
- [ ] Estimable and small enough to finish in a sprint

### Definition of Done (before a task is "done")
- [ ] Code merged via reviewed PR; CI green
- [ ] Unit + integration tests written and passing; coverage not decreased
- [ ] Docs/changelog updated; feature flag wired if risky
- [ ] Observability added (logs/metrics for the new path)
- [ ] Security considered (input validation, authz, secrets); no new lint/type errors

### Production Readiness Review (before GA) — the big one
- [ ] **Reliability**: SLOs defined; timeouts/retries/graceful degradation; load-tested to known ceiling
- [ ] **Observability**: dashboards, alerts tied to SLOs, structured logs, tracing, error tracking
- [ ] **Security**: threat model done; SAST/DAST/SCA/secret scans clean; authn/authz; TLS; rate limiting; pen test if warranted
- [ ] **Data**: migrations reversible; backups automated **and test-restored**; RPO/RTO defined; PII handled
- [ ] **Deploy**: IaC; staging parity; canary/blue-green; **rollback rehearsed**; feature flags
- [ ] **Ops**: on-call rotation; runbooks; incident process; escalation
- [ ] **Docs**: README, runbooks, ADRs, API docs current
- [ ] **Compliance/cost**: licenses checked; regulatory needs met; cost monitored + budget alerts

---

## 5. The maturity ladder (self-assess honestly)

| Capability | Level 0 (POC) | Level 1 (MVP) | Level 2 (Solid) | Level 3 (Elite) |
|---|---|---|---|---|
| Source control | local commits | remote + branches | PR + branch protection + CODEOWNERS | trunk-based, signed commits |
| Tests | none/manual | a few unit | pyramid + CI gate + coverage floor | + contract/load/chaos/mutation |
| CI/CD | manual deploy | CI runs tests | automated deploy to staging | canary + flags + auto-rollback |
| Security | none | deps updated | SAST/SCA/secrets in CI + authz | threat modeling + pen test + bug bounty |
| Observability | `print()` | basic logs | logs+metrics+alerts+dashboards | traces + SLOs + error budgets |
| Data | no backups | manual backups | automated + tested restores | multi-region DR + governance |
| Ops | you refresh & pray | someone watches | on-call + runbooks | incident mgmt + blameless postmortems |

You don't need Level 3 for everything — you need to **choose your target per capability, on purpose,
and write down why.** That decision *is* senior engineering.

---

## 6. Applied to FORGE (an honest gap analysis)

Where this project sits today and the prioritized path to "solid":

| Area | FORGE today | To reach Level 2 (Solid) |
|---|---|---|
| Source control | git, clean commits, CI added | ✅ close — add branch protection when collaborating |
| Tests | 34%, 15 tests, 0 frontend | Add API + pipeline tests; coverage floor in CI; a few Vitest tests |
| CI/CD | GitHub Actions runs tests/build | Add deploy step; make CI a *required* gate |
| Security | denylist `exec()`, open CORS, no authz | **Container-isolate the code sandbox**; lock CORS to the frontend origin; add rate limiting; `pip-audit`/Dependabot |
| Observability | `print()` / console | Structured logging + an error tracker (Sentry); basic metrics |
| Data/state | in-memory store, lost on restart | A datastore + job queue (RQ/Celery) for training; artifact storage |
| Reliability | synchronous thread, no limits | Timeouts, request size limits, graceful failure, load test |
| Ops/docs | README, WRITEUP, DEPLOYMENT | A runbook + a "known limitations" doc |

**For your portfolio, you don't need to *do* all of this** — but a `DESIGN.md` + this gap table shows
an interviewer you *understand the difference between a demo and a product*. That awareness is exactly
what separates junior from senior.

---

## 7. If you're solo or a small team (pragmatic subset)

You can't run all 11 roles at Level 3. The **minimum that still counts as "real":**
1. Git + PRs + **CI that gates on lint/type/test** (even reviewing your own PRs).
2. A **test pyramid** on the risky paths + a coverage floor.
3. **Secrets out of code**; deps auto-updated + scanned.
4. **One-command deploy + one-command rollback**; staging that mirrors prod.
5. **Structured logs + error tracking + a couple of alerts.**
6. **Automated, test-restored backups.**
7. **A README + a runbook** ("if it's down, do this").
8. **ADRs** for big decisions.

Everything else you add as scale, users, money, or compliance demand it — and you write down *when*
you'll add it. Maturity isn't doing everything; it's **making the trade-offs consciously and recording them.**

---

*Use this as a checklist you walk any project through. The goal is not to fear production — it's to
have already thought about the failure before it happens.*

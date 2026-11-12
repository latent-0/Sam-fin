# Income-Aware, Explainable Financial Advisory for Variable-Income Households

A reinforcement-learning-orchestrated microservice platform that answers three questions for gig workers,
freelancers, seasonal earners and new graduates:

1. **How large an instalment is safe** for *this* income process? A ruin-constrained safe EMI with a calibrated
   internal risk target.
2. **What can I change, and how long until a likely approval?** A volatility-aware, monotone loan-readiness model
   with conformal uncertainty, actionable counterfactuals and a history projection.
3. **Which advice should come first, and why?** A Double-DQN picks one of the 63 subsets of six specialist agents;
   exact Shapley values over the coalition lattice explain the choice.

Live demo: **https://my-financial-advisor.vercel.app**

The household population, agent mechanics and reward are simulated, and the volatility variables attached
to the public credit data are synthetic. See *Limitations* below.

### Beyond the paper

- **Pattern reader** (`fa/attention.py`): a two-head attention model, written and trained in NumPy, that
  reads the raw months and estimates the 12-month ruin probability. It is trained on five simulated
  income processes. It beats the smooth AR(1) estimate on long good and bad spells and on zero-income
  months, including when that income type is left out of training, and it does not help on seasonal
  income it has not seen (`paper/results/attention.json`). The UI shows its attention heads as heatmaps.
- **Shock lab**: cut income by any share, for any stretch of months, and the ruin probability, safe
  instalment, required buffer, cash paths and a month by cash-level density map are recomputed live
  (`POST /api/shock`).
- **Explanation heatmaps**: gradient x input saliency of the orchestrator's top five teams over the 14
  state features, the 6 x 6 helper interaction matrix, and the value of all 64 teams.
- **Onboarding tour** on first visit, and from the Tour button.

---

## Architecture

```
                         ┌──────────────────────────── API gateway :8100 ────────────────────────────┐
  browser ── profile ──▶ │  /api/advise  /api/advise/stream (SSE)  /api/feedback  /api/results  /     │ ──▶ advice report
                         └───────┬───────────────┬───────────────┬───────────────┬───────────┬───────┘
                       ① parallel│               │               │   ② │         ③ parallel  │
                  ┌──────────────▼──┐ ┌──────────▼──────┐ ┌──────▼─────┐ ┌──▼──────────┐ ┌──▼─────────┐ ┌─────────────┐
                  │ income   :8101  │ │ credit   :8102  │ │ feedback   │ │ rl   :8104  │ │ agents     │ │ xai  :8105  │
                  │ stability, ruin │ │ monotone GBM,   │ │ :8106      │ │ Double DQN, │ │ :8103      │ │ agent-      │
                  │ safe EMI,       │ │ conformal sets, │ │ ratings,   │ │ 63 subsets  │ │ 6 agents,  │ │ Shapley,    │
                  │ checklist       │ │ counterfactuals │ │ bias       │ │             │ │ conflicts, │ │ interactions│
                  └─────────────────┘ └─────────────────┘ └────────────┘ └─────────────┘ │ projection │ └─────────────┘
                     Monte Carlo        credit_model.txt     SQLite         dqn.npz       └────────────┘
                                                                                     shared household simulator (fa/env.py)
```

1. income, credit and the feedback bias run in parallel;
2. the orchestrator selects the agent subset from the household state plus the bias;
3. the plan builder and the Shapley explainer run in parallel on the chosen subset.

The plan builder and the explainer use the same household simulator the orchestrator was trained on, so the
plan shown to the user is exactly what the orchestrator scored.

**Two run modes, same code.** Locally (or with Docker) each service is its own uvicorn process talking HTTP
through one pooled keep-alive client. On Vercel (`app.py`, `INPROCESS=1`) every service is still its own FastAPI
app with its own contract, but the gateway calls them through in-memory ASGI transports, because a serverless
function cannot open seven ports.

## Repository layout

```
fa/                     core library (no web code)
  income.py             shrinkage log-AR(1) (OU discretisation), ISI, IaR/CVaR, ruin, safe EMI, buffer, calibrated alpha
  requirements.py       requirement-by-requirement readiness checklist
  data.py               OpenML datasets, schema harmonisation, volatility augmentation, stress-augmented label
  gbm.py                monotone LightGBM training, class-conditional conformal, metrics (AUC, PR-AUC, Brier, ECE)
  treemodel.py          dependency-free LightGBM text-model reader + exact TreeSHAP (used on Vercel)
  credit.py             loan-readiness inference, factors, counterfactual grid, history-builder
  env.py                vectorised household MDP: personas, six agents, reward, rule baseline
  dqn.py                Double DQN in NumPy (14-128-128-63), rule-guided exploration, evaluation
  agents.py             plan builder: actions, conflict resolution, 12-month projection
  xai.py                coalition values, exact Shapley values, interaction indices, Q-margins
  attention.py          pattern reader: multi-head attention over months, NumPy forward/backward
  processes.py          five simulated income processes used for stress tests and training
  features.py           model feature names (dependency-free, safe for the serverless runtime)
  feedback.py           ratings store (SQLite) and the bounded online preference update
services/               one FastAPI app per service + the gateway
frontend/               single-page UI (index.html) and example profiles (presets.json)
artifacts/              trained model files used by the services
scripts/
  train_all.py          rebuild every artifact (credit model, DQN, quick evaluation)
  train_credit.py, train_dqn.py
  paper/                one script per research question + figures (see below)
paper/results/          JSON behind every table and number in the manuscript
tests/                  unit + end-to-end tests
app.py, vercel.json     serverless entry point and function config
run_all.py              start all seven services locally
Dockerfile, docker-compose.yml
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt

python scripts/train_all.py      # downloads datasets once, trains all three models (~5 min)
python run_all.py                # starts 7 services; open http://127.0.0.1:8100
python -m pytest tests -q
```

The trained artifacts are committed, so `python run_all.py` works straight after installing; retraining
is only needed to change the models.

**Docker:** `docker compose up --build` (one image, seven containers, gateway on port 8100).

**Vercel:** `./deploy.sh` (or `vercel deploy --prod`). The function installs only `requirements.txt`
(numpy, fastapi, httpx, pydantic); the credit model is evaluated from its LightGBM text dump by
`fa/treemodel.py`, which gives identical predictions and SHAP values (checked in `tests/test_credit.py`).

## API

`POST /api/advise` (or `/api/advise/stream` for server-sent progress events) with a profile:

```json
{
  "income_history": [30000, 32000, 18000],
  "monthly_expenses": 21000, "existing_emi": 3000,
  "loan": {"amount": 300000, "annual_rate": 12, "months": 36},
  "liquid_savings": 8000, "emergency_fund": 17000, "investments": 0,
  "debt_balance": 60000, "debt_rate": 14,
  "goal_amount": 100000, "goal_saved": 10000, "goal_months": 24,
  "risk_tolerance": 0.4, "age": 29, "dependents": 1, "utilization": 0.35,
  "late_30": 0, "late_60": 0, "late_90": 0, "open_lines": 2, "missed_payments": null
}
```

The response contains `income` (stability index, model, IaR, ruin and shortfall probabilities, safe EMI,
required buffer, forecast fan, checklist), `credit` (PD, score, band, conformal set, SHAP factors,
counterfactuals, history path), `orchestrator` (chosen subset, rule subset, top-Q subsets), `plan` (actions,
conflicts, projections with and without the plan) and `explain` (Shapley values, interactions, Q-margins,
narrative). Other routes: `POST /api/feedback {bucket, mask, rating}`, `GET /api/services`,
`POST /api/shock {income_history, monthly_expenses, new_emi, buffer, drop, months, start}`,
`GET /api/results`, `GET /api/presets`.

Each service also exposes its own contract: `income /analyse`, `credit /assess`, `rl /select`,
`agents /plan`, `xai /explain`, `feedback /feedback` and `/bias/{bucket}`, and `/health` on all of them.

## Methods → code

| Paper | Code |
|---|---|
| EMI formula (eq. 1) | `fa/income.py: emi_amount` |
| Log-AR(1) / OU transition (eq. 2), shrinkage (eq. 3) | `fa/income.py: fit, simulate` |
| Income stability index (eq. 4) | `fa/income.py: isi` |
| Ruin probability (eq. 5), Proposition 1, safe EMI by bisection | `fa/income.py: ruin_probability, safe_emi, required_buffer` |
| Calibrated internal risk target alpha~(n) | `fa/income.py: CALIBRATED_ALPHA`, chosen by `scripts/paper/exp_income.py` |
| Volatility augmentation (eq. 6), stress-augmented label (eq. 7) | `fa/data.py: augment` |
| Monotone LightGBM, conformal sets, counterfactuals, history-builder | `fa/gbm.py`, `fa/credit.py` |
| Household MDP, agents (Table II), reward (eq. 8), personas (Table IX) | `fa/env.py` |
| Double DQN, rule-guided exploration, baselines | `fa/dqn.py`, `fa/env.py: rule_policy` |
| Agent-Shapley (eq. 9), interaction indices, Q-margins | `fa/xai.py` |
| Feedback loop | `fa/feedback.py` |

Main hyperparameters (Table X): 4,000 Monte Carlo paths x 12 months; shrinkage prior sigma0 = 0.25,
phi0 = 0.3, kappa = 4; GBM 400 rounds, lr 0.05, 31 leaves, min leaf 200; conformal alpha = 0.1,
class-conditional; DQN 14-128-128-63, gamma = 0.9, batch 256, 20,000 ticks, 64 envs, 2 updates/tick;
5 evaluation seeds, 3 training seeds; 32 Shapley draws per coalition in the service, 6-month horizon.

## Reproducing the paper

```bash
python scripts/train_all.py            # artifacts used by the services and by RQ4-RQ6
python scripts/paper/run_all.py        # all experiments + figures
```

| Question | Script | Output |
|---|---|---|
| RQ1 credit model: ranking, calibration, uncertainty, transfer | `exp_credit.py` | `credit.json`, `credit_shap.json` |
| RQ2 safe EMI under misspecified income, history needed | `exp_income.py` | `income.json` |
| RQ3 orchestrator vs selectors, personas, oracle, distribution shift | `exp_rl.py` | `rl.json`, `rl_returns.npz` |
| RQ4 faithfulness of agent-Shapley explanations | `exp_xai.py` | `xai.json` |
| RQ5/RQ6 counterfactuals, feedback study, latency | `exp_system.py` | `system.json` |
| Does "DQN beats the rule" depend on the reward weights? | `exp_reward.py` | `reward.json` |
| Pattern reader vs AR(1), by income type | `scripts/train_attention.py` | `attention.json` |
| Figures | `make_figures.py` | `paper/figures/*.png` |

`paper/results/` holds the results reported in the manuscript, which the web app's Evidence page reads.
Re-running the scripts regenerates them; exact values move slightly with seeds and library versions. To
experiment without overwriting them, set `RESULTS_DIR` and `FIGURES_DIR`:

```bash
RESULTS_DIR=/tmp/res FIGURES_DIR=/tmp/fig python scripts/paper/run_all.py
```

## Limitations

- No real household income records or lender decisions were used. Households, agent mechanics and the reward
  are simulated; the reward weights were set by inspection.
- The augmented credit label embeds the ruin simulation, so AUC on it is partly self-referential; AUC on the
  original public labels and a leave-one-dataset-out test are reported alongside. The model does not transfer
  to German Credit.
- The calibrated safe-EMI rule meets its target on average only with 12 or more months of history, and not for
  regime-switching income.
- The checklist is illustrative, not lender policy. The system gives preparation guidance, not credit decisions.

## Citation

S. Samhita, *Income-Aware, Explainable Financial Advisory for Variable-Income Households: A
Reinforcement-Learning-Orchestrated Microservice Architecture* (manuscript).

# robust-truck-drone-dispatch

Benchmark and research code for **dynamic truck–drone delivery dispatch under
weather, airspace, and demand uncertainty**. A fleet of ground vehicles and
drones serves a stream of online orders in an urban area: the road network has
time-dependent congestion, drone flight is governed by a Markov-chain weather
model (storms ground the fleet) and time-scheduled no-fly-zone polygons, and
orders arrive continuously over the operating day. A rolling-horizon simulator
re-opens the decision problem at event-driven epochs, an exact feasibility
checker enforces the normative rule set C1–C11, and controllers are evaluated
by **regret against a rolling re-optimization oracle** — with a distribution-free,
finite-sample audit of worst-case regret over an ε-ball of adversarially
perturbed environment parameters. An EoH-style outer loop can evolve dispatch
heuristics as Python programs against this adversarial fitness.

The Python package is named `dagcd` (Dynamic Air-Ground Collaborative
Delivery). The normative problem definition lives in
[`docs/SPECIFICATION.md`](docs/SPECIFICATION.md).

## Features

- **Exact checker** (`dagcd.checker`): hard constraints C1–C11 — coverage,
  capacity, time-dependent travel propagation, shifts, sortie validity,
  per-leg weather-weighted battery, endurance, NFZ geometry (segment–polygon
  intersection, degenerate-safe), per-drone no-overlap timeline, lateness
  accounting, horizon — plus the six-component objective `J` and full metrics.
- **Scenario generator** (`dagcd.generator`): the parameter vector θ (arrival
  rate curve, spatial clusters, congestion level, weather Markov chain, NFZ
  density) fully defines the instance distribution `D(θ)`; sampling is
  seed-reproducible, and `perturb_params` implements the ε-ball operator
  (log/additive/total-variation bounds with exact projection, optionally
  coupled along a coherent "bad-day" axis).
- **Rolling-horizon simulator** (`dagcd.simulator`): event-driven epochs
  (timer / order batch / weather change / NFZ toggle), controller re-planning
  on snapshots, deterministic execution of committed plans, `J_sim` accounting
  on actual delivery times.
- **Oracles and baselines** (`dagcd.oracle`, `dagcd.baselines`): deterministic
  regret-2 insertion oracle; a MILP epoch oracle (`MipOracle`) with dual
  backends — CPLEX via docplex and HiGHS (Apache-2.0) via highspy, solving one
  and the same formulation; greedy nearest-dispatch, dynamic ALNS, and a
  demand-lookahead reserve controller.
- **Adversarial regret audit** (`dagcd.audit`): N adversarially sampled θ′
  inside the ε-ball, paired heuristic/oracle runs, order-statistics
  (1−α)-confidence upper bound on the regret quantile, explicit inversion
  reporting.
- **LLM heuristic evolution** (`dagcd.evolve`): self-contained (stdlib-only)
  EoH-style loop that evolves controller programs against the adversarial
  regret fitness; OpenAI-compatible and Anthropic-style chat clients with
  injectable transports; full per-generation archives.

## Installation

Python ≥ 3.10. The core is pure standard library:

```bash
pip install -e .
```

Optional MIP-oracle backends (only needed for `MipOracle`/`CplexOracle`):

```bash
pip install -e ".[highs]"   # HiGHS, Apache-2.0, pip-installable
pip install -e ".[cplex]"   # docplex; additionally requires a CPLEX installation
```

## Quickstart

Generate a scenario and race two controllers under the rolling-horizon
protocol:

```python
from dagcd import (InstanceParams, sample_scenario, run_simulation,
                   GreedyController, InsertionOracle)

params = InstanceParams(sim_duration=240.0, lam_peak=0.5, n_clusters=2)
scenario = sample_scenario(params, seed=42)

greedy = run_simulation(scenario, GreedyController(), seed=0)
print("greedy   J =", round(greedy.metrics["objective"], 2),
      "| served:", greedy.metrics["served"], "/", len(scenario.orders))

oracle = run_simulation(scenario, InsertionOracle(), seed=0)
print("oracle   J =", round(oracle.metrics["objective"], 2),
      "| served:", oracle.metrics["served"], "/", len(scenario.orders))
```

Run the Shenzhen case study (real geography, cached road-network fit — see
[`case/README.md`](case/README.md)):

```python
from dagcd import load_scenario, run_simulation, GreedyController

scenario = load_scenario("case/scenario.json")
report = run_simulation(scenario, GreedyController(), seed=0)
print("case greedy J =", round(report.metrics["objective"], 2),
      "| served rate:", round(report.metrics["served_rate"], 3))
```

Load the evolved controller shipped in `controllers/` and compare it with the
greedy baseline over all five operating days of the case study (case protocol:
`epoch_delta=15.0`, `n_trig=5`):

```python
from dagcd import load_scenario, run_simulation, GreedyController
from dagcd.evolve import compile_controller

with open("controllers/evolved_v3.py", encoding="utf-8") as f:
    factory = compile_controller(f.read())

paths = ["case/scenario.json"] + [f"case/scenario_day{d}.json" for d in range(1, 5)]
for name, make in (("greedy", GreedyController), ("evolved_v3", factory)):
    total = 0.0
    for p in paths:
        sc = load_scenario(p)
        total += run_simulation(sc, make(), epoch_delta=15.0, n_trig=5,
                                seed=0).metrics["objective"]
    print(f"{name:12s} mean J over 5 days = {total / len(paths):.1f}")
```

Expected output (deterministic; evolved_v3 serves one more order on day 0 yet
trails greedy there — its advantage shows up across the week):

```
greedy       mean J over 5 days = 1208.6
evolved_v3   mean J over 5 days = 1048.5
```

Adversarial regret audit of any controller (factories are required so every
simulation gets a fresh controller instance):

```python
from dagcd import (InstanceParams, GreedyController, InsertionOracle,
                   adversarial_regret_audit)

params = InstanceParams(sim_duration=240.0, lam_peak=0.5, n_clusters=2)
report = adversarial_regret_audit(
    params,
    lambda: GreedyController(),
    lambda: InsertionOracle(),
    eps={"lam": 0.3, "cong": 0.2, "nfz": 0.3, "wx": 0.1, "clusters": 0.4},
    n_adversarial=16, nominal_seeds=5, base_seed=0)
print(report.summary["adversarial"]["regret"])
```

## Repository layout

| Path | Contents |
|---|---|
| `src/dagcd/` | the `dagcd` package: instance model, checker, generator, simulator, oracles, baselines, audit, evolution |
| `tests/` | unittest suite (checker rules, perturbation bounds, oracle optimality, audit invariants, evolution loop) |
| `docs/SPECIFICATION.md` | normative problem specification (models, constraints C1–C11, objective, epoch protocol, ε-ball semantics) |
| `controllers/evolved_v3.py` | a controller program evolved by `dagcd.evolve`, loadable with `compile_controller` |
| `case/` | Shenzhen case study: node coordinates, circuity fit, five operating-day scenarios, cached raw API responses, rebuild script |

## Running the tests

```bash
python -m unittest discover -s tests
```

CPLEX-dependent tests skip automatically when `docplex` is not importable;
HiGHS-dependent tests skip without `highspy`. Everything else runs on the
standard library alone.

## Case study data

`case/` contains an urban low-altitude logistics scenario built on the
Nanshan–Bao'an area of Shenzhen: 27 hand-picked real-world locations (1 depot,
3 vertiports, 23 customers), road distances approximated by a scalar circuity
factor fitted against 45 queried OD pairs and validated on 8 held-out pairs,
a rainy-season weather Markov chain, and two scheduled no-fly zones (airport
afternoon restriction plus a temporary event zone). Five one-day scenarios
(`scenario.json`, `scenario_day1..4.json`) are included; `build_case.py`
rebuilds them offline from the cached raw API responses. See
[`case/README.md`](case/README.md) for details.

## Citation

If you use this code, please cite the accompanying paper:

> J. X. Cao, "Robust Heuristic Design for Dynamic Truck–Drone Delivery
> Dispatch Under Weather, Airspace, and Demand Uncertainty," *IEEE
> Transactions on Intelligent Transportation Systems*, under review, 2026.

## License

MIT — see [LICENSE](LICENSE).

## Contact

Jin Xin Cao — imucjx@163.com

# Dynamic Truck–Drone Delivery Dispatch — Problem Specification

This document is the normative definition of the benchmark problem implemented by
`dagcd` (Dynamic Air-Ground Collaborative Delivery). It governs `checker.py`,
`generator.py`, and all experiments: when code and this document disagree, this
document wins.

---

## 1. Narrative & scope

A logistics operator serves delivery orders in an urban area. Each order is a
parcel that must be carried from the operator's depot/vertiport network to a
customer location.

- **Ground vehicles** travel on the road network with **time-dependent** travel
  times (congestion).
- **Drones** fly straight-line legs at fixed cruise speed, are **not** affected
  by road congestion, but are constrained by battery, payload, weather, and
  **no-fly zones (NFZ)**.
- **Collaboration**: at each decision epoch every released order is assigned to
  (a) a ground route, (b) a drone sortie, or (c) deferred/unserved (with
  penalty).
- **Dynamics**: new orders arrive continuously; weather evolves; NFZs
  activate/deactivate; the controller re-plans the *remaining* problem at
  decision epochs.

### Explicit exclusions (may become extensions)

- No mid-route truck–drone rendezvous (drones launch/land only at vertiports).
- No pickup-and-delivery (depot → customer delivery only).
- Stochastic optimization is handled at the *scenario/distribution* level
  (Sec. 6), not as chance constraints in the checker.

---

## 2. Sets and notation

| Symbol | Meaning |
|---|---|
| `V` | service nodes: customer locations, depot set `D`, vertiport set `P ⊆ V` |
| `O` | orders released up to current epoch |
| `K_g`, `K_a` | ground vehicles, drones |
| `H = [0, T]` | planning horizon of the current epoch |
| `Z(t)` | set of active NFZ polygons at time `t` |

### Order `o ∈ O`

`o = (n_o, r_o, w_o, s_o, e_o, l_o, p_o)` — destination node, release time,
weight (kg), service duration, window start, window end (soft), per-unit
lateness penalty.

### Ground vehicle `k ∈ K_g`

Capacity `Q_g` (kg), start depot `depot(k)`, shift end `shiftEnd(k)`.

### Drone `m ∈ K_a`

Payload `Q_a` (kg), battery `B` (Wh), cruise speed `v_a`, max sortie duration
`L_max`, weather energy multiplier table `mu(weather)`.

---

## 3. Travel and energy models

### Ground (time-dependent)

`tau_g(i, j, t) = dist(i, j) / (v_g * c(t))`, where `c(t) ∈ (0, 1]` is a
piecewise-constant congestion factor (lookup table on time-of-day).
Deterministic given departure time `t`.

### Drone (weather-dependent)

`tau_a(i, j, wx) = dist(i, j) / (v_a * speed_factor(wx))`

Energy of a leg `(i, j)` carrying load `w`:

`e(i, j, w, wx) = (alpha + beta * w) * dist(i, j) * mu(wx)`

with constants `alpha` (Wh/m, empty), `beta` (Wh/m/kg). Battery drains by leg;
load `w` on a leg equals the weight of all not-yet-delivered parcels aboard.

**Weather states**: `clear | windy | rain | storm`. `storm` grounds all drones
(`speed_factor = 0`, launch forbidden).

### No-fly zones

Each NFZ is a polygon. A drone leg between two points is **infeasible** if the
segment intersects any polygon in `Z(t_leg)` (closed sets) — checked at the
leg's departure time `t_leg`.

---

## 4. Decision variables (per epoch, remaining problem)

- `x_ok ∈ {0,1}` — order `o` served by ground vehicle `k`
- `y_om ∈ {0,1}` — order `o` served by drone `m`
- Ground routes: sequence `σ_k = (depot, o_1, ..., o_n, depot)` with arrival
  times
- Drone sorties: `π_m = (p0, [o_1..o_q], p1, t_launch)` — vertiports
  `p0, p1 ∈ P`
- `u_o ≥ 0` — lateness of order `o`

Each released order is served **exactly once** (ground or drone) or appears in
`unserved` (paying penalty `p_o^unserved`).

---

## 5. Constraint catalog (checker normative list)

**C1 Order coverage.** Every released order is served exactly once across
ground routes and drone sorties, or declared unserved. No duplicates.

**C2 Ground capacity.** `Σ_{o in σ_k} w_o ≤ Q_g` for every ground route.

**C3 Ground time propagation.** For consecutive stops `i → j` with service
start `t_i`: departure `= t_i + s_i`; arrival at `j`
`≥ t_i + s_i + tau_g(i, j, t_i + s_i)` (tolerance `TOL = 1e-6`). Service start
at `j` = `max(arrival, e_j)` (waiting allowed).

**C4 Ground shift.** Route starts at `depot(k)` at `t_start(k) ≤ shiftEnd(k)`;
return to any depot `≤ shiftEnd(k)`.

**C5 Drone sortie validity.** Each sortie: launch/land at vertiports;
`q ≤ M_max` orders (`M_max` = max parcels per sortie); per-order payload
`w_o ≤ Q_a`; cumulative load on every leg `≤ Q_a`.

**C6 Drone battery.** Total energy over legs ≤ `B` (weather multiplier applied
per leg).

**C7 Drone endurance.** Sortie wall-clock duration `≤ L_max`.

**C8 Drone time windows & NFZ.** Same arrival-time propagation as C3 (with
`tau_a`); every leg checked against `Z(t_leg)`; launch forbidden in `storm`.

**C9 Lateness accounting.** `u_o = max(0, service_start_o - l_o)`; objective
and metrics must use the same `u_o`.

**C10 Horizon.** All service starts within `[0, T]` (unserved orders exempt).

**C11 Drone timeline.** Each drone flies at most one sortie at a time: sorties
of the same drone, sorted by launch time, must satisfy
`launch_{k+1} >= land_k` (tolerance `TOL`). Chaining — launching again after
landing, possibly from a different vertiport — is allowed. Without C11 an epoch
solution could schedule physically impossible overlapping sorties on one drone
that C1–C10 cannot detect.

### Objective (single-epoch evaluation function, used as AHD fitness)

```
J = Σ_k fixedCost_g                     (per used ground vehicle)
  + Σ_m fixedCost_a                     (per used drone sortie)
  + varCost_g * ground_km
  + varCost_a * drone_km
  + Σ_o u_o * p_o                       (lateness penalties)
  + Σ_{o unserved} p_o^unserved
```

`checker.py` returns violations (hard constraints C1–C8, C10–C11) **and**
computes `J` plus metrics. Hard violations ⇒ solution infeasible, fitness =
large constant.

### 5.1 Robustness criterion

Absolute `J` is not a sufficient fitness for an online dispatch heuristic: the
same `J` means different quality under busy and quiet demand. Fitness is
therefore **regret against an oracle**:

```
R(h, θ') = E_{I ~ D(θ')} [ J_h(I) − J_oracle(I) ]
```

where `oracle` = rolling re-optimization under the same epoch protocol
(baseline (iii), insertion/LNS). Robustness criterion: bounded and low
**worst-case regret inside the ε-ball**, `sup_{θ' ∈ B_ε(θ)} R(h, θ')`; the
adversarial inner loop searches `argmax_{θ' ∈ B_ε} R(h, θ')` — regret against
the oracle, **not** the worst absolute-fitness instance.

The ε-ball is defined on the **environment generative parameters θ** (Sec. 6):
structured, multi-dimensional and correlated — joint perturbation of λ(t),
clusters, congestion, weather chain and NFZ density with projection back onto
the ball. This is a deliberate step toward distribution-level DRO.

**Audit protocol.** Formal certificates for arbitrary evolved code are out of
reach; the methodology claims *auditability*, not *provability*: adversarially
sample N parameter vectors θ′ᵢ from B_ε and report the distribution-free
order-statistics bound U = r_(k) — the k-th smallest sample regret — where k is
the smallest integer with `P(Binomial(N, q) ≤ k−1) ≥ 1−α`, giving a
(1−α)-confidence upper bound on the q-quantile of the regret distribution over
B_ε (implemented as `dagcd.audit._quantile_ub_index`, verified against
brute-force binomial sums in tests). The bound exists only if some k ≤ N
qualifies; for the worst-case (max) statement k = N this requires `q^N ≤ α` —
e.g. (α = 0.05, q = 0.95) needs **N ≥ 59**. Always co-report tail metrics
(95th-percentile lateness, worst-scenario cost). Recommended wording:
*auditable worst-case regret* — never *certified/provable*.

---

## 6. Online protocol & disturbance model

### Decision epochs

Trigger set: every `Δ` minutes **or** on event (batch of new orders ≥ `n_trig`,
weather state change, NFZ activation). At epoch `τ`, the controller sees all
released orders and replans the remaining routes/sorties (insertion allowed;
rerouting allowed). Simulation advances to the next epoch.

### Instance distribution parameterization `θ`

Instances are drawn from a parameterized family:

| Parameter | Meaning |
|---|---|
| `λ(t)` | order arrival rate curve (peak/trough multiplier) |
| `clusters` | spatial cluster centers & spreads of order destinations |
| `cong_level` | congestion curve severity |
| `wx_chain` | Markov chain transition matrix over weather states |
| `nfz_density` | expected # active NFZs per hour |

`θ` fully defines the training distribution `D(θ)`.

### Adversarial disturbance (robustness evaluation)

Given a training set from `D(θ)`, an adversary perturbs `θ → θ'` within an
`ε`-ball (e.g. `|log(λ'/λ)| ≤ ε_λ`, `‖clusters' − clusters‖ ≤ ε_c`, WX
perturbation bounded in total variation). A discovered heuristic is
**robust-certified** if its regret on the worst `θ'` inside the ball stays
below a threshold. This is the inner loop of robust AHD evolution.

**Coupled sampling.** `perturb_params(..., couple=True)` (default) makes the
multi-dimensional perturbation *correlated*, not just componentwise: with
probability 1/2 it draws a coherent "bad-day" axis — λ′ up (busier), NFZ
density up, WX transition matrix pushed stormward, congestion up — while every
magnitude is still sampled independently inside its own ε-bound and the
per-component bounds (log / additive / TV ≤ ε) hold exactly in both modes, with
the same legality-first projection; the other 1/2 (and the `clusters` spread
component, whose direction carries no intrinsic "bad" sense — and whose centres
are not part of `InstanceParams`, spread-only by construction) fall back to
independent box sampling. `couple=False` reproduces the uncoupled
(independent-box) draws bit-for-bit, so scenarios sampled with the uncoupled
operator stay reproducible.

---

## 7. Evaluation protocol

- **Metrics**: served rate, avg/95th-percentile lateness, cost per served order
  (J / served), ground-km, drone-km, Wh per order, runtime per epoch.
- **Baselines**: (i) greedy nearest-dispatch; (ii) insertion heuristic
  (OR-Tools style); (iii) static re-optimize-all each epoch with time-inflated
  travel times (rolling-horizon MILP/LNS); (iv) literature HGS on the static
  snapshot per epoch.
- **Robustness split**: nominal `D(θ)` test set + adversarial `θ'` sets
  (3 perturbation magnitudes).
- **A/B rule**: any evolved heuristic must beat (iii) on cost per order on
  nominal **and** show ≤ 5% degradation on adversarial sets to be declared
  non-trivial.
- **Adversarial audit (Sec. 5.1)**: every reported heuristic ships with a
  finite-sample worst-case regret bound over `B_ε` (N adversarially sampled θ′,
  confidence 1−α), plus an absolute-fitness ablation that isolates the effect
  of the regret objective.

---

## 8. Solution JSON schema (normative)

```jsonc
{
  "epoch": 0,
  "ground_routes": [
    {"vehicle": "gv0", "stops": [
       {"order": "o1", "node": 5, "arrival": 12.5},
       {"order": "o2", "node": 9, "arrival": 31.0}],
     "return_depot": 0, "return_time": 58.2}
  ],
  "drone_sorties": [
    {"drone": "da0", "launch_vertiport": 1, "launch_time": 5.0,
     "stops": [{"order": "o7", "node": 14, "arrival": 9.1}],
     "land_vertiport": 2, "land_time": 18.4, "energy_wh": 412.0}
  ],
  "unserved": ["o13"]
}
```

Arrival times are *claimed* service starts; the checker re-derives earliest
feasible times and flags any claim earlier than the earliest feasible
(C3/C8 propagation).

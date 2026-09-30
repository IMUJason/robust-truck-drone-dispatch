"""Adversarial regret audit harness — docs/SPECIFICATION.md Sec. 5.1 "Audit protocol".

Implements the paper's audit protocol: sample N parameter vectors theta'_i from
the epsilon-ball B_eps(theta) around a baseline theta, pair-run one heuristic and
one oracle per scenario under the rolling-horizon protocol, and report the
finite-sample worst-case regret with a distribution-free (1-alpha) confidence
upper bound on the `quantile`-quantile of the regret distribution over B_eps.

Invariants (enforced by construction and asserted by tests/test_audit.py)
---------------------------------------------------------------------------
1. Reproducibility. Same (params, factories, eps, sizes, base_seed) => identical
   AuditReport dict (``audit_to_dict`` equality across runs). Scenario seeds are
   derived deterministically from base_seed; the adversarial rng is
   ``random.Random(base_seed)`` consumed in fixed order by ``perturb_params``.
   Determinism is *conditional on the factories being deterministic*: a
   CplexOracle at a binding time limit may return different incumbents run to
   run (wall-time-dependent search, see oracle.py Invariant 5) — the harness
   never promises bitwise reproducibility through a non-deterministic oracle.
2. Regret definition. ``regret = J_heuristic - J_oracle`` where J is
   ``SimReport.metrics["objective"]`` (J_sim accounting, docs/SPECIFICATION.md
   Sec. 5), both
   controllers facing the *same* sampled scenario with the *same* sim seed.
3. Fresh controller instances. Every simulation calls ``factory()`` anew;
   controllers are never shared across simulations (a factory carrying internal
   state across runs would break Invariant 1). The descriptor stored in the
   report (class name + module + config) is built from a separate probe
   instantiation, so anyone can rebuild the exact controller objects.
4. quantile_ub (order-statistics bound). With regret samples sorted ascending
   r_(1..N), the confidence upper bound on the ``quantile``-quantile is U = r_(k)
   where k is the smallest integer with

       P( Binomial(N, quantile) <= k-1 ) >= 1 - alpha,

   computed exactly with ``math.comb`` (no normal approximation). If no k <= N
   satisfies the condition, U = None and a note is recorded. This is the
   distribution-free order-statistics bound in the spirit of the scenario
   approach (Campi & Garatti and successors): the k-th order statistic exceeds
   the true p-quantile only if fewer than k of N i.i.d. draws fell below it,
   which under the binomial tail bound happens with probability <= alpha.
   The paper wording is *auditable* worst-case regret — never *certified*.
5. Inversions are reported, never hidden. Every sample with
   ``regret < -TOL`` (oracle worse than heuristic) lands in
   ``summary["inversions"]`` with its numbers; an oracle is a baseline, not a
   mathematical lower bound, so small negative regrets are legitimate and must
   stay visible.
6. Timing fields are excluded from stored metrics (``runtime_per_epoch_avg`` is
   wall-clock and would break Invariant 1). All other
   ``SimReport.metrics`` fields are stored verbatim for adversarial samples.
7. Oracle fallback transparency. The oracle controller is wrapped so every
   ``oracle_note`` attached to an epoch solution (MILP size-cap fallback, solver
   error fallback, ...) is collected into ``summary["notes"]`` with the scenario
   context — approximations are documented, never silent (oracle.py contract).

I/O mirrors generator.py's scenario JSON helpers: ``audit_to_dict`` /
``audit_from_dict`` / ``save_audit`` / ``load_audit``.
"""

from __future__ import annotations

import json
import math
import random
import statistics
from dataclasses import asdict, dataclass, field, is_dataclass

from .generator import InstanceParams, perturb_params, sample_scenario
from .simulator import run_simulation

__all__ = ["AuditReport", "adversarial_regret_audit",
           "_quantile_ub_index", "audit_to_dict", "audit_from_dict",
           "save_audit", "load_audit"]

TOL = 1e-6          # inversion threshold: regret < -TOL counts as an inversion
_RUNTIME_KEYS = ("runtime_per_epoch_avg",)   # wall-clock fields (Invariant 6)


@dataclass
class AuditReport:
    """Audit report of one adversarial regret audit (JSON-serializable).

    ``heuristic`` / ``oracle`` are factory descriptors: {"class", "module",
    "config"} sufficient to rebuild identical controller objects. ``nominal``
    holds one dict per nominal seed; ``adversarial`` one dict per perturbed
    theta' sample (with the full heuristic metrics verbatim). ``summary``
    aggregates regret over both segments and carries the quantile bound,
    inversion list and anomaly notes.
    """

    theta: dict                      # asdict of the baseline InstanceParams
    eps: dict
    couple: bool = True              # perturb_params coupling mode (generator.py;
                                     # reports produced with the uncoupled
                                     # operator lack this field and correspond
                                     # to couple=False draws)
    alpha: float = 0.05
    quantile: float = 0.95
    n_adversarial: int = 0
    nominal_seeds: int = 0
    base_seed: int = 0
    heuristic: dict = field(default_factory=dict)   # {"class", "module", "config"}
    oracle: dict = field(default_factory=dict)
    nominal: list = field(default_factory=list)
    adversarial: list = field(default_factory=list)
    summary: dict = field(default_factory=dict)


# ---------------- order-statistics quantile bound ----------------

def _binomial_cdf(k_max: int, n: int, p: float) -> float:
    """P(Binomial(n, p) <= k_max), exact via math.comb (Invariant 4)."""
    if k_max < 0:
        return 0.0
    k_max = min(k_max, n)
    q = 1.0 - p
    return sum(math.comb(n, j) * p ** j * q ** (n - j) for j in range(k_max + 1))


def _quantile_ub_index(n: int, quantile: float, alpha: float) -> int | None:
    """Smallest k with P(Binomial(n, quantile) <= k-1) >= 1-alpha, or None.

    Returns the index (1-based, into ascending-sorted samples) of the
    (1-alpha)-confidence upper order statistic for the ``quantile``-quantile
    (see module Invariant 4 for the formula and its scenario-approach basis).
    None means n is too small to bound the quantile at this (alpha, quantile).
    """
    if n < 1:
        return None
    for k in range(1, n + 1):
        if _binomial_cdf(k - 1, n, quantile) >= 1.0 - alpha:
            return k
    return None


# ---------------- helpers ----------------

def _jsonable(obj):
    """Deep-convert dataclasses/tuples to JSON-safe structures (Invariant 3)."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return repr(obj)


def _factory_descriptor(factory) -> dict:
    """Build a rebuild descriptor from a probe instantiation of ``factory``.

    The probe is a *separate* call; per-run controllers are created inside the
    simulation loops (Invariant 3).
    """
    probe = factory()
    cls = type(probe)
    if is_dataclass(probe):
        cfg = asdict(probe)
    else:
        try:
            cfg = dict(vars(probe))
        except TypeError:
            cfg = {}
    return {"class": cls.__name__, "module": cls.__module__,
            "config": _jsonable(cfg)}


def _scrub_metrics(metrics: dict) -> dict:
    """Drop wall-clock fields so stored metrics are deterministic (Invariant 6)."""
    return {k: v for k, v in metrics.items() if k not in _RUNTIME_KEYS}


class _NoteSniffingController:
    """Delegate wrapper that records ``oracle_note`` fallbacks (Invariant 7)."""

    def __init__(self, inner, notes: list, context: str):
        self._inner = inner
        self._notes = notes
        self._context = context

    def __call__(self, snap, view, rng):
        sol = self._inner(snap, view, rng)
        note = getattr(sol, "oracle_note", None)
        if note:
            self._notes.append(f"{self._context}: oracle_note: {note}")
        return sol


def _regret_stats(values: list) -> dict:
    """{mean, std, max, q95} for a list of regrets (population std; q95 uses the
    same ceil convention as the simulator's p95)."""
    if not values:
        return {"mean": None, "std": None, "max": None, "q95": None}
    s = sorted(values)
    n = len(s)
    return {
        "mean": statistics.fmean(s),
        "std": statistics.pstdev(s),
        "max": s[-1],
        "q95": s[min(n - 1, math.ceil(0.95 * n) - 1)],
    }


def _run_pair(scenario, scenario_seed, heuristic_factory, oracle_factory,
              notes: list, context: str, sim_kwargs: dict):
    """Pair-run heuristic and oracle on the same scenario (Invariant 1/2/3).

    Returns (J_heuristic, J_oracle, heuristic_metrics) with wall-clock fields
    scrubbed from the stored metrics.
    """
    h_rep = run_simulation(scenario, heuristic_factory(), **sim_kwargs)
    o_rep = run_simulation(scenario, _NoteSniffingController(
        oracle_factory(), notes, context), **sim_kwargs)
    j_h = h_rep.metrics["objective"]
    j_o = o_rep.metrics["objective"]
    return j_h, j_o, _scrub_metrics(h_rep.metrics)


# ---------------- main entry ----------------

def adversarial_regret_audit(params: InstanceParams, heuristic_factory,
                             oracle_factory, *, eps: dict,
                             n_adversarial: int = 16, nominal_seeds: int = 5,
                             alpha: float = 0.05, quantile: float = 0.95,
                             base_seed: int = 0, sim_kwargs: dict | None = None,
                             couple: bool = True) -> AuditReport:
    """Run the Sec.5.1 adversarial regret audit of ``heuristic_factory`` against
    ``oracle_factory`` inside the eps-ball around ``params``.

    Protocol:
      * nominal segment — seeds s = 0..nominal_seeds-1, scenario_seed =
        base_seed*100000 + s, one scenario from D(theta), paired runs;
      * adversarial segment — rng = random.Random(base_seed); theta'_i =
        perturb_params(params, eps, rng) (rng consumed in order, so the theta'
        sequence is reproducible); scenario_seed_i = base_seed*100000 + 1000 + i,
        paired runs;
      * regret_i = J_heuristic - J_oracle with J = SimReport.metrics["objective"];
      * summary carries per-segment regret stats, the (1-alpha) confidence upper
        bound ``quantile_ub`` on the ``quantile``-quantile of the adversarial
        regret distribution (order statistics, see _quantile_ub_index), the
        inversion list (regret < -TOL, Invariant 5) and anomaly notes.

    ``sim_kwargs`` is passed through to ``run_simulation`` (epoch_delta, n_trig,
    cost, ...); if it contains ``seed`` that value is used for the controller
    rng of every run, otherwise the controller rng seed defaults to 0.
    ``couple`` selects the perturb_params coupling mode (generator.py; default
    True = 50% coherent bad-day axis + 50% independent box) and is recorded in
    the report so archives stay self-describing.

    Determinism (Invariant 1): identical inputs give identical reports *provided
    the factories build deterministic controllers* — a CplexOracle under a
    binding time limit can legitimately vary run to run (oracle.py Invariant 5).
    Every simulation uses a fresh ``factory()`` instance (Invariant 3); a shared
    stateful factory would void the guarantee.
    """
    sim_kwargs = dict(sim_kwargs or {})
    theta = asdict(params)
    notes: list[str] = []

    report = AuditReport(
        theta=theta, eps=dict(eps or {}), couple=couple, alpha=alpha,
        quantile=quantile,
        n_adversarial=n_adversarial, nominal_seeds=nominal_seeds,
        base_seed=base_seed,
        heuristic=_factory_descriptor(heuristic_factory),
        oracle=_factory_descriptor(oracle_factory),
    )

    inversions: list[dict] = []
    nominal_regrets: list[float] = []
    adversarial_regrets: list[float] = []

    # ---- nominal segment ----
    for s in range(nominal_seeds):
        scenario_seed = base_seed * 100000 + s
        sc = sample_scenario(params, scenario_seed)
        j_h, j_o, metrics_h = _run_pair(
            sc, scenario_seed, heuristic_factory, oracle_factory, notes,
            f"nominal s={s}", sim_kwargs)
        regret = j_h - j_o
        nominal_regrets.append(regret)
        if regret < -TOL:
            inversions.append({"segment": "nominal", "index": s,
                               "scenario_seed": scenario_seed,
                               "J_heuristic": j_h, "J_oracle": j_o,
                               "regret": regret})
        report.nominal.append({
            "scenario_seed": scenario_seed,
            "J_heuristic": j_h, "J_oracle": j_o, "regret": regret,
            "served_rate_heuristic": metrics_h["served_rate"],
            "p95_lateness_heuristic": metrics_h["p95_lateness"],
        })

    # ---- adversarial segment ----
    rng = random.Random(base_seed)
    for i in range(n_adversarial):
        theta_prime = perturb_params(params, eps, rng, couple=couple)
        scenario_seed = base_seed * 100000 + 1000 + i
        sc = sample_scenario(theta_prime, scenario_seed)
        j_h, j_o, metrics_h = _run_pair(
            sc, scenario_seed, heuristic_factory, oracle_factory, notes,
            f"adversarial i={i}", sim_kwargs)
        regret = j_h - j_o
        adversarial_regrets.append(regret)
        if regret < -TOL:
            inv = {"segment": "adversarial", "index": i,
                   "scenario_seed": scenario_seed,
                   "theta_prime": asdict(theta_prime),
                   "J_heuristic": j_h, "J_oracle": j_o, "regret": regret}
            inversions.append(inv)
            notes.append(f"inversion at adversarial i={i}: regret={regret:.6f} "
                         f"(J_h={j_h:.6f}, J_o={j_o:.6f}); the oracle baseline "
                         "lost to the heuristic on this sample")
        report.adversarial.append({
            "index": i,
            "theta_prime": asdict(theta_prime),
            "scenario_seed": scenario_seed,
            "J_heuristic": j_h, "J_oracle": j_o, "regret": regret,
            "served_rate_heuristic": metrics_h["served_rate"],
            "p95_lateness_heuristic": metrics_h["p95_lateness"],
            "metrics_heuristic": metrics_h,
        })

    # ---- quantile upper bound ----
    k = _quantile_ub_index(n_adversarial, quantile, alpha)
    if k is None:
        quantile_ub = None
        notes.append(f"N={n_adversarial} too small: no (alpha={alpha}, "
                     f"quantile={quantile}) upper bound exists for the "
                     "adversarial-regret quantile; rerun with more samples")
    else:
        srt = sorted(adversarial_regrets)
        quantile_ub = srt[k - 1]

    report.summary = {
        "nominal": {"regret": _regret_stats(nominal_regrets)},
        "adversarial": {"regret": _regret_stats(adversarial_regrets)},
        "quantile_ub": quantile_ub,
        "quantile_ub_index": k,
        "inversions": inversions,
        "notes": notes,
    }
    return report


# ---------------- JSON I/O ----------------

def audit_to_dict(rep: AuditReport) -> dict:
    return asdict(rep)


def audit_from_dict(d: dict) -> AuditReport:
    return AuditReport(**d)


def save_audit(rep: AuditReport, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit_to_dict(rep), f, indent=2)


def load_audit(path: str) -> AuditReport:
    with open(path, "r", encoding="utf-8") as f:
        return audit_from_dict(json.load(f))

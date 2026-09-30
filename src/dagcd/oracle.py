"""Epoch-level re-optimization oracle for DAGCD (docs/SPECIFICATION.md Sec. 5.1 /
Sec. 7 baseline iii).

This module is the source of ``J_oracle`` for the regret objective
(docs/SPECIFICATION.md Sec. 5.1: ``R(h, theta') = E[J_h(I) - J_oracle(I)]``) and doubles as the
rolling re-optimize-all baseline (iii) — a ``time_inflation`` factor can inflate the
fixed travel times inside the MILP exactly as baseline (iii) time-inflates its
re-optimization, while output claims are always re-derived with the true models.

Approximation contract (documented, never silent)
-------------------------------------------------
* **Fixed-departure ground linearization.** Inside the MILP the ground travel time
  ``tau_g(i, j)`` is frozen at its value at ``t_fix = view.tau`` (optionally
  inflated by ``time_inflation``). The C10 horizon is enforced on all service
  starts, and the C4 return leg is enforced as a *fixed-tau strengthening*
  (``Tm[last] + s + tau_fix(last, nearest depot) <= shift_end`` on the terminal
  arc). Because the true time-dependent tau can differ, the output routes are
  re-propagated with the true ``tau_g`` (earliest-feasible semantics identical to
  ``simulator._propagate_ground``) and any residual C4/C10 violation caused by the
  linearization is repaired (see ``_repair_ground``).
* **Sortie enumeration is exact.** Drone sorties (1 or 2 parcels, up to ``M_max``)
  are *enumerated*, not decided inside the MILP: each candidate is forward-propagated
  with the true per-leg ``tau_a`` under checker-exact storm / NFZ / energy semantics,
  pre-checked for payload, remaining battery (``view.drones[d].energy_wh``),
  endurance ``L_max`` and the C10 horizon, and priced as
  ``fixed_sortie + var_drone*km + own lateness penalty``. All candidates of one drone
  launch at the same earliest instant ``max(available_at, tau)``; therefore any two
  candidates of the same drone overlap in time and the spec's pairwise overlap
  constraints ``a_{s,d} + a_{s',d} <= 1`` reduce to the single clique constraint
  ``sum_s a_{s,d} <= 1`` (one sortie per idle drone per epoch — chaining is left to
  later epochs, where the simulator forwards ``available_at``).
* **Size cap.** If the snapshot carries more than ``max_orders`` orders, the MILP is
  skipped and a deterministic :class:`InsertionOracle` plan is returned instead; the
  returned ``Solution`` carries ``sol.oracle_note`` describing the approximation
  (also documented here — never silent).

Dual backends (open-source reproducibility + solver sensitivity)
----------------------------------------------------------------
:class:`MipOracle` solves the *same* docplex model object with either backend:

* ``backend="cplex"`` — docplex/CPLEX (lazy import), the validated path.
* ``backend="highs"`` — the model is exported to an LP file (variable names are
  preserved; all variables carry explicit ``name=``) and solved by HiGHS
  (Apache-2.0) through ``highspy``; solution values are mapped back by name and
  fed through the identical extraction code. Both backends therefore optimise
  exactly the same formulation — solver is the only varied factor.

Provenance attributes attached to every returned ``Solution``:

* ``sol.oracle_backend`` — ``"cplex"`` | ``"highs"`` (the requested backend; an
  InsertionOracle fallback keeps the backend name, see ``oracle_note``).
* ``sol.oracle_proven`` — ``True`` iff the backend returned an *optimality proof*
  for the epoch MIP (CPLEX: solve status optimal or relative MIP gap within
  ``mip_gap``; HiGHS: model status ``kOptimal`` or MIP gap within ``mip_gap``).
  Proven implies the optimal *objective value* is unique, but the solution *path*
  need not be: multiple optimal routes/sortie sets can exist, and the two backends
  (or repeated runs at a binding time limit) may legitimately return different
  equal-value plans. Bit-identical cross-backend trajectories are NOT a contract —
  objective-value agreement on mutually proven snapshots is the consistency
  criterion (asserted in ``tests/test_oracle.py``).

Warm starts (from :class:`InsertionOracle`) are applied on the CPLEX path only;
LP export does not carry MIP starts, and HiGHS relies on its own MIP heuristics.
A CP-SAT backend (OR-Tools) would require re-expressing the model (circuit
constraints etc.) and is deliberately deferred: it would confound "solver
sensitivity" with "model re-implementation". Recorded as future work.

Invariants (enforced by construction and asserted by tests)
-----------------------------------------------------------
1. Every returned ``Solution`` passes ``checker.check_solution`` on the snapshot it
   was built from (hard guarantee; enforced by a self-check with escalating repair,
   ending in an all-unserved fallback that is always feasible).
2. Claimed times are the *earliest feasible* times under the snapshot's true
   time-dependent ``tau_g`` / ``tau_a`` (equality with the checker's re-derived
   earliest values, not merely ``>=``).
3. ``docplex`` / ``highspy`` are imported lazily inside the backend solvers;
   importing this module and using :class:`InsertionOracle` never requires either.
4. The MILP objective and the checker's objective of the returned solution coincide
   whenever the fixed-tau model equals the true model (e.g. constant congestion);
   verified against brute force in ``tests/test_oracle.py`` for BOTH backends.
5. Both oracles are deterministic functions of ``(snapshot, view, cost)`` up to
   solver behavior: runs are identical whenever the backend proves optimality; at
   a binding time limit the incumbent may vary run-to-run (wall-time-dependent
   search) — every variant still passes the checker (Invariant 1). ``rng`` is
   accepted for controller-protocol compatibility but never consumed.
"""

from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass

from .checker import CostParams, check_solution
from .instance import TOL, DroneSortie, DroneStop, GroundRoute, GroundStop, Solution
from .simulator import FleetView, _leg_blocked

__all__ = ["InsertionOracle", "MipOracle", "CplexOracle", "enumerate_sorties"]

_FALLBACK_NOTE = ("MILP skipped ({reason}); deterministic InsertionOracle plan "
                  "returned instead (documented approximation)")


# ---------------- sortie enumeration ----------------

@dataclass
class SortiePlan:
    """One pre-priced drone sortie candidate (true per-leg forward pass, checker-exact)."""

    drone: str
    launch_vertiport: int
    launch_time: float
    stops: list          # list[DroneStop], earliest-feasible claimed arrivals
    land_vertiport: int
    land_time: float
    energy_wh: float
    km: float
    late_pen: float      # lateness penalties of contained orders
    cost: float          # fixed_sortie + var_drone*km + late_pen
    orders: tuple        # covered order ids


def _drone_leg(snap, dr, i: int, j: int, t: float, load: float):
    """One drone leg departing at ``t`` carrying ``load`` kg.

    Mirrors the checker's derivation exactly: storm (or any zero-speed weather) at
    departure blocks the leg, NFZ status is read at departure, energy uses the
    per-leg weather multiplier. Returns (travel_time, energy_wh) or None if blocked.
    """
    wx = snap.weather_at(t)
    if wx == "storm":
        return None
    travel = snap.tau_a(i, j, t, dr.cruise_speed)
    if math.isinf(travel):
        return None
    if _leg_blocked(snap, i, j, t):
        return None
    d = snap.dist(i, j)
    energy = (snap.energy_alpha + snap.energy_beta * load) * d \
        * snap.energy_mu.get(wx, 1.0)
    return travel, energy


def enumerate_sorties(snap, view: FleetView, cost: CostParams) -> list:
    """All feasible 1- and 2-parcel sorties for every idle drone (module docstring).

    Launch is the earliest feasible instant ``max(available_at, view.tau)``; every
    stop arrival must respect the C10 horizon; landings may occur at any vertiport.
    """
    vertiports = sorted(nid for nid, n in snap.nodes.items() if n.kind == "vertiport")
    T_end = snap.horizon
    onboard = {oid for v in view.vehicles.values() for oid in v.onboard}
    free_ids = sorted(oid for oid in snap.orders if oid not in onboard)
    out: list[SortiePlan] = []
    if not vertiports:
        return out
    for did in sorted(view.drones):
        dv = view.drones[did]
        if dv.busy_until is not None:
            continue  # only idle drones may launch (spec)
        dr = snap.drones[did]
        max_q = min(2, dr.max_parcel)
        if max_q < 1:
            continue
        p0 = dv.vertiport
        if snap.nodes[p0].kind != "vertiport":
            continue
        launch = max(dv.available_at, view.tau)
        if snap.weather_at(launch) == "storm":
            continue  # takeoff forbidden in storm (C8)
        avail_energy = min(dr.battery_wh, dv.energy_wh)

        # ---- 1-parcel candidates ----
        singles: dict[str, tuple] = {}
        for oid in free_ids:
            o = snap.orders[oid]
            if o.weight > dr.payload + TOL:
                continue
            leg1 = _drone_leg(snap, dr, p0, o.node, launch, o.weight)
            if leg1 is None:
                continue
            arr = launch + leg1[0]
            if arr > T_end + TOL:
                continue
            start = max(arr, o.e)
            dep = start + o.service
            u = max(0.0, start - o.l) * o.lateness_penalty
            singles[oid] = (arr, start, dep, leg1[1], snap.dist(p0, o.node))
            for p1 in vertiports:
                leg2 = _drone_leg(snap, dr, o.node, p1, dep, 0.0)
                if leg2 is None:
                    continue
                land = dep + leg2[0]
                if land - launch > dr.max_sortie_min + TOL:
                    continue
                energy = leg1[1] + leg2[1]
                if energy > avail_energy + TOL:
                    continue
                km = (snap.dist(p0, o.node) + snap.dist(o.node, p1)) / 1000.0
                out.append(SortiePlan(
                    drone=did, launch_vertiport=p0, launch_time=launch,
                    stops=[DroneStop(oid, o.node, arr)],
                    land_vertiport=p1, land_time=land, energy_wh=energy,
                    km=km, late_pen=u,
                    cost=cost.fixed_sortie + cost.var_drone_per_km * km + u,
                    orders=(oid,)))

        # ---- 2-parcel candidates (ordered pairs; both aboard on the first leg) ----
        if max_q >= 2:
            seconds = [oid for oid in free_ids
                       if snap.orders[oid].weight <= dr.payload + TOL]
            wx0 = snap.weather_at(launch)
            for a in list(singles):
                oa = snap.orders[a]
                arr_a, start_a, dep_a, _e_single, d_a = singles[a]
                for b in seconds:
                    if b == a:
                        continue
                    ob = snap.orders[b]
                    load_ab = oa.weight + ob.weight
                    if load_ab > dr.payload + TOL:
                        continue
                    # leg 1 carries BOTH parcels: recompute its energy at combined
                    # load (the singles entry used oa's weight alone — C6 re-derivation
                    # by the checker uses the true aboard load)
                    e1 = (snap.energy_alpha + snap.energy_beta * load_ab) * d_a \
                        * snap.energy_mu.get(wx0, 1.0)
                    leg2 = _drone_leg(snap, dr, oa.node, ob.node, dep_a, ob.weight)
                    if leg2 is None:
                        continue
                    arr_b = dep_a + leg2[0]
                    if arr_b > T_end + TOL:
                        continue
                    start_b = max(arr_b, ob.e)
                    dep_b = start_b + ob.service
                    u_b = max(0.0, start_b - ob.l) * ob.lateness_penalty
                    u_a = max(0.0, start_a - oa.l) * oa.lateness_penalty
                    for p1 in vertiports:
                        leg3 = _drone_leg(snap, dr, ob.node, p1, dep_b, 0.0)
                        if leg3 is None:
                            continue
                        land = dep_b + leg3[0]
                        if land - launch > dr.max_sortie_min + TOL:
                            continue
                        energy = e1 + leg2[1] + leg3[1]
                        if energy > avail_energy + TOL:
                            continue
                        km = (d_a + snap.dist(oa.node, ob.node)
                              + snap.dist(ob.node, p1)) / 1000.0
                        out.append(SortiePlan(
                            drone=did, launch_vertiport=p0, launch_time=launch,
                            stops=[DroneStop(a, oa.node, arr_a),
                                   DroneStop(b, ob.node, arr_b)],
                            land_vertiport=p1, land_time=land, energy_wh=energy,
                            km=km, late_pen=u_a + u_b,
                            cost=cost.fixed_sortie + cost.var_drone_per_km * km
                                 + u_a + u_b,
                            orders=(a, b)))
    return out


# ---------------- ground forward propagation (true time-dependent tau_g) ----------------

def _forward_ground(snap, veh, seq: list, depots: list, T_end: float):
    """Earliest-feasible ground route for stop sequence ``seq`` (onboard first).

    Semantics identical to ``simulator._propagate_ground`` but *reporting* instead of
    discarding: returns (stops, return_depot, return_time, late_pen, km, ok_shift,
    ok_horizon) where the ok-flags compare against C4 (start/return vs shift_end) and
    C10 (claimed arrivals <= T_end). Never returns None — violations are the caller's.
    """
    t = veh.start_time
    prev = veh.start_depot
    stops: list[GroundStop] = []
    late = 0.0
    km = 0.0
    for oid in seq:
        o = snap.orders[oid]
        arr = max(t + snap.tau_g(prev, o.node, t), o.release)
        start = max(arr, o.e)
        km += snap.dist(prev, o.node) / 1000.0
        late += max(0.0, start - o.l) * o.lateness_penalty
        stops.append(GroundStop(oid, o.node, arr))
        t = start + o.service
        prev = o.node
    if depots:
        rd = min(depots, key=lambda d: (snap.dist(prev, d), d))
        rt = t + snap.tau_g(prev, rd, t)
        km += snap.dist(prev, rd) / 1000.0
    else:
        rd, rt = prev, t
    ok_shift = veh.start_time <= veh.shift_end + TOL and rt <= veh.shift_end + TOL
    ok_horizon = all(s.arrival <= T_end + TOL for s in stops)
    return stops, rd, rt, late, km, ok_shift, ok_horizon


def _vehicle_usable(veh) -> bool:
    """A vehicle whose shift ends before it can start cannot carry anything."""
    return veh.start_time <= veh.shift_end + TOL


# ---------------- shared solution assembly ----------------

def _assemble(snap, view, seq_routes: dict, sortie_plans: list,
              unserved: list) -> Solution:
    """Turn per-vehicle stop sequences + chosen sortie plans into a Solution with
    earliest-feasible claimed times (Invariant 2). Empty sequences become no route."""
    depots = sorted(nid for nid, n in snap.nodes.items() if n.kind == "depot")
    T_end = snap.horizon
    routes = []
    for vid in sorted(seq_routes):
        seq = seq_routes[vid]
        if not seq:
            continue
        stops, rd, rt, _late, _km, _ok_s, _ok_h = _forward_ground(
            snap, snap.vehicles[vid], seq, depots, T_end)
        routes.append(GroundRoute(vid, stops, rd, rt))
    sorties = [DroneSortie(sp.drone, sp.launch_vertiport, sp.launch_time,
                           list(sp.stops), sp.land_vertiport, sp.land_time,
                           sp.energy_wh)
               for sp in sortie_plans]
    return Solution(epoch=0, ground_routes=routes, drone_sorties=sorties,
                    unserved=list(unserved))


def _repair_ground(snap, view, seq_routes: dict, unserved: list) -> None:
    """Repair C4/C10 violations caused by the fixed-tau linearization.

    Iteratively (<= 3 rounds) remove the largest-lateness removable order from each
    violating vehicle and re-propagate with the true tau_g; orders on vehicles that
    still violate afterwards are dropped to unserved. Onboard parcels are never
    removable (they are physically on the truck).
    """
    depots = sorted(nid for nid, n in snap.nodes.items() if n.kind == "depot")
    T_end = snap.horizon
    onboard = {vid: set(view.vehicles[vid].onboard) for vid in view.vehicles}

    def violations(vid):
        seq = seq_routes[vid]
        if not seq:
            return False
        _stops, _rd, _rt, _late, _km, ok_shift, ok_horizon = _forward_ground(
            snap, snap.vehicles[vid], seq, depots, T_end)
        return not (ok_shift and ok_horizon)

    for _round in range(3):
        bad = [vid for vid in sorted(seq_routes) if violations(vid)]
        if not bad:
            return
        for vid in bad:
            seq = seq_routes[vid]
            removable = [oid for oid in seq if oid not in onboard.get(vid, set())]
            if not removable:
                continue  # prefix itself infeasible: final sweep decides
            # largest current lateness contribution along the route
            veh = snap.vehicles[vid]
            contrib: dict[str, float] = dict.fromkeys(removable, 0.0)
            t, prev = veh.start_time, veh.start_depot
            for oid in seq:
                o = snap.orders[oid]
                arr = max(t + snap.tau_g(prev, o.node, t), o.release)
                start = max(arr, o.e)
                if oid in contrib:
                    contrib[oid] = max(0.0, start - o.l) * o.lateness_penalty
                t = start + o.service
                prev = o.node
            worst = max(removable, key=lambda oid: (contrib[oid], oid))
            seq.remove(worst)
            unserved.append(worst)
    # final sweep: vehicles still violating drop all removable orders; if the onboard
    # prefix itself is infeasible, the whole sequence is given up (fallback net)
    for vid in sorted(seq_routes):
        if not violations(vid):
            continue
        seq = seq_routes[vid]
        removable = [oid for oid in seq if oid not in onboard.get(vid, set())]
        for oid in removable:
            seq.remove(oid)
            unserved.append(oid)
        if violations(vid):
            for oid in list(seq):
                seq.remove(oid)
                unserved.append(oid)


def _checked(snap, cost: CostParams, sol: Solution, note: str | None):
    """Return ``sol`` if it passes the checker (annotated with ``note``), else None."""
    rep = check_solution(snap, sol, cost)
    if rep.violations:
        return None
    sol.oracle_note = note
    return sol


# ---------------- InsertionOracle ----------------

class InsertionOracle:
    """Deterministic regret-insertion oracle (insertion/LNS-style baseline,
    docs/SPECIFICATION.md Sec. 7).

    Ground: classic regret-2 insertion — every iteration inserts the unassigned order
    with the largest regret (second-best minus best marginal insertion cost, including
    the fixed-vehicle cost trigger) at its best position. Drone: afterwards, remaining
    orders greedily take the cheapest feasible sortie option (candidates from
    :func:`enumerate_sorties`, at most one per drone, true per-leg pre-checks).
    Orders feasible by neither are declared unserved (retried at later epochs).

    Inherits the module invariants: output always passes ``check_solution`` and claims
    earliest-feasible times. Deterministic; ``rng`` is accepted per the controller
    protocol but never consumed.

    Redesign history (adversarial-audit driven, base_seed=42): a unified ground/drone regret competition — literal interleaved
    pool (inversions 37.5% -> 81%, phantom-sortie deferral strands the final release
    wave), drones-first pre-pass (mean
    adversarial regret -117, empty-route alt inflation), and true-marginal drone
    substitution with and without C11 chaining (-85.6 / -185.5, per-epoch J gains
    lost to cross-epoch myopia and shift-end slack erosion) ALL underperformed this
    v1 design. Root cause of the oracle's residual inversions vs GreedyController is
    rolling-horizon lateness myopia (long committed routes deliver late; per-epoch
    regret-2 cannot see it), not drone starvation — a documented caveat of the
    regret-reference design, not fixable by pool design alone.
    """

    def __init__(self, cost: CostParams | None = None):
        self.cost = cost or CostParams()

    def __call__(self, snap, view: FleetView, rng: random.Random) -> Solution:
        depots = sorted(nid for nid, n in snap.nodes.items() if n.kind == "depot")
        T_end = snap.horizon
        routes: dict[str, list] = {vid: list(view.vehicles[vid].onboard)
                                   for vid in sorted(view.vehicles)}
        prefix = {vid: len(ids) for vid, ids in routes.items()}
        load = {vid: sum(snap.orders[o].weight for o in ids)
                for vid, ids in routes.items()}
        usable = {vid for vid in snap.vehicles
                  if _vehicle_usable(snap.vehicles[vid])}
        onboard = {oid for ids in routes.values() for oid in ids}
        pending = sorted((o for o in snap.orders.values() if o.id not in onboard),
                         key=lambda o: (o.e, o.release, o.id))

        # ---- ground regret-2 insertion ----
        while pending:
            best = None  # (regret, delta, order, vid, pos); maximize regret
            for o in pending:
                opts = []  # (delta, vid, pos)
                for vid in sorted(usable):
                    base = routes[vid]
                    if load[vid] + o.weight > snap.vehicles[vid].capacity + TOL:
                        continue
                    cur = _forward_ground(snap, snap.vehicles[vid], base, depots,
                                          T_end)
                    for pos in range(prefix[vid], len(base) + 1):
                        seq = base[:pos] + [o.id] + base[pos:]
                        prop = _forward_ground(snap, snap.vehicles[vid], seq,
                                               depots, T_end)
                        if not (prop[5] and prop[6]):
                            continue  # C4/C10 violated by the insertion
                        delta = self.cost.var_ground_per_km * (prop[4] - cur[4]) \
                            + (prop[3] - cur[3])
                        if not base and not view.vehicles[vid].used:
                            delta += self.cost.fixed_ground
                        opts.append((delta, vid, pos))
                if not opts:
                    continue
                opts.sort(key=lambda t: (t[0], t[1], t[2]))
                c1, c2 = opts[0], (opts[1] if len(opts) > 1 else None)
                regret = c2[0] - c1[0] if c2 else c1[0]
                cand = (regret, c1[0], o, c1[1], c1[2])
                if best is None or (cand[0] > best[0]) or \
                        (cand[0] == best[0] and (cand[1], cand[2].id)
                         < (best[1], best[2].id)):
                    best = cand
            if best is None:
                break
            _regret, _delta, o, vid, pos = best
            routes[vid].insert(pos, o.id)
            load[vid] += o.weight
            pending.remove(o)

        # ---- drone greedy for the remainder ----
        chosen: list[SortiePlan] = []
        taken: set[str] = set()
        served_ground = {oid for ids in routes.values() for oid in ids}
        for sp in sorted(enumerate_sorties(snap, view, self.cost),
                         key=lambda s: (s.cost, s.drone, s.orders)):
            if any(oid in taken or oid in onboard or oid in served_ground
                   for oid in sp.orders):
                continue
            if any(c.drone == sp.drone for c in chosen):
                continue  # one sortie per drone (clique constraint)
            chosen.append(sp)
            taken.update(sp.orders)

        served = {oid for ids in routes.values() for oid in ids}
        unserved = sorted(o.id for o in snap.orders.values()
                          if o.id not in served and o.id not in taken)
        sol = _assemble(snap, view, routes, chosen, unserved)
        checked = _checked(snap, self.cost, sol, None)
        if checked is not None:
            return checked
        # should never happen; ultimate safe net (always feasible)
        sol = Solution(epoch=0, ground_routes=[], drone_sorties=[],
                       unserved=sorted(snap.orders))
        sol.oracle_note = "InsertionOracle self-check failed; all-unserved fallback"
        return sol


# ---------------- MipOracle (dual-backend epoch re-optimization) ----------------

@dataclass
class _EpochCtx:
    """Variable/structure mapping handed from the model builder to the backend
    solver and the shared extractor (backend-independent extraction logic)."""

    x: dict
    y: dict
    a: dict
    g: dict
    z: dict
    used: list
    sorties: list
    s_index: list
    idx: dict
    orders_all: list
    free_idx: list
    vids: list
    onboard: dict
    S_CODE: dict
    T_CODE: int
    drone_of: dict


class MipOracle:
    """Epoch re-optimization oracle via MILP with dual backends (module docstring).

    Rolling re-optimize-all over the remaining problem (docs/SPECIFICATION.md
    Sec. 5.1 ``J_oracle`` source; with ``time_inflation > 1`` it is the
    time-inflated variant of baseline (iii)). One docplex model — multi-vehicle CVRPTW with arc variables ``x[i,j,v]``
    (per-vehicle start arcs, single terminal node charged at the nearest-depot
    distance), MTZ positions, per-vehicle capacity, hard C10 horizon, in-model
    return-leg shift bound (C4, fixed-tau strengthening), soft lateness and
    unserved ``z_o``; plus enumerated sortie candidates with selection ``y_s`` and
    assignment ``a_{s,d}`` under the one-sortie-per-drone clique — is solved by
    either backend: ``"cplex"`` (docplex, lazy import, warm-started from the
    insertion plan) or ``"highs"`` (LP export -> ``highspy`` -> name-mapped
    extraction, Apache-2.0, no warm start). Objective:
    ``fixed_ground*used + var_ground*km(fixed tau) + sum y_s c_s + sum p_o u_o
    + sum p_unserved z_o`` — minimized.

    Every returned ``Solution`` carries provenance: ``sol.oracle_backend``
    (``"cplex"`` | ``"highs"``) and ``sol.oracle_proven`` (optimality proof
    returned by the backend; see module docstring for the exact semantics and the
    multiple-optima caveat). After each epoch call the instance also exposes
    ``self.last_epoch`` = ``{"objective", "proven", "backend", "solve_time"}`` for
    diagnostics (``objective`` is the epoch-MIP objective, ``None`` on fallback).

    Solve params (configurable): 120 s time limit, 1e-4 MIP gap; on timeout the
    incumbent is used. Approximations are documented in the module docstring and
    flagged on the returned solution via ``sol.oracle_note`` whenever they activate.
    """

    _BACKENDS = ("cplex", "highs")

    def __init__(self, cost: CostParams | None = None, *, backend: str = "highs",
                 max_orders: int = 30, time_limit: float = 120.0,
                 mip_gap: float = 1e-4, time_inflation: float = 1.0,
                 threads: int | None = None):
        if backend not in self._BACKENDS:
            raise ValueError(f"unknown backend {backend!r}; expected one of "
                             f"{self._BACKENDS}")
        self.backend = backend
        self.cost = cost or CostParams()
        self.max_orders = max_orders
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.time_inflation = time_inflation
        # Pinned thread count for reproducible incumbents under a time limit.
        # threads=1 removes thread-scheduling nondeterminism (the main source
        # of cross-machine drift in time-limited incumbents); epochs whose
        # wall-clock limit actually binds can still vary with machine load.
        # The default None leaves the solver's own default, which uses all
        # cores — see Invariant 5.
        self.threads = threads
        self.last_epoch: dict | None = None

    def _warn_fallback(self, note: str) -> None:
        """Emit a one-time process-level warning on a solver-error fallback.

        The per-solution ``oracle_note`` (Invariant on documented fallbacks)
        is the machine-readable record; this warning makes the same event
        visible on stderr for interactive use, where a missing CPLEX runtime
        would otherwise silently turn a MILP experiment into an insertion
        experiment (once per oracle instance, not per epoch).
        """
        if not getattr(self, "_fallback_warned", False):
            self._fallback_warned = True
            warnings.warn(
                f"{type(self).__name__}: solver fallback active, results are "
                f"NOT from the {self.backend} MILP — {note}", RuntimeWarning,
                stacklevel=3)

    # -- controller protocol --
    def __call__(self, snap, view: FleetView, rng: random.Random) -> Solution:
        self.last_epoch = None
        note = None
        if len(snap.orders) > self.max_orders:
            sol = InsertionOracle(self.cost)(snap, view, rng)
            sol.oracle_note = _FALLBACK_NOTE.format(
                reason=f"{len(snap.orders)} orders > max_orders={self.max_orders}")
            self._provenance(sol, False)
            return sol
        try:
            solved = self._solve_epoch(snap, view)
        except ImportError:
            solved = None
            note = _FALLBACK_NOTE.format(reason=f"{self.backend} backend unavailable")
            self._warn_fallback(note)
        except Exception as exc:  # solver failure: documented fallback, never silent
            solved = None
            note = _FALLBACK_NOTE.format(reason=f"solver error: {exc!r}")
            self._warn_fallback(note)
        if solved is None:
            sol = InsertionOracle(self.cost)(snap, view, rng)
            sol.oracle_note = note
            self._provenance(sol, False)
            return sol
        plan, proven = solved
        seq_routes, chosen, unserved = plan
        _repair_ground(snap, view, seq_routes, unserved)
        unserved = sorted(set(unserved))
        sol = _assemble(snap, view, seq_routes, chosen, unserved)
        checked = _checked(snap, self.cost, sol, note)
        if checked is not None:
            self._provenance(checked, proven)
            return checked
        # escalate: drop violating sorties one by one, then insertion, then all-unserved
        for sp in list(chosen):
            probe_unserved = sorted(set(unserved + list(sp.orders)))
            probe = _assemble(snap, view, seq_routes,
                              [s for s in chosen if s is not sp], probe_unserved)
            checked = _checked(snap, self.cost, probe, note)
            if checked is not None:
                self._provenance(checked, proven)
                return checked
        sol = InsertionOracle(self.cost)(snap, view, rng)
        checked = _checked(snap, self.cost, sol, note)
        if checked is not None:
            sol.oracle_note = ((note + "; ") if note else "") + \
                "post-MILP self-check failed; InsertionOracle fallback"
            self._provenance(sol, False)
            return sol
        sol = Solution(epoch=0, ground_routes=[], drone_sorties=[],
                       unserved=sorted(snap.orders))
        sol.oracle_note = "all-unserved fallback (checker safety net)"
        self._provenance(sol, False)
        return sol

    def _provenance(self, sol: Solution, proven: bool) -> None:
        sol.oracle_backend = self.backend
        sol.oracle_proven = bool(proven)

    # -- backend dispatch --
    def _solve_epoch(self, snap, view: FleetView):
        mdl, ctx = self._build_model(snap, view)
        if self.backend == "cplex":
            return self._solve_cplex(mdl, ctx, snap, view)
        return self._solve_highs(mdl, ctx)

    # -- MILP model construction (backend-independent) --
    def _build_model(self, snap, view: FleetView):
        from docplex.mp.model import Model  # lazy: module import never needs docplex

        cost = self.cost
        t_fix = view.tau
        T_end = snap.horizon
        depots = sorted(nid for nid, n in snap.nodes.items() if n.kind == "depot")
        orders_all = sorted(snap.orders)
        idx = {oid: k for k, oid in enumerate(orders_all)}
        F = len(orders_all)
        vids = sorted(v for v in snap.vehicles
                      if _vehicle_usable(snap.vehicles[v]))
        nv = len(vids)
        S_CODE = {v: F + 2 * k for k, v in enumerate(vids)}
        T_CODE = F + 2 * nv  # single shared terminal (nearest-depot return)
        node_of = {idx[o]: snap.orders[o].node for o in orders_all}
        start_node = {S_CODE[v]: snap.vehicles[v].start_depot for v in vids}

        def tau_fix(i: int, j: int) -> float:
            return snap.tau_g(i, j, t_fix) * self.time_inflation

        onboard = {vid: [oid for oid in view.vehicles[vid].onboard
                         if oid in snap.orders] for vid in vids}
        onboard_set = {o for ids in onboard.values() for o in ids}
        free_ids = [oid for oid in orders_all if oid not in onboard_set]
        free_idx = [idx[o] for o in free_ids]

        sorties = enumerate_sorties(snap, view, cost)
        s_index = list(range(len(sorties)))
        by_order: dict[str, list[int]] = {oid: [] for oid in free_ids}
        for si, sp in enumerate(sorties):
            for oid in sp.orders:
                if oid in by_order:
                    by_order[oid].append(si)
        drone_of = {did: k for k, did in enumerate(sorted(view.drones))}
        sorties_of_drone: dict[str, list[int]] = {}
        for si, sp in enumerate(sorties):
            sorties_of_drone.setdefault(sp.drone, []).append(si)

        mdl = Model(name="dagcd_epoch")

        x_keys = []
        for k, v in enumerate(vids):
            pre = onboard[v]
            if pre:
                x_keys.append((S_CODE[v], idx[pre[0]], k))
                for a, b in zip(pre, pre[1:]):
                    x_keys.append((idx[a], idx[b], k))
                last = idx[pre[-1]]
                for j in free_idx:
                    x_keys.append((last, j, k))
                x_keys.append((last, T_CODE, k))
            else:
                for j in free_idx:
                    x_keys.append((S_CODE[v], j, k))
            for i in free_idx:
                for j in free_idx:
                    if i != j:
                        x_keys.append((i, j, k))
                x_keys.append((i, T_CODE, k))
        x = mdl.binary_var_dict(x_keys, name="x")
        g = mdl.binary_var_dict(free_idx, name="g")
        z = mdl.binary_var_dict(free_idx, name="z")
        u = mdl.continuous_var_dict(free_idx, lb=0.0, name="u")
        # pos/Tm exist for ALL orders (incl. the onboard prefix): prefix arcs and
        # prefix position fixing reference them; free-order bounds are added below
        pos = mdl.continuous_var_dict(range(F), lb=0.0, ub=max(F, 1), name="pos")
        Tm = mdl.continuous_var_dict(range(F), lb=0.0, ub=T_end, name="Tm")
        used = mdl.binary_var_list(range(nv), name="used")
        y = mdl.binary_var_dict(s_index, name="y")
        a = mdl.binary_var_dict(
            ((si, drone_of[sp.drone]) for si, sp in enumerate(sorties)), name="a")

        M_pos = max(F, 1)
        max_s = max((snap.orders[o].service for o in orders_all), default=0.0)

        # Tightened per-arc big-M for time propagation. Validity of
        #   Tm[j] >= Tm[i] + s_i + tau_ij - M_ij (1 - x_ij)
        # at x_ij = 0 requires M_ij >= Tm[i]^max + s_i + tau_ij - Tm[j]^min; with
        # Tm <= T_end (C10) and Tm[j] >= e_j this holds for
        # M_ij = T_end + s_i + tau_ij - e_j (a global M would be much looser, and
        # loose big-M time constraints are the main weakness of the LP relaxation).
        def m_arc(i_idx, j_idx):
            oi = snap.orders[orders_all[i_idx]]
            oj = snap.orders[orders_all[j_idx]]
            return max(0.0, T_end + oi.service
                       + tau_fix(node_of[i_idx], node_of[j_idx]) - oj.e)

        def m_start(k, j_idx):
            veh = snap.vehicles[vids[k]]
            oj = snap.orders[orders_all[j_idx]]
            trav = tau_fix(veh.start_depot, node_of[j_idx])
            return max(0.0, veh.start_time + trav - oj.e)

        # return-leg shift bound (C4): a route whose last stop i returns to the
        # nearest depot after shift_end is infeasible. Under the fixed-tau model
        # this is Tm[i] + s_i + tau_fix(i, rd_i) <= shift_end. The true-tau check
        # remains the post-hoc repair's job (congestion may relax after t_fix, so
        # this in-model bound is only a strengthening of the fixed-tau model).
        max_tau_ret = 1.0
        for i in free_idx:
            ni = node_of[i]
            rd_i = min(depots, key=lambda d: (snap.dist(ni, d), d))
            max_tau_ret = max(max_tau_ret, tau_fix(ni, rd_i))
        M_ret = T_end + max_tau_ret + max_s + 1.0

        def ret_shift_ok(i_idx, k):
            v = vids[k]
            ni = node_of[i_idx]
            rd_i = min(depots, key=lambda d: (snap.dist(ni, d), d))
            mdl.add_constraint(
                Tm[i_idx] + snap.orders[orders_all[i_idx]].service
                + tau_fix(ni, rd_i)
                <= snap.vehicles[v].shift_end
                + M_ret * (1 - x[(i_idx, T_CODE, k)]))

        def inflow_vars(k, j):
            v = vids[k]
            pre = onboard[v]
            vs = []
            if (S_CODE[v], j, k) in x:
                vs.append(x[(S_CODE[v], j, k)])
            if pre and (idx[pre[-1]], j, k) in x:
                vs.append(x[(idx[pre[-1]], j, k)])
            for i in free_idx:
                if i != j and (i, j, k) in x:
                    vs.append(x[(i, j, k)])
            return vs

        for k, v in enumerate(vids):
            pre = onboard[v]
            if pre:
                mdl.add_constraint(x[(S_CODE[v], idx[pre[0]], k)] == 1)
                for aa, bb in zip(pre, pre[1:]):
                    mdl.add_constraint(x[(idx[aa], idx[bb], k)] == 1)
                mdl.add_constraint(used[k] == 1)
            out_s = [x[(S_CODE[v], j, k)] for j in free_idx
                     if (S_CODE[v], j, k) in x]
            mdl.add_constraint(mdl.sum(out_s) <= 1)
            if not pre:
                mdl.add_constraint(used[k] == mdl.sum(out_s))
            # per-vehicle capacity (onboard prefix weight included)
            pre_w = sum(snap.orders[oid].weight for oid in pre)
            mdl.add_constraint(
                mdl.sum(snap.orders[orders_all[j]].weight * xx
                        for j in free_idx for xx in inflow_vars(k, j))
                + pre_w <= snap.vehicles[v].capacity)
            # arc-implication cuts: an arc used by vehicle k forces both endpoints
            # to be ground-served (strengthens the LP relaxation coupling)
            for (i, j, kk), xx in x.items():
                if kk != k:
                    continue
                if j in g:
                    mdl.add_constraint(xx <= g[j])
                if i in g:
                    mdl.add_constraint(xx <= g[i])

        for j in free_idx:
            oj = orders_all[j]
            mdl.add_constraint(g[j] == mdl.sum(
                xx for k in range(nv) for xx in inflow_vars(k, j)))
            mdl.add_constraint(g[j] + mdl.sum(y[si] for si in by_order[oj]) + z[j]
                               == 1)
            mdl.add_constraint(Tm[j] >= snap.orders[oj].release)
            mdl.add_constraint(Tm[j] >= snap.orders[oj].e)
            mdl.add_constraint(u[j] >= Tm[j] - snap.orders[oj].l)

        for k, v in enumerate(vids):
            veh = snap.vehicles[v]
            pre = onboard[v]
            for j in free_idx:
                inf = inflow_vars(k, j)
                out = [x[(j, jj, k)] for jj in free_idx
                       if jj != j and (j, jj, k) in x]
                out.append(x[(j, T_CODE, k)])
                mdl.add_constraint(mdl.sum(inf) == mdl.sum(out))
                if depots:
                    ret_shift_ok(j, k)
            if pre and depots:
                ret_shift_ok(idx[pre[-1]], k)
            # time propagation (tightened per-arc big-M, see m_arc/m_start above)
            if pre:
                first = idx[pre[0]]
                mdl.add_constraint(
                    Tm[first] >= veh.start_time
                    + tau_fix(veh.start_depot, node_of[first])
                    - m_start(k, first) * (1 - x[(S_CODE[v], first, k)]))
                for aa, bb in zip(pre, pre[1:]):
                    ia, ib = idx[aa], idx[bb]
                    mdl.add_constraint(
                        Tm[ib] >= Tm[ia] + snap.orders[aa].service
                        + tau_fix(node_of[ia], node_of[ib])
                        - m_arc(ia, ib) * (1 - x[(ia, ib, k)]))
            for j in free_idx:
                if (S_CODE[v], j, k) in x:
                    mdl.add_constraint(Tm[j] >= veh.start_time
                                       + tau_fix(veh.start_depot, node_of[j])
                                       - m_start(k, j) * (1 - x[(S_CODE[v], j, k)]))
                if pre and (idx[pre[-1]], j, k) in x:
                    i = idx[pre[-1]]
                    mdl.add_constraint(Tm[j] >= Tm[i]
                                       + snap.orders[pre[-1]].service
                                       + tau_fix(node_of[i], node_of[j])
                                       - m_arc(i, j) * (1 - x[(i, j, k)]))
                for i in free_idx:
                    if i != j and (i, j, k) in x:
                        mdl.add_constraint(
                            Tm[j] >= Tm[i] + snap.orders[orders_all[i]].service
                            + tau_fix(node_of[i], node_of[j])
                            - m_arc(i, j) * (1 - x[(i, j, k)]))
            # MTZ positions
            for j in free_idx:
                if (S_CODE[v], j, k) in x:
                    mdl.add_constraint(pos[j] >= 1 - M_pos * (1 - x[(S_CODE[v], j, k)]))
                if pre and (idx[pre[-1]], j, k) in x:
                    i = idx[pre[-1]]
                    mdl.add_constraint(pos[j] >= len(pre) + 1
                                       - M_pos * (1 - x[(i, j, k)]))
                for i in free_idx:
                    if i != j and (i, j, k) in x:
                        mdl.add_constraint(pos[j] >= pos[i] + 1
                                           - M_pos * (1 - x[(i, j, k)]))
            if pre:
                for n_p, oid_p in enumerate(pre, start=1):
                    mdl.add_constraint(pos[idx[oid_p]] == n_p)

        for si in s_index:
            d = drone_of[sorties[si].drone]
            mdl.add_constraint(a[(si, d)] == y[si])
        for did, sis in sorties_of_drone.items():
            d = drone_of[did]
            mdl.add_constraint(mdl.sum(a[(si, d)] for si in sis) <= 1)

        def km_arc(i_code, j_code):
            node_i = start_node.get(i_code, node_of.get(i_code))
            if j_code == T_CODE:
                return min(snap.dist(node_i, dpt) for dpt in depots) / 1000.0
            node_j = start_node.get(j_code, node_of.get(j_code))
            return snap.dist(node_i, node_j) / 1000.0

        ground_var = mdl.sum(cost.var_ground_per_km * km_arc(i, j) * xx
                             for (i, j, _k), xx in x.items())
        ground_fixed = mdl.sum(cost.fixed_ground * used[k] for k in range(nv))
        sortie_cost = mdl.sum(sorties[si].cost * y[si] for si in s_index)
        late_cost = mdl.sum(snap.orders[orders_all[j]].lateness_penalty * u[j]
                            for j in free_idx)
        unserved_cost = mdl.sum(snap.orders[orders_all[j]].unserved_penalty * z[j]
                                for j in free_idx)
        mdl.minimize(ground_fixed + ground_var + sortie_cost + late_cost
                     + unserved_cost)

        ctx = _EpochCtx(x=x, y=y, a=a, g=g, z=z, used=used, sorties=sorties,
                        s_index=s_index, idx=idx, orders_all=orders_all,
                        free_idx=free_idx, vids=vids, onboard=onboard,
                        S_CODE=S_CODE, T_CODE=T_CODE, drone_of=drone_of)
        return mdl, ctx

    # -- backend: CPLEX --
    def _solve_cplex(self, mdl, ctx: _EpochCtx, snap, view: FleetView):
        from time import perf_counter
        from docplex.util.status import JobSolveStatus

        mdl.set_time_limit(self.time_limit)
        mdl.parameters.mip.tolerances.mipgap = self.mip_gap
        if self.threads is not None:
            mdl.parameters.threads = self.threads

        # Warm start from the deterministic insertion plan (a feasible MIP start
        # gives CPLEX a strong incumbent immediately; purely a hint — any failure
        # here silently falls back to the solver's own heuristics). CPLEX-only:
        # LP export for the HiGHS path does not carry MIP starts.
        cost = self.cost
        try:
            warm_sol = InsertionOracle(cost)(snap, view, random.Random(0))
            warm: dict = {}
            sig = {(sp.drone, sp.orders, sp.land_vertiport,
                    sp.launch_vertiport, sp.launch_time): si
                   for si, sp in enumerate(ctx.sorties)}
            for k, v in enumerate(ctx.vids):
                pre = ctx.onboard[v]
                by_stop = {}
                for r in warm_sol.ground_routes:
                    if r.vehicle == v:
                        for n_st, s in enumerate(r.stops):
                            by_stop[n_st] = ctx.idx[s.order]
                if not by_stop:
                    continue
                seq_idx = [ctx.idx[o] for o in pre] + \
                          [by_stop[n] for n in sorted(by_stop)]
                prev_code = ctx.idx[pre[-1]] if pre else ctx.S_CODE[v]
                for j in seq_idx[len(pre):]:
                    warm[ctx.x[(prev_code, j, k)]] = 1
                    prev_code = j
                warm[ctx.x[(prev_code, ctx.T_CODE, k)]] = 1
                warm[ctx.used[k]] = 1
                for j in seq_idx[len(pre):]:
                    warm[ctx.g[j]] = 1
            for so in warm_sol.drone_sorties:
                key = (so.drone, tuple(s.order for s in so.stops),
                       so.land_vertiport, so.launch_vertiport, so.launch_time)
                si = sig.get(key)
                if si is not None:
                    warm[ctx.y[si]] = 1
                    warm[ctx.a[(si, ctx.drone_of[so.drone])]] = 1
                    for oid in ctx.sorties[si].orders:
                        warm[ctx.g[ctx.idx[oid]]] = 0
            for oid in warm_sol.unserved:
                if oid in ctx.idx:
                    warm[ctx.z[ctx.idx[oid]]] = 1
            if warm:
                mdl.add_mip_start(mdl.new_solution(var_value_dict=warm))
        except Exception:
            pass

        t0 = perf_counter()
        sol = mdl.solve(log_output=False)
        solve_time = perf_counter() - t0
        if sol is None:
            return None
        try:
            det = mdl.solve_details
            proven = (mdl.get_solve_status() == JobSolveStatus.OPTIMAL_SOLUTION
                      or det.mip_relative_gap <= self.mip_gap)
        except Exception:
            proven = False
        values = {v: float(sol[v]) for v in mdl.iter_variables()}
        self.last_epoch = {"objective": float(sol.objective_value),
                           "proven": bool(proven), "backend": "cplex",
                           "solve_time": solve_time}
        result = self._extract(values, ctx)
        if result is None:
            return None
        return result, proven

    # -- backend: HiGHS (LP export -> highspy -> name-mapped extraction) --
    def _solve_highs(self, mdl, ctx: _EpochCtx):
        import os
        import tempfile
        from time import perf_counter

        import highspy

        with tempfile.TemporaryDirectory() as td:
            lp_path = os.path.join(td, "dagcd_epoch.lp")
            mdl.export_as_lp(path=lp_path)
            h = highspy.Highs()
            try:
                h.setOptionValue("log_to_console", False)
            except Exception:
                pass
            if h.readModel(lp_path) != highspy.HighsStatus.kOk:
                return None
            h.setOptionValue("time_limit", float(self.time_limit))
            h.setOptionValue("mip_rel_gap", float(self.mip_gap))
            if self.threads is not None:
                try:
                    h.setOptionValue("threads", int(self.threads))
                except Exception:
                    pass
            t0 = perf_counter()
            run_status = h.solve()
            solve_time = perf_counter() - t0
            # kWarning covers "solved with warnings" (e.g. time limit hit) — an
            # incumbent may still be present; value_valid below decides.
            if run_status not in (highspy.HighsStatus.kOk,
                                  highspy.HighsStatus.kWarning):
                return None
            hsol = h.getSolution()
            if not hsol.value_valid:
                return None  # no incumbent (should not happen: all-unserved is feasible)
            info = h.getInfo()
            model_status = h.getModelStatus()
            proven = (model_status == highspy.HighsModelStatus.kOptimal
                      or (math.isfinite(info.mip_gap)
                          and info.mip_gap <= self.mip_gap))
            col_names = list(h.getLp().col_names_)
            by_name = dict(zip(col_names, list(hsol.col_value)))
        values = {}
        for var in mdl.iter_variables():
            val = by_name.get(var.name)
            if val is not None:
                values[var] = float(val)
        objective = float(info.objective_function_value) \
            if math.isfinite(info.objective_function_value) else None
        self.last_epoch = {"objective": objective, "proven": bool(proven),
                           "backend": "highs", "solve_time": solve_time}
        result = self._extract(values, ctx)
        if result is None:
            return None
        return result, proven

    # -- shared extraction (backend-independent) --
    def _extract(self, values, ctx: _EpochCtx):
        x, y = ctx.x, ctx.y
        idx, orders_all = ctx.idx, ctx.orders_all
        free_idx, vids = ctx.free_idx, ctx.vids
        onboard = ctx.onboard
        S_CODE, T_CODE = ctx.S_CODE, ctx.T_CODE

        seq_routes: dict[str, list] = {}
        for k, v in enumerate(vids):
            seq = list(onboard[v])
            if onboard[v]:
                nxt = None
                for j in free_idx:
                    if (idx[onboard[v][-1]], j, k) in x \
                            and values[x[(idx[onboard[v][-1]], j, k)]] > 0.5:
                        nxt = j
                if nxt is None:
                    if values[x[(idx[onboard[v][-1]], T_CODE, k)]] <= 0.5:
                        return None  # extraction anomaly -> documented fallback
                    seq_routes[v] = seq
                    continue
            else:
                start_vals = [(j, values[x[(S_CODE[v], j, k)]])
                              for j in free_idx if (S_CODE[v], j, k) in x]
                active = [(j, val) for j, val in start_vals if val > 0.5]
                if not active:
                    seq_routes[v] = seq
                    continue
                if len(active) > 1:
                    return None  # branching anomaly -> fallback
                nxt = active[0][0]
            seen = set(seq)
            while nxt is not None:
                if nxt in seen:  # subtour despite MTZ -> fallback
                    return None
                seen.add(nxt)
                seq.append(orders_all[nxt])
                succ = [jj for jj in free_idx
                        if jj != nxt and (nxt, jj, k) in x
                        and values[x[(nxt, jj, k)]] > 0.5]
                to_term = values[x[(nxt, T_CODE, k)]] > 0.5
                if to_term:
                    if succ:
                        return None  # branching anomaly -> fallback
                    nxt = None
                elif len(succ) == 1:
                    nxt = succ[0]
                else:
                    return None  # dangling route -> fallback
            seq_routes[v] = seq

        chosen = [ctx.sorties[si] for si in ctx.s_index if values[y[si]] > 0.5]
        served = {oid for seq in seq_routes.values() for oid in seq}
        served.update(o for sp in chosen for o in sp.orders)
        unserved = [oid for oid in orders_all if oid not in served]
        return seq_routes, chosen, unserved


class CplexOracle(MipOracle):
    """:class:`MipOracle` with ``backend="cplex"`` — backward-compatible name.

    It keeps the constructor signature
    ``(cost=None, *, max_orders=30, time_limit=120.0, mip_gap=1e-4,
    time_inflation=1.0)`` and its observable behavior (CPLEX solves, insertion
    fallback chain, annotations). New code that wants the open-source backend
    should use ``MipOracle(backend="highs")``.
    """

    def __init__(self, cost: CostParams | None = None, *, max_orders: int = 30,
                 time_limit: float = 120.0, mip_gap: float = 1e-4,
                 time_inflation: float = 1.0):
        super().__init__(cost, backend="cplex", max_orders=max_orders,
                         time_limit=time_limit, mip_gap=mip_gap,
                         time_inflation=time_inflation)

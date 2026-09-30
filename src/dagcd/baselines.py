"""Rolling-horizon baseline controllers (docs/SPECIFICATION.md Sec. 7 baselines).

Two controllers on the standard ``(snap, view, rng) -> Solution`` protocol; every
returned solution passes ``checker.check_solution`` (self-checked, with a
deterministic InsertionOracle fallback so a defect can never raise inside
``run_simulation``'s guard).

* :class:`DynamicALNS` — per-epoch adaptive large neighbourhood search on the
  remaining problem. Destroy operators: random removal / worst-lateness removal /
  route clearing; repair: regret-2 ground insertion + per-drone chained sortie
  greedy; acceptance: improving moves, or simulated-annealing moves with linear
  temperature decay. Default budget 200 iterations per epoch. Deterministic: a
  fresh ``random.Random((seed, tau, n_orders))`` per epoch, so plans do not depend
  on call order. Explicit deferral control: ``max_deferral`` caps how many orders
  may be deferred (declared unserved, retried next epoch) when they would be
  ground-insertable; the repair orders pooled orders narrow-window-first, so when
  deferral happens it prefers wide-window orders.

* :class:`LookaheadReserve` — InsertionOracle skeleton + demand lookahead. The
  arrival-rate curve λ(t) of the generating configuration (default: peak window
  [420, 660], trough ratio 0.3, peak rate 0.6) is *not* part of the epoch
  snapshot, so the curve parameters are constructor arguments (defaults as
  documented there, overridable per experiment). During ``[peak_start - reserve_lead, peak_end]``
  the controller reserves resources for the peak: (a) new order assignments (ground insertions and
  drone sorties) are restricted to tight-window orders (``l - e <= tight_window_max``) — wide-window
  orders are deferred to a later epoch; (b) drone sorties must leave at least
  ``battery_reserve_frac`` of the pack energy after landing (battery floor). Both
  rules are deliberately simple and auditable (see docstrings).

Invariants (asserted by tests/test_baselines.py):
1. checker-clean output (self-check + documented fallback);
2. deterministic per (snapshot, view, configuration) — no dependence on the
   simulator-provided rng or on call order;
3. deferral is retry-semantics only: orders declared unserved remain in the
   snapshot pool at later epochs (simulator forwards released orders).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .checker import CostParams, check_solution
from .instance import TOL, DroneStop, Solution
from .oracle import (InsertionOracle, _assemble, _forward_ground, _leg_blocked,
                     _vehicle_usable)
from .simulator import FleetView

__all__ = ["DynamicALNS", "LookaheadReserve"]


@dataclass
class _PlanSortie:
    """One constructed sortie with its pre-computed epoch cost."""

    drone: str
    launch_vertiport: int
    launch_time: float
    stops: list          # list[DroneStop]
    land_vertiport: int
    land_time: float
    energy_wh: float
    km: float
    late_pen: float
    cost: float          # fixed_sortie + var_drone*km + late_pen
    orders: tuple


def _cheapest_single_sortie(snap, o, did, plan_avail, plan_energy, T_end,
                            vertiports, cost: CostParams):
    """Cheapest single-parcel sortie for order o on drone did given the drone's
    current plan state (available time/vertiport, remaining energy). Prechecks
    mirror the checker's semantics (storm / NFZ at leg departure, payload, L_max,
    C10 horizon, energy vs remaining). Returns _PlanSortie or None."""
    if not vertiports:
        return None
    dr = snap.drones[did]
    if o.weight > dr.payload + TOL:
        return None
    avail_t, p0 = plan_avail[did]
    launch = max(avail_t, o.release)
    if snap.weather_at(launch) == "storm":
        return None
    d1 = snap.dist(p0, o.node)
    t1 = snap.tau_a(p0, o.node, launch, dr.cruise_speed)
    if math.isinf(t1) or _leg_blocked(snap, p0, o.node, launch):
        return None
    arr = launch + t1
    if arr > T_end + TOL:
        return None
    start = max(arr, o.e)
    dep = start + o.service
    if snap.weather_at(dep) == "storm":
        return None
    p1 = min(vertiports, key=lambda p: (snap.dist(o.node, p), p))
    d2 = snap.dist(o.node, p1)
    t2 = snap.tau_a(o.node, p1, dep, dr.cruise_speed)
    if math.isinf(t2) or _leg_blocked(snap, o.node, p1, dep):
        return None
    land = dep + t2
    if land - launch > dr.max_sortie_min + TOL:
        return None
    wx1, wx2 = snap.weather_at(launch), snap.weather_at(dep)
    energy = (snap.energy_alpha + snap.energy_beta * o.weight) * d1 \
        * snap.energy_mu.get(wx1, 1.0) \
        + snap.energy_alpha * d2 * snap.energy_mu.get(wx2, 1.0)
    if energy > plan_energy[did] + TOL:
        return None
    u = max(0.0, start - o.l) * o.lateness_penalty
    km = (d1 + d2) / 1000.0
    return _PlanSortie(drone=did, launch_vertiport=p0, launch_time=launch,
                       stops=[DroneStop(o.id, o.node, arr)],
                       land_vertiport=p1, land_time=land, energy_wh=energy,
                       km=km, late_pen=u,
                       cost=cost.fixed_sortie + cost.var_drone_per_km * km + u,
                       orders=(o.id,))


def _regret2_insert(snap, view, routes, prefix, load, pool, cost: CostParams,
                    depots, T_end, usable, eligible=None):
    """True regret-2 insertion of the pooled orders into ground routes (same
    winner selection as InsertionOracle's ground phase): each round inserts the
    pooled order with the largest (second-best minus best) insertion-cost regret
    at its best position. ``eligible`` optionally filters orders
    (LookaheadReserve's tight-window rule); ineligible orders stay in pool."""
    pending = sorted((snap.orders[o] for o in pool if o in snap.orders
                      and (eligible is None or eligible(snap.orders[o]))),
                     key=lambda o: (o.e, o.release, o.id))
    while pending:
        best = None
        for o in pending:
            opts = []
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
                        continue
                    delta = cost.var_ground_per_km * (prop[4] - cur[4]) \
                        + (prop[3] - cur[3])
                    if not base and not view.vehicles[vid].used:
                        delta += cost.fixed_ground
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
        pool.discard(o.id)


class _GroundState:
    """Shared per-epoch state used by both baselines (routes/sorties/unserved)."""

    def __init__(self, snap, view):
        self.routes = {vid: list(view.vehicles[vid].onboard)
                       for vid in sorted(view.vehicles)}
        self.prefix = {vid: len(ids) for vid, ids in self.routes.items()}
        self.load = {vid: sum(snap.orders[o].weight for o in ids)
                     for vid, ids in self.routes.items()}
        self.sorties: list[_PlanSortie] = []
        self.unserved: set[str] = set()
        self.onboard = {o for ids in self.routes.values() for o in ids}

    def assigned(self):
        out = {o for ids in self.routes.values() for o in ids}
        out.update(o for sp in self.sorties for o in sp.orders)
        return out

    def copy(self):
        cl = _GroundState.__new__(_GroundState)
        cl.routes = {v: list(s) for v, s in self.routes.items()}
        cl.prefix = dict(self.prefix)
        cl.load = dict(self.load)
        cl.sorties = list(self.sorties)
        cl.unserved = set(self.unserved)
        cl.onboard = set(self.onboard)
        return cl


def _state_cost(snap, state, cost: CostParams, depots, T_end):
    """Epoch-model objective of a state (checker-consistent decomposition)."""
    total = 0.0
    for vid in sorted(state.routes):
        seq = state.routes[vid]
        if not seq:
            continue
        total += cost.fixed_ground
        prop = _forward_ground(snap, snap.vehicles[vid], seq, depots, T_end)
        total += cost.var_ground_per_km * prop[4] + prop[3]
    for sp in state.sorties:
        total += sp.cost
    for oid in state.unserved:
        total += snap.orders[oid].unserved_penalty
    return total


def _drone_greedy(snap, view, state, pool, cost: CostParams, depots, T_end,
                  vertiports, energy_floor_frac: float = 0.0,
                  eligible=None):
    """Chained per-drone sortie greedy for the pooled orders; rest stays unserved.
    ``eligible`` optionally filters orders (LookaheadReserve's tight-window rule
    applies to drones as well as ground during the reserve period)."""
    plan_avail = {did: (max(dv.available_at, view.tau), dv.vertiport)
                  for did, dv in view.drones.items() if dv.busy_until is None}
    plan_energy = {did: min(snap.drones[did].battery_wh, dv.energy_wh)
                   for did, dv in view.drones.items() if dv.busy_until is None}
    floor = {did: energy_floor_frac * snap.drones[did].battery_wh
             for did in plan_energy}
    for oid in sorted(pool, key=lambda x: (snap.orders[x].e,
                                           snap.orders[x].release, x)):
        o = snap.orders[oid]
        if eligible is not None and not eligible(o):
            continue
        best = None
        for did in sorted(plan_avail):
            sp = _cheapest_single_sortie(snap, o, did, plan_avail, plan_energy,
                                         T_end, vertiports, cost)
            if sp is None:
                continue
            if plan_energy[did] - sp.energy_wh < floor[did] - TOL:
                continue
            if best is None or sp.cost < best.cost:
                best = sp
        if best is not None:
            state.sorties.append(best)
            plan_avail[best.drone] = (best.land_time, best.land_vertiport)
            plan_energy[best.drone] -= best.energy_wh
            state.unserved.discard(oid)


def _enforce_deferral_cap(snap, view, state, cost: CostParams, depots, T_end,
                          usable):
    """Force-insert deferred orders (cheapest feasible ground position) until the
    deferral cap is met; orders with no feasible insertion stay deferred."""
    cap = state.deferral_cap
    while len(state.unserved) > cap:
        best = None
        for oid in sorted(state.unserved):
            o = snap.orders[oid]
            for vid in sorted(usable):
                base = state.routes[vid]
                if state.load[vid] + o.weight \
                        > snap.vehicles[vid].capacity + TOL:
                    continue
                cur = _forward_ground(snap, snap.vehicles[vid], base, depots,
                                      T_end)
                for pos in range(state.prefix[vid], len(base) + 1):
                    seq = base[:pos] + [oid] + base[pos:]
                    prop = _forward_ground(snap, snap.vehicles[vid], seq,
                                           depots, T_end)
                    if not (prop[5] and prop[6]):
                        continue
                    delta = cost.var_ground_per_km * (prop[4] - cur[4]) \
                        + (prop[3] - cur[3])
                    if not base and not view.vehicles[vid].used:
                        delta += cost.fixed_ground
                    if best is None or delta < best[0]:
                        best = (delta, oid, vid, pos)
        if best is None:
            return  # nothing force-insertable; cap stays exceeded (documented)
        _d, oid, vid, pos = best
        state.routes[vid].insert(pos, oid)
        state.load[vid] += snap.orders[oid].weight
        state.unserved.discard(oid)


def _finish(snap, view, state, cost: CostParams):
    depots = sorted(nid for nid, n in snap.nodes.items() if n.kind == "depot")
    sol = _assemble(snap, view, state.routes, state.sorties,
                    sorted(state.unserved))
    rep = check_solution(snap, sol, cost)
    if rep.violations:
        sol = InsertionOracle(cost)(snap, view, random.Random(0))
    return sol


class DynamicALNS:
    """Rolling adaptive large neighbourhood search (module docstring).

    Parameters: ``iterations`` (default 200), ``destroy_q`` (max orders removed
    per destroy, default 3), ``sa_start_temp`` (initial SA temperature on the J
    scale, default 5.0; linear decay to 0), ``max_deferral`` (default None =
    unlimited; else the number of ground-insertable orders allowed to be deferred
    per epoch), ``seed`` (deterministic per-epoch RNG seed).
    """

    def __init__(self, cost: CostParams | None = None, *, iterations: int = 200,
                 destroy_q: int = 3, sa_start_temp: float = 5.0,
                 max_deferral: int | None = None, seed: int = 7):
        self.cost = cost or CostParams()
        self.iterations = iterations
        self.destroy_q = max(1, destroy_q)
        self.sa_start_temp = sa_start_temp
        self.max_deferral = max_deferral
        self.seed = seed

    def _initial_state(self, snap, view, depots, T_end, vertiports, usable):
        state = _GroundState(snap, view)
        pool = {o.id for o in snap.orders.values() if o.id not in state.onboard}
        _regret2_insert(snap, view, state.routes, state.prefix, state.load,
                        pool, self.cost, depots, T_end, usable)
        state.unserved = set(pool)
        _drone_greedy(snap, view, state, set(pool), self.cost, depots, T_end,
                      vertiports)
        state.unserved = {o for o in pool if o not in state.assigned()
                          and o not in state.onboard}
        return state

    def __call__(self, snap, view: FleetView, rng: random.Random) -> Solution:
        depots = sorted(nid for nid, n in snap.nodes.items() if n.kind == "depot")
        vertiports = sorted(nid for nid, n in snap.nodes.items()
                            if n.kind == "vertiport")
        T_end = snap.horizon
        usable = {vid for vid in snap.vehicles
                  if _vehicle_usable(snap.vehicles[vid])}
        erng = random.Random((self.seed, round(view.tau, 6), len(snap.orders)))
        current = self._initial_state(snap, view, depots, T_end, vertiports,
                                      usable)
        current.deferral_cap = self.max_deferral
        if self.max_deferral is not None:
            _enforce_deferral_cap(snap, view, current, self.cost, depots, T_end,
                                  usable)
        best = current.copy()
        best_cost = _state_cost(snap, current, self.cost, depots, T_end)
        cur_cost = best_cost
        free_all = sorted(o for o in snap.orders if o not in current.onboard)
        if not free_all:
            return _finish(snap, view, current, self.cost)
        for it in range(self.iterations):
            cand = current.copy()
            cand.deferral_cap = self.max_deferral
            op = erng.choice(["random", "worst", "clear"])
            pool = self._destroy(snap, cand, op, erng, depots, T_end)
            _regret2_insert(snap, view, cand.routes, cand.prefix, cand.load,
                            pool, self.cost, depots, T_end, usable)
            cand.unserved = {o for o in pool if o not in cand.assigned()
                             and o not in cand.onboard}
            _drone_greedy(snap, view, cand, set(cand.unserved), self.cost,
                          depots, T_end, vertiports)
            cand.unserved = {o for o in pool if o not in cand.assigned()
                             and o not in cand.onboard}
            if self.max_deferral is not None:
                _enforce_deferral_cap(snap, view, cand, self.cost, depots,
                                      T_end, usable)
            c_cand = _state_cost(snap, cand, self.cost, depots, T_end)
            T = self.sa_start_temp * (1.0 - it / max(1, self.iterations))
            if c_cand <= cur_cost or erng.random() < math.exp(
                    -(c_cand - cur_cost) / max(T, 1e-9)):
                current = cand
                cur_cost = c_cand
            if c_cand < best_cost:
                best = cand.copy()
                best_cost = c_cand
        return _finish(snap, view, best, self.cost)

    def _destroy(self, snap, state, op, erng, depots, T_end):
        """Remove orders from the assignment; return the pooled order ids."""
        removable = []
        for vid in sorted(state.routes):
            for oid in state.routes[vid][state.prefix[vid]:]:
                removable.append(oid)
        removable.extend(o for sp in state.sorties for o in sp.orders)
        removable = sorted(set(removable))
        q = erng.randint(1, self.destroy_q)
        pool: set[str] = set()
        if not removable:
            return pool
        if op == "random":
            pool = set(erng.sample(removable, min(q, len(removable))))
        elif op == "worst":
            late = []
            for vid in sorted(state.routes):
                seq = state.routes[vid]
                if len(seq) > state.prefix[vid]:
                    prop = _forward_ground(snap, snap.vehicles[vid], seq,
                                           depots, T_end)
                    late.extend((max(0.0, max(s.arrival,
                                              snap.orders[s.order].e)
                                       - snap.orders[s.order].l), s.order)
                                for s in prop[0])
            by_oid = {}
            for val, oid in late:
                by_oid[oid] = max(by_oid.get(oid, 0.0), val)
            pool = set(sorted(by_oid, key=lambda x: (-by_oid[x], x))[:q])
        else:  # route clearing: pool the vehicle's free orders (removal + load
            # fix happen in the shared removal step below)
            vids = [v for v in sorted(state.routes)
                    if len(state.routes[v]) > state.prefix[v]]
            if vids:
                v = erng.choice(vids)
                pool = set(state.routes[v][state.prefix[v]:])
        # apply removal from routes/sorties/load
        for vid in sorted(state.routes):
            seq = state.routes[vid]
            kept = seq[:state.prefix[vid]] + [o for o in seq[state.prefix[vid]:]
                                              if o not in pool]
            if len(kept) != len(seq):
                removed = set(seq) - set(kept)
                state.load[vid] -= sum(snap.orders[o].weight for o in removed)
                state.routes[vid] = kept
        state.sorties = [sp for sp in state.sorties
                         if not any(o in pool for o in sp.orders)]
        return pool


class LookaheadReserve:
    """Insertion skeleton + demand-lookahead resource reservation (module docstring).

    The λ(t) curve of the generating configuration is not part of the epoch
    snapshot, so the curve parameters are constructor arguments (defaults:
    peak window [420, 660], reserve lead 60 min). During
    ``[peak_start - reserve_lead, peak_end]``: (a) new ground insertions are
    restricted to tight-window orders (``l - e <= tight_window_max``; wide-window
    orders are deferred, retry semantics); (b) sorties must leave at least
    ``battery_reserve_frac`` of the pack after landing.
    """

    def __init__(self, cost: CostParams | None = None, *,
                 peak_start: float = 420.0, peak_end: float = 660.0,
                 reserve_lead: float = 60.0, tight_window_max: float = 60.0,
                 battery_reserve_frac: float = 0.25):
        self.cost = cost or CostParams()
        self.peak_start = peak_start
        self.peak_end = peak_end
        self.reserve_lead = reserve_lead
        # default 60 = median window width (l - e ~ U(30, 90)) of the default
        # generating distribution. NOTE: a threshold at or below the minimum
        # window width (e.g. 30) turns the reserve period into a defer-all
        # policy. Recalibrate per experiment distribution.
        self.tight_window_max = tight_window_max
        self.battery_reserve_frac = battery_reserve_frac

    def __call__(self, snap, view: FleetView, rng: random.Random) -> Solution:
        depots = sorted(nid for nid, n in snap.nodes.items() if n.kind == "depot")
        vertiports = sorted(nid for nid, n in snap.nodes.items()
                            if n.kind == "vertiport")
        T_end = snap.horizon
        usable = {vid for vid in snap.vehicles
                  if _vehicle_usable(snap.vehicles[vid])}
        in_reserve = (self.peak_start - self.reserve_lead <= view.tau
                      <= self.peak_end)

        def eligible(o):
            if not in_reserve:
                return True
            return (o.l - o.e) <= self.tight_window_max + TOL

        state = _GroundState(snap, view)
        pool = {o.id for o in snap.orders.values() if o.id not in state.onboard}
        _regret2_insert(snap, view, state.routes, state.prefix, state.load,
                        pool, self.cost, depots, T_end, usable,
                        eligible=eligible)
        state.unserved = set(pool)
        _drone_greedy(snap, view, state, set(pool), self.cost, depots, T_end,
                      vertiports,
                      energy_floor_frac=self.battery_reserve_frac if in_reserve
                      else 0.0,
                      eligible=eligible if in_reserve else None)
        state.unserved = {o for o in pool if o not in state.assigned()
                          and o not in state.onboard}
        return _finish(snap, view, state, self.cost)

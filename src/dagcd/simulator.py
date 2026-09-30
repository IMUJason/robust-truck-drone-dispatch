"""Rolling-horizon simulator for DAGCD (online protocol: docs/SPECIFICATION.md Sec. 6).

Event-driven epochs (timer / order batch / weather change / NFZ toggle), controller
re-planning on snapshots, deterministic execution of committed plans, and J_sim
accounting on actual delivery times.
"""

from __future__ import annotations

import math
import random
import time as _time
from bisect import bisect_right
from dataclasses import dataclass, asdict

from .checker import CostParams, check_solution, _leg_hits_nfz
from .generator import Scenario, snapshot_at
from .instance import (Instance, Solution, GroundVehicle, GroundRoute, GroundStop,
                       DroneSortie, DroneStop, TOL)


# ---------------- fleet state ----------------

@dataclass
class VehicleState:
    """Ground-vehicle state maintained internally by the simulator."""
    current_node: int
    available_at: float
    onboard: list[str]      # parcels committed to the vehicle, not yet delivered
    used: bool              # ever carried a route (J_sim fixed cost counts once)
    odometer_km: float


@dataclass
class DroneState:
    """Drone state maintained internally by the simulator."""
    current_vertiport: int
    available_at: float
    busy_until: float | None   # land_time of the in-flight sortie; None = idle
    odometer_km: float
    energy_wh: float           # remaining battery


@dataclass
class FleetState:
    vehicles: dict[str, VehicleState]
    drones: dict[str, DroneState]


@dataclass(frozen=True)
class VehicleView:
    """Read-only ground-vehicle state exposed to controllers."""
    node: int
    available_at: float
    onboard: tuple[str, ...]
    used: bool


@dataclass(frozen=True)
class DroneView:
    """Read-only drone state exposed to controllers."""
    vertiport: int
    available_at: float
    busy_until: float | None
    energy_wh: float


@dataclass(frozen=True)
class FleetView:
    """Read-only fleet snapshot handed to controllers at every epoch."""
    vehicles: dict[str, VehicleView]
    drones: dict[str, DroneView]
    tau: float
    sim_end: float


@dataclass
class EpochRecord:
    """One decision epoch of the simulation."""
    tau: float
    trigger: str          # init | order_batch | weather | nfz | timer
    released: int         # cumulative released orders at tau
    runtime: float        # controller wall time [s]
    violations: int       # hard violations (always 0; any would raise)


@dataclass
class SimReport:
    """Audit report of one simulation run (metrics follow docs/SPECIFICATION.md Sec. 7)."""
    metrics: dict
    epochs: list[EpochRecord]
    deliveries: list[dict]   # {"order_id", "time", "mode"}
    unserved: list[str]
    seed: int
    params: dict


# ---------------- helpers ----------------

_TRIGGER_PRIO = {"order_batch": 0, "weather": 1, "nfz": 2, "timer": 3, "sim_end": 4}


def _next_trigger(sc: Scenario, releases: list[float], m_released: int, tau: float,
                  epoch_delta: float, n_trig: int) -> tuple[float, str]:
    """Earliest next decision epoch and its trigger type (docs/SPECIFICATION.md Sec. 6)."""
    cands: list[tuple[float, str]] = [(sc.sim_end, "sim_end"),
                                      (tau + epoch_delta, "timer")]
    if n_trig > 0:
        idx = m_released + n_trig - 1
        if idx < len(releases):
            cands.append((releases[idx], "order_batch"))
    for w in sc.weather:
        if tau + TOL < w.t_end < sc.sim_end:
            cands.append((w.t_end, "weather"))
            break
    for z in sc.nfzs:
        for t in (z.active_from, z.active_to):
            if t is not None and tau + TOL < t < sc.sim_end:
                cands.append((t, "nfz"))
    t_best, label = min(cands, key=lambda c: (c[0], _TRIGGER_PRIO[c[1]]))
    return t_best, label


def _weather_at(sc: Scenario, t: float) -> str:
    for w in sc.weather:
        if w.t_start - TOL <= t < w.t_end:
            return w.state
    return "clear"


def _dist(sc: Scenario, i: int, j: int) -> float:
    """Euclidean distance [m]; sqrt(dx*dx + dy*dy) for IEEE-754 bit-determinism
    across platforms/CPython versions (math.hypot is libm-dependent — see
    Instance.dist docstring; the two must stay formula-identical)."""
    a, b = sc.nodes[i], sc.nodes[j]
    dx, dy = a.x - b.x, a.y - b.y
    return math.sqrt(dx * dx + dy * dy)


def _sortie_km(sc: Scenario, so: DroneSortie) -> float:
    prev = so.launch_vertiport
    km = 0.0
    for s in so.stops:
        km += _dist(sc, prev, s.node) / 1000.0
        prev = s.node
    km += _dist(sc, prev, so.land_vertiport) / 1000.0
    return km


def _sortie_energy(sc: Scenario, so: DroneSortie) -> float:
    """Re-derive sortie energy with the checker's per-leg weather model (docs/SPECIFICATION.md Sec. 3)."""
    t = so.launch_time
    prev = so.launch_vertiport
    load = sum(sc.orders[s.order].weight for s in so.stops)
    energy = 0.0
    for s in so.stops:
        wx = _weather_at(sc, t)
        energy += (sc.energy_alpha + sc.energy_beta * load) \
            * _dist(sc, prev, s.node) * sc.energy_mu.get(wx, 1.0)
        o = sc.orders[s.order]
        t = max(s.arrival, o.e) + o.service
        load -= o.weight
        prev = s.node
    wx = _weather_at(sc, t)
    energy += (sc.energy_alpha + sc.energy_beta * load) \
        * _dist(sc, prev, so.land_vertiport) * sc.energy_mu.get(wx, 1.0)
    return energy


# ---------------- main loop ----------------

def run_simulation(scenario: Scenario, controller, *, epoch_delta: float = 30.0,
                   n_trig: int = 5, cost: CostParams | None = None,
                   seed: int = 0) -> SimReport:
    """Run the rolling-horizon simulation of `scenario` under `controller`.

    Epoch protocol (docs/SPECIFICATION.md Sec. 6): at tau the controller re-plans the remaining
    problem on a snapshot (delivered orders and parcels on in-flight sorties excluded,
    fleet state forwarded as epoch vehicle starts); committed routes/sorties then
    execute deterministically until the next trigger. Infeasible epoch solutions
    raise RuntimeError with the checker's violation list (foundation guard).
    """
    cost = cost or CostParams()
    sc = scenario
    sim_end = sc.sim_end
    releases = sorted(o.release for o in sc.orders.values())
    rng = random.Random(seed)

    vehicles = {vid: VehicleState(v.start_depot, v.start_time, [], False, 0.0)
                for vid, v in sc.vehicles.items()}
    vertiports = sorted(nid for nid, n in sc.nodes.items() if n.kind == "vertiport")
    drones = {did: DroneState(vertiports[0] if vertiports else 0, 0.0, None, 0.0,
                              d.battery_wh)
              for did, d in sc.drones.items()}
    ground_plans: dict[str, GroundRoute] = {}
    air_plans: list[DroneSortie] = []
    delivered: dict[str, tuple[float, str]] = {}
    deliveries: list[dict] = []
    epochs: list[EpochRecord] = []
    energy_total = 0.0
    n_sorties = 0

    def advance(t_limit: float) -> None:
        """Execute committed plans up to t_limit (math.inf = run everything to completion)."""
        nonlocal energy_total, n_sorties
        for vid in list(ground_plans):
            plan = ground_plans[vid]
            st = vehicles[vid]
            prev, t = st.current_node, st.available_at
            remaining = []
            for stop in plan.stops:
                o = sc.orders[stop.order]
                start = max(stop.arrival, o.e)
                if start <= t_limit + TOL:
                    st.odometer_km += _dist(sc, prev, stop.node) / 1000.0
                    delivered[stop.order] = (start, "ground")
                    deliveries.append({"order_id": stop.order, "time": start,
                                       "mode": "ground"})
                    if stop.order in st.onboard:
                        st.onboard.remove(stop.order)
                    prev, t = stop.node, start + o.service
                else:
                    remaining.append(stop)
            plan.stops = remaining
            st.current_node, st.available_at = prev, t
            if not plan.stops and plan.return_time <= t_limit + TOL:
                st.odometer_km += _dist(sc, prev, plan.return_depot) / 1000.0
                st.current_node = plan.return_depot
                st.available_at = plan.return_time
                del ground_plans[vid]
        for so in list(air_plans):
            if so.land_time <= t_limit + TOL:
                for s in so.stops:
                    o = sc.orders[s.order]
                    start = max(s.arrival, o.e)
                    delivered[s.order] = (start, "drone")
                    deliveries.append({"order_id": s.order, "time": start,
                                       "mode": "drone"})
                dst = drones[so.drone]
                dst.current_vertiport = so.land_vertiport
                dst.available_at = so.land_time
                dst.busy_until = None
                dst.odometer_km += _sortie_km(sc, so)
                energy_total += _sortie_energy(sc, so)
                n_sorties += 1
                air_plans.remove(so)

    tau, trigger = 0.0, "init"
    while tau < sim_end - TOL:
        m = bisect_right(releases, tau)
        excluded = set(delivered)
        for so in air_plans:
            excluded.update(s.order for s in so.stops)
        snap = snapshot_at(sc, tau, sc.params.horizon, excluded)
        # epoch-state vehicles: route starts from current node/availability
        snap.vehicles = {vid: GroundVehicle(vid, sc.vehicles[vid].capacity,
                                            vehicles[vid].current_node,
                                            vehicles[vid].available_at,
                                            sc.vehicles[vid].shift_end)
                         for vid in sc.vehicles}
        view = FleetView(
            vehicles={vid: VehicleView(st.current_node, st.available_at,
                                       tuple(st.onboard), st.used)
                      for vid, st in vehicles.items()},
            drones={did: DroneView(st.current_vertiport, st.available_at,
                                   st.busy_until, st.energy_wh)
                    for did, st in drones.items()},
            tau=tau, sim_end=sim_end)
        t0 = _time.perf_counter()
        sol = controller(snap, view, rng)
        runtime = _time.perf_counter() - t0
        if not isinstance(sol, Solution):
            raise RuntimeError(f"controller returned {type(sol).__name__} at tau={tau}, "
                               "expected Solution")
        rep = check_solution(snap, sol, cost)
        if rep.violations:
            raise RuntimeError(f"infeasible epoch solution at tau={tau}: {rep.violations}")
        epochs.append(EpochRecord(tau, trigger, m, runtime, 0))
        # commit epoch plans
        for r in sol.ground_routes:
            st = vehicles[r.vehicle]
            st.onboard = [s.order for s in r.stops]
            if r.stops:
                st.used = True
                ground_plans[r.vehicle] = r
            else:
                ground_plans.pop(r.vehicle, None)
        for so in sol.drone_sorties:
            if not so.stops:
                continue
            dst = drones[so.drone]
            dst.busy_until = so.land_time
            dst.energy_wh -= _sortie_energy(sc, so)
            air_plans.append(so)
        tau_next, trigger = _next_trigger(sc, releases, m, tau, epoch_delta, n_trig)
        advance(tau_next)
        tau = tau_next

    advance(math.inf)
    unserved = [oid for oid, o in sc.orders.items() if oid not in delivered]

    # ---- final accounting on actual delivery times (docs/SPECIFICATION.md Sec. 5/7) ----
    served = len(delivered)
    n_released = len(sc.orders)
    late_vals = []
    lateness_pen = 0.0
    for oid, (start, _mode) in delivered.items():
        o = sc.orders[oid]
        u = max(0.0, start - o.l)
        late_vals.append(u)
        lateness_pen += u * o.lateness_penalty
    late_sorted = sorted(late_vals)
    n = len(late_sorted)
    p95 = late_sorted[min(n - 1, math.ceil(0.95 * n) - 1)] if n else 0.0
    ground_km = sum(st.odometer_km for st in vehicles.values())
    drone_km = sum(st.odometer_km for st in drones.values())
    vehicles_used = sum(1 for st in vehicles.values() if st.used)
    unserved_pen = sum(sc.orders[oid].unserved_penalty for oid in unserved)
    breakdown = {
        "fixed_ground": cost.fixed_ground * vehicles_used,
        "fixed_sortie": cost.fixed_sortie * n_sorties,
        "var_ground": cost.var_ground_per_km * ground_km,
        "var_drone": cost.var_drone_per_km * drone_km,
        "lateness": lateness_pen,
        "unserved": unserved_pen,
    }
    objective = sum(breakdown.values())
    runtime_avg = sum(e.runtime for e in epochs) / len(epochs) if epochs else 0.0
    metrics = {
        "served": served,
        "served_rate": served / n_released if n_released else 0.0,
        "unserved": len(unserved),
        "late_orders": sum(1 for u in late_vals if u > TOL),
        "avg_lateness": sum(late_vals) / served if served else 0.0,
        "p95_lateness": p95,
        "ground_km": ground_km,
        "drone_km": drone_km,
        "energy_wh": energy_total,
        "vehicles_used": vehicles_used,
        "sorties": n_sorties,
        "cost_per_served": objective / served if served else 0.0,
        "runtime_per_epoch_avg": runtime_avg,
        "cost_breakdown": breakdown,
        "objective": objective,
    }
    return SimReport(metrics=metrics, epochs=epochs, deliveries=deliveries,
                     unserved=unserved, seed=seed, params=asdict(sc.params))


# ---------------- baseline controller ----------------

def _leg_blocked(snap: Instance, i: int, j: int, t: float) -> bool:
    """True if leg i->j crosses an active NFZ at departure time t (checker semantics)."""
    a = (snap.nodes[i].x, snap.nodes[i].y)
    b = (snap.nodes[j].x, snap.nodes[j].y)
    return any(_leg_hits_nfz(a, b, z.polygon) for z in snap.active_nfzs(t))


def _propagate_ground(snap: Instance, veh: GroundVehicle, seq: list[str], T: float,
                      depots: list[int], enforce_horizon: bool):
    """Earliest-feasible ground route for stop sequence `seq` (onboard first); None if
    the shift-end constraint (or, when enforced, the horizon) is violated."""
    if not depots:
        return None
    t = veh.start_time
    prev = veh.start_depot
    stops = []
    km = 0.0
    late = 0.0
    for oid in seq:
        o = snap.orders[oid]
        arr = max(t + snap.tau_g(prev, o.node, t), o.release)
        start = max(arr, o.e)
        if enforce_horizon and start > T + TOL:
            return None
        km += snap.dist(prev, o.node) / 1000.0
        late += max(0.0, start - o.l) * o.lateness_penalty
        stops.append(GroundStop(oid, o.node, arr))
        t = start + o.service
        prev = o.node
    rd = min(depots, key=lambda d: (snap.dist(prev, d), d))
    rt = t + snap.tau_g(prev, rd, t)
    if rt > veh.shift_end + TOL:
        return None
    km += snap.dist(prev, rd) / 1000.0
    return stops, rd, rt, late, km


class GreedyController:
    """Baseline (i) greedy nearest-dispatch, epoch level (docs/SPECIFICATION.md Sec. 7).

    For every unassigned released order (parcels already onboard ground vehicles are
    excluded — they keep their vehicle and stay first on its route), two options are
    compared and the cheaper marginal cost wins:
      (a) single-parcel drone sortie from the nearest vertiport, returning to the
          nearest vertiport to the customer;
      (b) insertion into the route of the nearest ground vehicle with spare capacity.
    Orders feasible by neither option are declared unserved in this epoch's solution
    and may be retried at a later epoch. Claimed arrival times are the earliest
    feasible times under the snapshot's tau_g/tau_a, including release and window
    start (e) waiting.
    """
    GROUND = 0
    DRONE = 1

    def __init__(self, cost: CostParams | None = None):
        self.cost = cost or CostParams()

    def __call__(self, snap: Instance, view: FleetView, rng: random.Random) -> Solution:
        depots = sorted(nid for nid, n in snap.nodes.items() if n.kind == "depot")
        vertiports = sorted(nid for nid, n in snap.nodes.items() if n.kind == "vertiport")
        T = snap.horizon
        onboard = {oid for v in view.vehicles.values() for oid in v.onboard}
        routes: dict[str, list[str]] = {vid: list(v.onboard)
                                        for vid, v in view.vehicles.items()}
        prefix = {vid: len(v.onboard) for vid, v in view.vehicles.items()}
        load = {vid: sum(snap.orders[o].weight for o in ids)
                for vid, ids in routes.items()}
        sorties: list[DroneSortie] = []
        assigned: set[str] = set()
        # per-drone plan state within this epoch (C11 chaining): earliest launch
        # instant, vertiport after the last planned sortie, remaining battery
        self._plan_avail = {did: (max(dv.available_at, view.tau), dv.vertiport)
                            for did, dv in view.drones.items()
                            if dv.busy_until is None}
        self._plan_energy = {did: min(snap.drones[did].battery_wh, dv.energy_wh)
                             for did, dv in view.drones.items()
                             if dv.busy_until is None}

        for o in sorted(snap.orders.values(), key=lambda x: (x.e, x.release, x.id)):
            if o.id in onboard:
                continue
            picks = [p for p in (self._ground_option(snap, view, o, routes, prefix,
                                                     load, T, depots),
                                 self._drone_option(snap, view, o, T, vertiports,
                                                    sorties))
                     if p is not None]
            if not picks:
                continue
            picks.sort(key=lambda p: (p[0], p[1]))
            picks[0][2]()
            assigned.add(o.id)

        ground_routes = []
        for vid in sorted(routes):
            seq = routes[vid]
            if not seq:
                continue
            veh = snap.vehicles[vid]
            built = _propagate_ground(snap, veh, seq, T, depots, enforce_horizon=False)
            stops, rd, rt = built[0], built[1], built[2]
            ground_routes.append(GroundRoute(vid, stops, rd, rt))
        unserved = [o.id for o in snap.orders.values()
                    if o.id not in assigned and o.id not in onboard]
        return Solution(epoch=0, ground_routes=ground_routes,
                        drone_sorties=sorties, unserved=unserved)

    def _drone_option(self, snap: Instance, view: FleetView, o, T: float,
                      vertiports: list[int], sorties: list[DroneSortie]):
        """Single-parcel sortie from/to the nearest vertiport; None if infeasible
        (payload / storm / NFZ / battery / L_max / horizon pre-checks, checker semantics)."""
        if not vertiports:
            return None
        best = None
        for did in sorted(self._plan_avail):
            dr = snap.drones[did]
            if o.weight > dr.payload + TOL:
                continue
            avail_t, cur_vp = self._plan_avail[did]
            p0 = cur_vp
            launch = max(avail_t, o.release)
            if snap.weather_at(launch) == "storm":
                continue
            d1 = snap.dist(p0, o.node)
            t1 = snap.tau_a(p0, o.node, launch, dr.cruise_speed)
            if math.isinf(t1) or _leg_blocked(snap, p0, o.node, launch):
                continue
            arr = launch + t1
            if arr > T + TOL:
                continue
            dep = max(arr, o.e) + o.service
            if snap.weather_at(dep) == "storm":
                continue
            p1 = min(vertiports, key=lambda p: (snap.dist(o.node, p), p))
            d2 = snap.dist(o.node, p1)
            t2 = snap.tau_a(o.node, p1, dep, dr.cruise_speed)
            if math.isinf(t2) or _leg_blocked(snap, o.node, p1, dep):
                continue
            land = dep + t2
            if land - launch > dr.max_sortie_min + TOL:
                continue
            wx1, wx2 = snap.weather_at(launch), snap.weather_at(dep)
            energy = (snap.energy_alpha + snap.energy_beta * o.weight) * d1 \
                * snap.energy_mu.get(wx1, 1.0) \
                + snap.energy_alpha * d2 * snap.energy_mu.get(wx2, 1.0)
            if energy > self._plan_energy[did] + TOL:
                continue
            u = max(0.0, max(arr, o.e) - o.l)
            c = self.cost.fixed_sortie \
                + self.cost.var_drone_per_km * (d1 + d2) / 1000.0 \
                + u * o.lateness_penalty
            if best is None or c < best[0]:
                def apply(did=did, p0=p0, launch=launch, arr=arr, p1=p1, land=land,
                          energy=energy, oid=o.id, node=o.node):
                    sorties.append(DroneSortie(
                        drone=did, launch_vertiport=p0, launch_time=launch,
                        stops=[DroneStop(oid, node, arr)],
                        land_vertiport=p1, land_time=land, energy_wh=energy))
                    self._plan_avail[did] = (land, p1)
                    self._plan_energy[did] -= energy
                best = (c, self.DRONE, apply)
        return best

    def _ground_option(self, snap: Instance, view: FleetView, o, routes, prefix,
                       load, T: float, depots: list[int]):
        """Cheapest insertion into the route of the nearest capacity-available vehicle;
        None if that vehicle cannot absorb the order."""
        for vid in sorted(view.vehicles,
                          key=lambda x: (snap.dist(view.vehicles[x].node, o.node), x)):
            veh = snap.vehicles[vid]
            if load[vid] + o.weight > veh.capacity + TOL:
                continue
            base = routes[vid]
            cur = _propagate_ground(snap, veh, base, T, depots, enforce_horizon=True) \
                if base else None
            cur_late = cur[3] if cur else 0.0
            cur_km = cur[4] if cur else 0.0
            best_pos = None
            for pos in range(prefix[vid], len(base) + 1):
                seq = base[:pos] + [o.id] + base[pos:]
                prop = _propagate_ground(snap, veh, seq, T, depots, enforce_horizon=True)
                if prop is None:
                    continue
                delta = self.cost.var_ground_per_km * (prop[4] - cur_km) \
                    + (prop[3] - cur_late)
                if not base and not view.vehicles[vid].used:
                    delta += self.cost.fixed_ground
                if best_pos is None or delta < best_pos[0]:
                    best_pos = (delta, pos)
            if best_pos is None:
                return None
            delta, pos = best_pos

            def apply(vid=vid, pos=pos, oid=o.id):
                routes[vid].insert(pos, oid)
                load[vid] += snap.orders[oid].weight
            return (delta, self.GROUND, apply)
        return None

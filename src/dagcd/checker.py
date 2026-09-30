"""Feasibility checker for DAGCD (normative rules: docs/SPECIFICATION.md C1-C11).

check_solution(instance, solution) -> CheckReport
  - violations: list of human-readable strings (non-empty => infeasible)
  - objective: J as in docs/SPECIFICATION.md Sec. 5 (valid only if feasible)
  - metrics: served rate, lateness, km, energy, cost breakdown
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .instance import Instance, Solution, TOL


# ---------------- cost model (defaults; override per experiment) ----------------

@dataclass
class CostParams:
    fixed_ground: float = 30.0        # per used ground vehicle
    fixed_sortie: float = 8.0         # per drone sortie
    var_ground_per_km: float = 1.0
    var_drone_per_km: float = 0.3
    BIG: float = 1e9                  # fitness when infeasible


# ---------------- geometry ----------------

def _orient(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p) -> bool:
    """True iff p (collinear with a-b) lies within the bounding box of segment a-b."""
    return (min(a[0], b[0]) - 1e-12 <= p[0] <= max(a[0], b[0]) + 1e-12
            and min(a[1], b[1]) - 1e-12 <= p[1] <= max(a[1], b[1]) + 1e-12)


def _seg_intersect(a, b, c, d) -> bool:
    """Proper segment intersection incl. degenerate cases (bounding-box on-segment
    checks). Collinear-but-disjoint segments must NOT report an intersection
    (the bare orientation-zero shortcut would be a false positive there)."""
    o1, o2 = _orient(a, b, c), _orient(a, b, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b)
    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)):
        return True
    return ((abs(o1) < 1e-12 and _on_segment(a, b, c))
            or (abs(o2) < 1e-12 and _on_segment(a, b, d))
            or (abs(o3) < 1e-12 and _on_segment(c, d, a))
            or (abs(o4) < 1e-12 and _on_segment(c, d, b)))


def _point_in_poly(p, poly) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > p[1]) != (yj > p[1])) and \
           (p[0] < (xj - xi) * (p[1] - yi) / (yj - yi + 1e-300) + xi):
            inside = not inside
        j = i
    return inside


def _leg_hits_nfz(a, b, poly) -> bool:
    if _point_in_poly(a, poly) or _point_in_poly(b, poly):
        return True
    pts = poly + [poly[0]]
    return any(_seg_intersect(a, b, pts[i], pts[i + 1]) for i in range(len(poly)))


# ---------------- report ----------------

@dataclass
class CheckReport:
    violations: list[str] = field(default_factory=list)
    objective: float = 0.0
    feasible: bool = True
    metrics: dict = field(default_factory=dict)

    def add(self, msg: str) -> None:
        self.violations.append(msg)
        self.feasible = False


# ---------------- main entry ----------------

def check_solution(inst: Instance, sol: Solution,
                   cost: CostParams | None = None) -> CheckReport:
    cost = cost or CostParams()
    rep = CheckReport()
    served = {}  # order id -> ("ground"|"drone", location label)

    # ---- C1: coverage + duplicates + unknown ids ----
    for r in sol.ground_routes:
        for s in r.stops:
            if s.order not in inst.orders:
                rep.add(f"unknown order '{s.order}' on ground route {r.vehicle}")
            elif s.order in served:
                rep.add(f"order '{s.order}' served twice ({served[s.order][0]}, ground/{r.vehicle})")
            else:
                served[s.order] = ("ground", r.vehicle)
    for so in sol.drone_sorties:
        for s in so.stops:
            if s.order not in inst.orders:
                rep.add(f"unknown order '{s.order}' on sortie {so.drone}")
            elif s.order in served:
                rep.add(f"order '{s.order}' served twice ({served[s.order][0]}, drone/{so.drone})")
            else:
                served[s.order] = ("drone", so.drone)
    for o in sol.unserved:
        if o in served:
            rep.add(f"order '{o}' both served and listed unserved")
        elif o not in inst.orders:
            rep.add(f"unknown unserved order '{o}'")
        else:
            served[o] = ("unserved", "-")
    for oid, o in inst.orders.items():
        if oid not in served:
            rep.add(f"released order '{oid}' not served and not declared unserved (C1)")

    # ---- ground routes: C2, C3, C4, C10 ----
    ground_km = 0.0
    n_ground_used = 0
    for r in sol.ground_routes:
        veh = inst.vehicles.get(r.vehicle)
        if veh is None:
            rep.add(f"unknown vehicle '{r.vehicle}'")
            continue
        if r.stops:
            n_ground_used += 1
        load = sum(inst.orders[s.order].weight for s in r.stops if s.order in inst.orders)
        if load > veh.capacity + TOL:
            rep.add(f"{r.vehicle} capacity exceeded: {load:.2f} > {veh.capacity} (C2)")
        if veh.start_time > veh.shift_end + TOL:
            rep.add(f"{r.vehicle} shift_end {veh.shift_end} before start {veh.start_time} (C4)")

        t = veh.start_time
        prev = veh.start_depot
        for s in r.stops:
            o = inst.orders.get(s.order)
            if o is None:
                continue
            travel = inst.tau_g(prev, s.node, t)
            earliest = t + travel
            if s.arrival < earliest - TOL:
                rep.add(f"{r.vehicle}/{s.order}: claimed arrival {s.arrival:.2f} earlier than "
                        f"earliest feasible {earliest:.2f} (C3)")
            if s.arrival < o.release - TOL:
                rep.add(f"{r.vehicle}/{s.order}: served before release {o.release} (C3)")
            if s.arrival > inst.horizon + TOL:
                rep.add(f"{r.vehicle}/{s.order}: service start beyond horizon (C10)")
            ground_km += inst.dist(prev, s.node) / 1000.0
            start = max(s.arrival, o.e)
            t = start + o.service
            prev = s.node
        rt = t + inst.tau_g(prev, r.return_depot, t)
        if r.return_time < rt - TOL:
            rep.add(f"{r.vehicle}: claimed return {r.return_time:.2f} earlier than feasible {rt:.2f} (C3)")
        if r.return_time > veh.shift_end + TOL:
            rep.add(f"{r.vehicle}: return {r.return_time:.2f} after shift_end {veh.shift_end} (C4)")
        ground_km += inst.dist(prev, r.return_depot) / 1000.0

    # ---- drone sorties: C5-C8, C10 ----
    drone_km = 0.0
    energy_total = 0.0
    n_sorties = 0
    for so in sol.drone_sorties:
        dr = inst.drones.get(so.drone)
        if dr is None:
            rep.add(f"unknown drone '{so.drone}'")
            continue
        if not so.stops:
            continue
        n_sorties += 1
        if inst.nodes[so.launch_vertiport].kind != "vertiport" or \
           inst.nodes[so.land_vertiport].kind != "vertiport":
            rep.add(f"sortie {so.drone}: launch/land node not a vertiport (C5)")
        if inst.weather_at(so.launch_time) == "storm":
            rep.add(f"sortie {so.drone}: launch during storm (C8)")
        if len(so.stops) > dr.max_parcel:
            rep.add(f"sortie {so.drone}: {len(so.stops)} parcels > M_max {dr.max_parcel} (C5)")
        orders = [inst.orders[s.order] for s in so.stops if s.order in inst.orders]
        if any(o.weight > dr.payload + TOL for o in orders):
            rep.add(f"sortie {so.drone}: parcel heavier than payload {dr.payload} (C5)")
        if sum(o.weight for o in orders) > dr.payload + TOL:
            rep.add(f"sortie {so.drone}: total payload exceeded (C5)")

        t = so.launch_time
        prev = so.launch_vertiport
        load = sum(o.weight for o in orders)
        energy = 0.0
        pa = inst.nodes[prev]
        for s in so.stops:
            o = inst.orders.get(s.order)
            if o is None:
                continue
            a_pt, b_pt = (pa.x, pa.y), (inst.nodes[s.node].x, inst.nodes[s.node].y)
            travel = inst.tau_a(prev, s.node, t, dr.cruise_speed)
            if travel == float("inf"):
                rep.add(f"sortie {so.drone} leg {prev}->{s.node}: storm en route (C8)")
                travel = inst.dist(prev, s.node) / max(dr.cruise_speed, 1e-9)
            wx = inst.weather_at(t)
            energy += (inst.energy_alpha + inst.energy_beta * load) \
                * inst.dist(prev, s.node) * inst.energy_mu.get(wx, 1.0)
            for z in inst.active_nfzs(t):
                if _leg_hits_nfz(a_pt, b_pt, z.polygon):
                    rep.add(f"sortie {so.drone} leg {prev}->{s.node} crosses NFZ '{z.id}' (C8)")
            earliest = t + travel
            if s.arrival < earliest - TOL:
                rep.add(f"sortie {so.drone}/{s.order}: claimed arrival {s.arrival:.2f} earlier "
                        f"than feasible {earliest:.2f} (C8)")
            if s.arrival < o.release - TOL:
                rep.add(f"sortie {so.drone}/{s.order}: served before release (C8)")
            if s.arrival > inst.horizon + TOL:
                rep.add(f"sortie {so.drone}/{s.order}: service beyond horizon (C10)")
            drone_km += inst.dist(prev, s.node) / 1000.0
            t = max(s.arrival, o.e) + o.service
            load -= o.weight
            prev = s.node
            pa = inst.nodes[prev]
        # landing leg
        wx = inst.weather_at(t)
        energy += (inst.energy_alpha + inst.energy_beta * load) \
            * inst.dist(prev, so.land_vertiport) * inst.energy_mu.get(wx, 1.0)
        a_pt = (inst.nodes[prev].x, inst.nodes[prev].y)
        b_pt = (inst.nodes[so.land_vertiport].x, inst.nodes[so.land_vertiport].y)
        for z in inst.active_nfzs(t):
            if _leg_hits_nfz(a_pt, b_pt, z.polygon):
                rep.add(f"sortie {so.drone} landing leg crosses NFZ '{z.id}' (C8)")
        land_earliest = t + inst.tau_a(prev, so.land_vertiport, t, dr.cruise_speed)
        if so.land_time < land_earliest - TOL:
            rep.add(f"sortie {so.drone}: claimed land {so.land_time:.2f} earlier than feasible "
                    f"{land_earliest:.2f} (C8)")
        drone_km += inst.dist(prev, so.land_vertiport) / 1000.0
        energy_total += energy
        if energy > dr.battery_wh + TOL:
            rep.add(f"sortie {so.drone}: energy {energy:.1f}Wh > battery {dr.battery_wh}Wh (C6)")
        if so.energy_wh > 0 and abs(so.energy_wh - energy) > max(1.0, 0.01 * energy):
            rep.add(f"sortie {so.drone}: claimed energy {so.energy_wh:.1f} != derived {energy:.1f} (C6)")
        if so.land_time - so.launch_time > dr.max_sortie_min + TOL:
            rep.add(f"sortie {so.drone}: duration {so.land_time - so.launch_time:.1f} > L_max "
                    f"{dr.max_sortie_min} (C7)")

    # ---- C11: per-drone sortie timeline (no overlap; chaining after landing allowed) ----
    by_drone: dict[str, list[tuple[float, float]]] = {}
    for so in sol.drone_sorties:
        if so.stops and so.drone in inst.drones:
            by_drone.setdefault(so.drone, []).append((so.launch_time, so.land_time))
    for did, ivals in by_drone.items():
        ivals.sort()
        for (_l0, t0), (l1, _t1) in zip(ivals, ivals[1:]):
            if l1 < t0 - TOL:
                rep.add(f"drone '{did}': sortie launched at {l1:.2f} overlaps previous "
                        f"sortie airborne until {t0:.2f} (C11)")

    # ---- C9: lateness + objective ----
    lateness_sum = 0.0
    late_orders = 0
    served_count = 0
    for oid, (mode, _) in served.items():
        if mode == "unserved":
            continue
        o = inst.orders[oid]
        claimed = None
        for r in sol.ground_routes:
            for s in r.stops:
                if s.order == oid:
                    claimed = s.arrival
        for so in sol.drone_sorties:
            for s in so.stops:
                if s.order == oid:
                    claimed = s.arrival
        if claimed is None:
            continue
        served_count += 1
        start = max(claimed, o.e)
        u = max(0.0, start - o.l)
        lateness_sum += u
        if u > TOL:
            late_orders += 1

    n_released = len(inst.orders)
    unserved_pen = sum(inst.orders[o].unserved_penalty for o in sol.unserved
                       if o in inst.orders)
    lateness_pen = 0.0
    for oid, (mode, _) in served.items():
        if mode == "unserved":
            continue
        for r in sol.ground_routes:
            for s in r.stops:
                if s.order == oid:
                    o = inst.orders[oid]
                    lateness_pen += max(0.0, max(s.arrival, o.e) - o.l) * o.lateness_penalty
        for so in sol.drone_sorties:
            for s in so.stops:
                if s.order == oid:
                    o = inst.orders[oid]
                    lateness_pen += max(0.0, max(s.arrival, o.e) - o.l) * o.lateness_penalty

    obj = (cost.fixed_ground * n_ground_used
           + cost.fixed_sortie * n_sorties
           + cost.var_ground_per_km * ground_km
           + cost.var_drone_per_km * drone_km
           + lateness_pen + unserved_pen)

    rep.objective = obj if rep.feasible else cost.BIG
    rep.metrics = {
        "served": served_count,
        "served_rate": served_count / n_released if n_released else 0.0,
        "unserved": len(sol.unserved),
        "late_orders": late_orders,
        "avg_lateness": lateness_sum / max(served_count, 1),
        "ground_km": ground_km,
        "drone_km": drone_km,
        "energy_wh": energy_total,
        "ground_vehicles_used": n_ground_used,
        "sorties": n_sorties,
        "cost_per_served": obj / max(served_count, 1),
        "cost_breakdown": {
            "fixed_ground": cost.fixed_ground * n_ground_used,
            "fixed_sortie": cost.fixed_sortie * n_sorties,
            "var_ground": cost.var_ground_per_km * ground_km,
            "var_drone": cost.var_drone_per_km * drone_km,
            "lateness": lateness_pen,
            "unserved": unserved_pen,
        },
    }
    return rep

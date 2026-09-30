"""Offspring: cheapest-insertion across ALL ground vehicles (no early return
on the nearest vehicle) + earliest-deadline-first order consideration."""


def create_controller(cost=None):
    cost = cost or CostParams()
    GROUND, DRONE = 0, 1

    def controller(snap, view, rng):
        depots = sorted(n for n, nd in snap.nodes.items() if nd.kind == "depot")
        vertiports = sorted(n for n, nd in snap.nodes.items()
                            if nd.kind == "vertiport")
        T = snap.horizon
        onboard = {oid for v in view.vehicles.values() for oid in v.onboard}
        routes = {vid: list(v.onboard) for vid, v in view.vehicles.items()}
        prefix = {vid: len(v.onboard) for vid, v in view.vehicles.items()}
        load = {vid: sum(snap.orders[o].weight for o in ids)
                for vid, ids in routes.items()}
        sorties = []
        assigned = set()
        cache = {}
        plan_avail = {did: (max(dv.available_at, view.tau), dv.vertiport)
                      for did, dv in view.drones.items()
                      if dv.busy_until is None}
        plan_energy = {did: min(snap.drones[did].battery_wh, dv.energy_wh)
                       for did, dv in view.drones.items()
                       if dv.busy_until is None}

        def drone_option(o):
            if not vertiports:
                return None
            best = None
            for did in sorted(plan_avail):
                dr = snap.drones[did]
                if o.weight > dr.payload + TOL:
                    continue
                avail_t, cur_vp = plan_avail[did]
                launch = max(avail_t, o.release)
                if snap.weather_at(launch) == "storm":
                    continue
                d1 = snap.dist(cur_vp, o.node)
                t1 = snap.tau_a(cur_vp, o.node, launch, dr.cruise_speed)
                if math.isinf(t1) or _leg_blocked(snap, cur_vp, o.node, launch):
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
                wx1 = snap.weather_at(launch)
                wx2 = snap.weather_at(dep)
                energy = ((snap.energy_alpha + snap.energy_beta * o.weight)
                          * d1 * snap.energy_mu.get(wx1, 1.0)
                          + snap.energy_alpha * d2
                          * snap.energy_mu.get(wx2, 1.0))
                if energy > plan_energy[did] + TOL:
                    continue
                u = max(0.0, max(arr, o.e) - o.l)
                c = (cost.fixed_sortie
                     + cost.var_drone_per_km * (d1 + d2) / 1000.0
                     + u * o.lateness_penalty)
                if best is None or c < best[0]:
                    def apply(did=did, p0=cur_vp, launch=launch, arr=arr,
                              p1=p1, land=land, energy=energy, oid=o.id,
                              node=o.node):
                        sorties.append(DroneSortie(
                            drone=did, launch_vertiport=p0,
                            launch_time=launch,
                            stops=[DroneStop(oid, node, arr)],
                            land_vertiport=p1, land_time=land,
                            energy_wh=energy))
                        plan_avail[did] = (land, p1)
                        plan_energy[did] -= energy
                    best = (c, DRONE, apply)
            return best

        def ground_option(o):
            best = None
            for vid in sorted(view.vehicles):
                veh = snap.vehicles[vid]
                if load[vid] + o.weight > veh.capacity + TOL:
                    continue
                base = routes[vid]
                key = tuple(base)
                c = cache.get(vid)
                if c is None or c[0] != key:
                    c = (key, _propagate_ground(snap, veh, base, T, depots,
                                                enforce_horizon=True))
                    cache[vid] = c
                cur = c[1]
                cur_late = cur[3] if cur else 0.0
                cur_km = cur[4] if cur else 0.0
                best_pos = None
                for pos in range(prefix[vid], len(base) + 1):
                    seq = base[:pos] + [o.id] + base[pos:]
                    prop = _propagate_ground(snap, veh, seq, T, depots,
                                             enforce_horizon=True)
                    if prop is None:
                        continue
                    delta = (cost.var_ground_per_km * (prop[4] - cur_km)
                             + (prop[2] - cur_late))
                    if not base and not view.vehicles[vid].used:
                        delta += cost.fixed_ground
                    if best_pos is None or delta < best_pos[0]:
                        best_pos = (delta, pos)
                if best_pos is None:
                    continue
                if best is None or best_pos[0] < best[0]:
                    best = (best_pos[0], vid, best_pos[1])
            if best is None:
                return None
            delta, vid, pos = best

            def apply(vid=vid, pos=pos, oid=o.id):
                routes[vid].insert(pos, oid)
                load[vid] += snap.orders[oid].weight
                cache.pop(vid, None)
            return (delta, GROUND, apply)

        for o in sorted(snap.orders.values(),
                        key=lambda x: (x.l, x.release, x.id)):
            if o.id in onboard:
                continue
            picks = [p for p in (ground_option(o), drone_option(o))
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
            built = _propagate_ground(snap, veh, seq, T, depots,
                                      enforce_horizon=False)
            stops, rd, rt = built[0], built[1], built[2]
            ground_routes.append(GroundRoute(vid, stops, rd, rt))
        unserved = [o.id for o in snap.orders.values()
                    if o.id not in assigned and o.id not in onboard]
        return Solution(epoch=0, ground_routes=ground_routes,
                        drone_sorties=sorties, unserved=unserved)

    return controller
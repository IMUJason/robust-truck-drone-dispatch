"""Unit tests for the DAGCD epoch re-optimization oracle (docs/SPECIFICATION.md Sec. 5.1/7).

Coverage:
  * hand-verified brute-force optimality on a 3-order / 1-vehicle / 1-drone snapshot
    (constant congestion => the MILP's fixed-tau model equals the true model, so the
    checker's objective of the oracle solution must match the exhaustive optimum of
    the same candidate space to <1e-6 relative);
  * checker safety: 20 random snapshots, both oracles must return solutions with an
    empty violation list, and claimed times must equal the earliest feasible times
    re-derived from the true tau_g/tau_a (Invariant 2 of oracle.py);
  * quality: on 5 medium scenarios (sim_duration=240, lam_peak=0.5) the oracle's
    J_sim must not exceed GreedyController's in >= 4/5 runs (inversions, if any,
    are root-caused in this docstring — never silently relaxed);
  * robustness: degenerate snapshots (broken vehicle, no drones, empty orders) must
    not raise and must pass the checker;
  * onboard parcels keep their vehicle and stay first on its route.

CPLEX-dependent tests skip when docplex is not importable; they are exercised
for real in any environment with a working CPLEX installation.
"""

import importlib.util
import math
import os
import random
import sys
import unittest
from itertools import combinations, permutations

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dagcd import (CostParams, InstanceParams, Node, Order, GroundVehicle, Drone,
                   WeatherEvent, Instance, Solution, GroundRoute, GroundStop,
                   DroneSortie, DroneStop, check_solution, sample_scenario,
                   snapshot_at, run_simulation, GreedyController,
                   InsertionOracle, CplexOracle)
from dagcd.oracle import enumerate_sorties, SortiePlan, _forward_ground
from dagcd.simulator import FleetView, VehicleView, DroneView

HAS_DOCPLEX = importlib.util.find_spec("docplex") is not None
requires_docplex = unittest.skipUnless(HAS_DOCPLEX,
                                       "docplex/cplex not available in this interpreter")


def tiny_snapshot():
    """3 orders, 1 vehicle, 1 drone, constant congestion/weather (true model ==
    fixed-tau model), all released at t=0."""
    nodes = {0: Node(0, 0, 0, "depot"), 1: Node(1, 800, 0, "vertiport"),
             2: Node(2, 2000, 0, "customer"), 3: Node(3, 0, 2000, "customer"),
             4: Node(4, 1505, 1490, "customer")}
    orders = {
        "o1": Order("o1", 2, 0.0, 1.0, 1.0, 0.0, 60.0),
        "o2": Order("o2", 3, 0.0, 2.0, 1.0, 0.0, 50.0),
        "o3": Order("o3", 4, 0.0, 1.5, 2.0, 0.0, 45.0),
    }
    snap = Instance(
        nodes=nodes, orders=orders,
        vehicles={"gv0": GroundVehicle("gv0", 10.0, 0, 0.0, 240.0)},
        drones={"da0": Drone("da0", 5.0, 500.0, 600.0, 60.0, 3)},
        nfzs=[], weather=[WeatherEvent(0.0, 240.0, "clear")], horizon=240.0,
        congestion=[(0.0, 240.0, 1.0)])
    view = FleetView(vehicles={"gv0": VehicleView(0, 0.0, (), False)},
                     drones={"da0": DroneView(1, 0.0, None, 500.0)},
                     tau=0.0, sim_end=240.0)
    return snap, view


def random_snapshot(seed):
    """Small random snapshot at tau=30 with the original (idle) fleet view."""
    params = InstanceParams(sim_duration=60.0, lam_peak=0.6, n_clusters=2,
                            n_vehicles=2, n_drones=2, n_vertiports=2,
                            peak_start=10.0, peak_end=50.0, nfz_density=1.0,
                            mean_event_min=15.0)
    sc = sample_scenario(params, seed=seed)
    tau = 30.0
    snap = snapshot_at(sc, tau, sc.params.horizon)
    vertiports = sorted(n for n, nd in sc.nodes.items() if nd.kind == "vertiport")
    view = FleetView(
        vehicles={vid: VehicleView(v.start_depot, v.start_time, (), False)
                  for vid, v in snap.vehicles.items()},
        drones={did: DroneView(vertiports[0] if vertiports else 0, 0.0, None,
                               d.battery_wh)
                for did, d in snap.drones.items()},
        tau=tau, sim_end=sc.sim_end)
    return snap, view


def assert_earliest_claims(testcase, snap, sol):
    """Invariant 2: claimed times equal the earliest feasible times under the true
    time-dependent tau_g/tau_a (re-derived independently of the oracle's code path)."""
    TOL = 1e-6
    for r in sol.ground_routes:
        veh = snap.vehicles[r.vehicle]
        t, prev = veh.start_time, veh.start_depot
        for s in r.stops:
            o = snap.orders[s.order]
            earliest = max(t + snap.tau_g(prev, o.node, t), o.release)
            testcase.assertAlmostEqual(s.arrival, earliest, delta=TOL,
                                       msg=f"{r.vehicle}/{s.order} arrival")
            t = max(s.arrival, o.e) + o.service
            prev = s.node
    for so in sol.drone_sorties:
        dr = snap.drones[so.drone]
        t, prev = so.launch_time, so.launch_vertiport
        for s in so.stops:
            o = snap.orders[s.order]
            earliest = t + snap.tau_a(prev, s.node, t, dr.cruise_speed)
            testcase.assertAlmostEqual(s.arrival, earliest, delta=TOL,
                                       msg=f"{so.drone}/{s.order} arrival")
            t = max(s.arrival, o.e) + o.service
            prev = s.node
        land_earliest = t + snap.tau_a(prev, so.land_vertiport, t, dr.cruise_speed)
        testcase.assertAlmostEqual(so.land_time, land_earliest, delta=TOL,
                                   msg=f"{so.drone} land time")


def _build_solution(snap, perm, chosen, unserved):
    depots = sorted(nid for nid, n in snap.nodes.items() if n.kind == "depot")
    routes = []
    if perm:
        veh = snap.vehicles["gv0"]
        stops, rd, rt, _l, _k, _a, _b = _forward_ground(snap, veh, list(perm),
                                                        depots, snap.horizon)
        routes.append(GroundRoute("gv0", stops, rd, rt))
    return Solution(epoch=0, ground_routes=routes,
                    drone_sorties=[DroneSortie(sp.drone, sp.launch_vertiport,
                                               sp.launch_time, list(sp.stops),
                                               sp.land_vertiport, sp.land_time,
                                               sp.energy_wh)
                                   for sp in chosen],
                    unserved=sorted(unserved))


def brute_force_optimum(snap, view, cost):
    """Exhaustive optimum over the oracle's own candidate space: every ground
    subset/permutation, every disjoint cover of the rest by enumerated sorties or
    unserved. Constant congestion => checker objective == fixed-tau model."""
    orders = sorted(snap.orders)
    sorties = enumerate_sorties(snap, view, cost)
    best = math.inf
    for r in range(len(orders) + 1):
        for gsub in combinations(orders, r):
            rest = [o for o in orders if o not in gsub]
            for perm in permutations(gsub):

                def cover(i, chosen, uns):
                    nonlocal best
                    if i == len(rest):
                        sol = _build_solution(snap, perm, chosen, uns)
                        rep = check_solution(snap, sol, cost)
                        if not rep.violations and rep.objective < best:
                            best = rep.objective
                        return
                    oid = rest[i]
                    if any(oid in c.orders for c in chosen):
                        cover(i + 1, chosen, uns)  # already covered (multi-parcel)
                        return
                    cover(i + 1, chosen, uns + [oid])          # unserved
                    for si, sp in enumerate(sorties):
                        if oid not in sp.orders:
                            continue
                        if any(o not in rest for o in sp.orders):
                            continue
                        if any(o in uns for o in sp.orders):
                            continue
                        if any(o in c.orders for o in sp.orders
                               for c in chosen):
                            continue
                        if any(sp.drone == c.drone for c in chosen):
                            continue  # one sortie per drone (time-overlap clique)
                        cover(i + 1, chosen + [sp], uns)

                cover(0, [], [])
    return best


class TestOracleBruteForce(unittest.TestCase):
    @requires_docplex
    def test_cplex_matches_exhaustive_optimum(self):
        snap, view = tiny_snapshot()
        cost = CostParams()
        oracle = CplexOracle(cost, mip_gap=1e-9, time_limit=60.0)
        sol = oracle(snap, view, random.Random(0))
        self.assertIsNone(sol.oracle_note)  # no fallback: MILP solved directly
        rep = check_solution(snap, sol, cost)
        self.assertEqual(rep.violations, [])
        got = rep.objective
        want = brute_force_optimum(snap, view, cost)
        self.assertFalse(math.isinf(want))
        self.assertLessEqual(abs(got - want), 1e-6 * max(1.0, abs(want)),
                             msg=f"oracle objective {got} != brute force {want}")
        # the tiny instance is cheapest all-ground: fixed 30 + 7.159 km (hand-checked
        # best permutation o1->o3->o2 = 2000+1589.064+1570.136+2000 metres)
        self.assertAlmostEqual(want, 37.15913584443457, delta=1e-6)
        self.assertEqual(len(sol.drone_sorties), 0)
        self.assertEqual([s.order for s in sol.ground_routes[0].stops],
                         ["o1", "o3", "o2"])

    @requires_docplex
    def test_cplex_drone_option_selected_when_ground_infeasible(self):
        # identical fleet, but the vehicle's shift ends too early for the remote
        # customers (depot round trip = 24 min > shift_end = 20) => the drone must
        # carry both parcels on one 2-parcel sortie. Hand-checked: sortie legs
        # 5200.0 + 4569.464 + 5403.702 m => cost 8 + 0.3*15.173 = 12.552, well
        # below two unserved penalties (100) and below any ground route (C4).
        snap, view = tiny_snapshot()
        snap.vehicles["gv0"] = GroundVehicle("gv0", 10.0, 0, 0.0, 20.0)
        snap.nodes[2] = Node(2, 6000.0, 0.0, "customer")
        snap.nodes[3] = Node(3, 4200.0, 4200.0, "customer")
        snap.orders["o1"].node = 2
        snap.orders["o2"].node = 3
        del snap.orders["o3"]  # keep the case minimal and hand-checkable
        view = FleetView(vehicles={"gv0": VehicleView(0, 0.0, (), False)},
                         drones={"da0": DroneView(1, 0.0, None, 500.0)},
                         tau=0.0, sim_end=240.0)
        cost = CostParams()
        oracle = CplexOracle(cost, mip_gap=1e-9, time_limit=60.0)
        sol = oracle(snap, view, random.Random(0))
        self.assertIsNone(sol.oracle_note)
        rep = check_solution(snap, sol, cost)
        self.assertEqual(rep.violations, [])
        got = rep.objective
        want = brute_force_optimum(snap, view, cost)
        self.assertLessEqual(abs(got - want), 1e-6 * max(1.0, abs(want)),
                             msg=f"oracle {got} != brute force {want}")
        # hand-checked sortie: (5200.0 + 4569.464 + 5403.702) m, cost 12.55195
        self.assertAlmostEqual(want, 12.551949889388258, delta=1e-6)
        self.assertEqual(sol.ground_routes, [])  # ground shift-infeasible
        self.assertEqual(len(sol.drone_sorties), 1)  # exactly one sortie flies
        self.assertEqual(len(sol.drone_sorties[0].stops), 2)  # 2-parcel sortie
        self.assertEqual(sol.unserved, [])


class TestOracleSafety(unittest.TestCase):
    @requires_docplex
    def test_checker_safety_20_random_snapshots(self):
        cost = CostParams()
        cplex = CplexOracle(cost, time_limit=8.0)  # bounded solve per snapshot
        for seed in range(20):
            snap, view = random_snapshot(1000 + seed)
            with self.subTest(seed=seed):
                for ctl in (cplex, InsertionOracle(cost)):
                    sol = ctl(snap, view, random.Random(seed))
                    rep = check_solution(snap, sol, cost)
                    self.assertEqual(rep.violations, [])
                    assert_earliest_claims(self, snap, sol)
                # determinism (Invariant 5): identical inputs -> identical plan.
                # InsertionOracle is a pure function (exact equality). CplexOracle
                # is deterministic whenever CPLEX proves optimality; at a binding
                # time limit the incumbent can vary run-to-run (wall-time-dependent
                # search), so exact equality is only asserted on a snapshot that
                # solves to proven optimality.
                i1 = InsertionOracle(cost)(snap, view, random.Random(seed))
                i2 = InsertionOracle(cost)(snap, view, random.Random(seed))
                self.assertEqual(
                    [(r.vehicle, [(s.order, s.arrival) for s in r.stops])
                     for r in i1.ground_routes],
                    [(r.vehicle, [(s.order, s.arrival) for s in r.stops])
                     for r in i2.ground_routes])
                self.assertEqual(sorted(i1.unserved), sorted(i2.unserved))
        tsnap, tview = tiny_snapshot()
        exact = CplexOracle(cost, mip_gap=1e-9, time_limit=60.0)
        c1 = exact(tsnap, tview, random.Random(0))
        c2 = exact(tsnap, tview, random.Random(0))
        self.assertEqual([[(s.order, s.arrival) for s in r.stops]
                          for r in c1.ground_routes],
                         [[(s.order, s.arrival) for s in r.stops]
                          for r in c2.ground_routes])
        self.assertEqual(c1.unserved, c2.unserved)

    def test_insertion_time_consistency_tiny(self):
        # pure-python path: no docplex needed
        snap, view = tiny_snapshot()
        sol = InsertionOracle()(snap, view, random.Random(0))
        self.assertEqual(check_solution(snap, sol, CostParams()).violations, [])
        assert_earliest_claims(self, snap, sol)

    @requires_docplex
    def test_onboard_parcels_stay_on_their_vehicle(self):
        snap, view = tiny_snapshot()
        # o1 is onboard gv0, which is mid-route at node 2 available at t=10
        snap.vehicles["gv0"] = GroundVehicle("gv0", 10.0, 2, 10.0, 240.0)
        view = FleetView(vehicles={"gv0": VehicleView(2, 10.0, ("o1",), True)},
                         drones={"da0": DroneView(1, 10.0, None, 500.0)},
                         tau=10.0, sim_end=240.0)
        sol = CplexOracle(CostParams(), mip_gap=1e-9)(snap, view, random.Random(0))
        rep = check_solution(snap, sol, CostParams())
        self.assertEqual(rep.violations, [])
        self.assertIsNone(sol.oracle_note)  # MILP itself must handle the prefix
        route = next(r for r in sol.ground_routes if r.vehicle == "gv0")
        self.assertEqual(route.stops[0].order, "o1")  # onboard keeps first place
        assert_earliest_claims(self, snap, sol)

    def test_degenerate_snapshots_do_not_raise(self):
        # broken vehicle (shift ends before start), no drones, 2 released orders
        nodes = {0: Node(0, 0, 0, "depot"), 2: Node(2, 1000, 0, "customer"),
                 3: Node(3, 0, 1000, "customer")}
        orders = {"o1": Order("o1", 2, 0.0, 1.0, 1.0, 0.0, 90.0),
                  "o2": Order("o2", 3, 0.0, 1.0, 1.0, 0.0, 90.0)}
        snap = Instance(nodes=nodes, orders=orders,
                        vehicles={"gv0": GroundVehicle("gv0", 10.0, 0, 300.0, 240.0)},
                        drones={}, nfzs=[],
                        weather=[WeatherEvent(0.0, 300.0, "clear")], horizon=240.0,
                        congestion=[(0.0, 300.0, 1.0)])
        view = FleetView(vehicles={"gv0": VehicleView(0, 300.0, (), False)},
                         drones={}, tau=0.0, sim_end=300.0)
        for ctl in (InsertionOracle(),):
            sol = ctl(snap, view, random.Random(0))  # must not raise
            rep = check_solution(snap, sol, CostParams())
            self.assertEqual(rep.violations, [])
            self.assertEqual(sol.unserved, ["o1", "o2"])
        # empty snapshot
        snap.orders = {}
        sol = InsertionOracle()(snap, view, random.Random(0))
        self.assertEqual(check_solution(snap, sol, CostParams()).violations, [])
        self.assertEqual(sol.unserved, [])

    @requires_docplex
    def test_cplex_degenerate_and_empty(self):
        nodes = {0: Node(0, 0, 0, "depot"), 2: Node(2, 1000, 0, "customer")}
        orders = {"o1": Order("o1", 2, 0.0, 1.0, 1.0, 0.0, 90.0)}
        snap = Instance(nodes=nodes, orders=orders,
                        vehicles={"gv0": GroundVehicle("gv0", 10.0, 0, 300.0, 240.0)},
                        drones={}, nfzs=[],
                        weather=[WeatherEvent(0.0, 300.0, "clear")], horizon=240.0,
                        congestion=[(0.0, 300.0, 1.0)])
        view = FleetView(vehicles={"gv0": VehicleView(0, 300.0, (), False)},
                         drones={}, tau=0.0, sim_end=300.0)
        sol = CplexOracle()(snap, view, random.Random(0))
        self.assertEqual(check_solution(snap, sol, CostParams()).violations, [])
        self.assertEqual(sol.unserved, ["o1"])
        snap.orders = {}
        sol = CplexOracle()(snap, view, random.Random(0))
        self.assertEqual(check_solution(snap, sol, CostParams()).violations, [])
        self.assertEqual(sol.unserved, [])

    @requires_docplex
    def test_size_cap_falls_back_with_annotation(self):
        snap, view = random_snapshot(7)
        while len(snap.orders) <= 3:
            snap, view = random_snapshot(7 + len(snap.orders))
        oracle = CplexOracle(CostParams(), max_orders=2)
        sol = oracle(snap, view, random.Random(0))
        self.assertIsNotNone(sol.oracle_note)
        self.assertIn("max_orders", sol.oracle_note)
        # fallback plan still passes the checker
        self.assertEqual(check_solution(snap, sol, CostParams()).violations, [])


class TestOracleQuality(unittest.TestCase):
    @requires_docplex
    def test_oracle_beats_greedy_5_scenarios(self):
        """Rolling J_sim of CplexOracle must be <= GreedyController's in >= 4/5
        medium scenarios (sim_duration=240, lam_peak=0.5; fixed seeds 100..104).

        Why C11 exists: a greedy epoch planner that holds a read-only FleetView
        fixed while planning can double/triple-book one drone with *overlapping*
        sorties inside a single epoch. Physically impossible, invisible to
        checker rules C1-C10, and flattering to greedy's J_sim. The hard
        constraint C11 (per-drone sortie timeline, chaining after landing
        allowed) closes this in checker.py (see docs/SPECIFICATION.md), and
        GreedyController chains sorties on a per-drone plan state (position +
        battery). Residual oracle-side approximations: fixed-tau ground
        linearization with post-hoc C4/C10 repair; the max_orders cap
        (InsertionOracle fallback); return-leg shift bound; one sortie per
        drone per epoch (clique). Assertion kept at >= 4/5 because CPLEX
        timeout incumbents vary run-to-run.
        """
        cost = CostParams()
        results = []
        for seed in range(5):
            params = InstanceParams(sim_duration=240.0, lam_peak=0.5, n_clusters=3)
            sc = sample_scenario(params, seed=100 + seed)
            oracle = CplexOracle(cost, time_limit=15.0)  # incumbent on timeout
            j_oracle = run_simulation(sc, oracle, cost=cost,
                                      seed=seed).metrics["objective"]
            j_greedy = run_simulation(sc, GreedyController(cost), cost=cost,
                                      seed=seed).metrics["objective"]
            results.append((seed, j_oracle, j_greedy))
        wins = sum(1 for _s, jo, jg in results if jo <= jg + 1e-9)
        detail = "; ".join(f"seed{s}: oracle={jo:.2f} greedy={jg:.2f}"
                           for s, jo, jg in results)
        self.assertGreaterEqual(wins, 4,
                                msg=f"oracle wins only {wins}/5 ({detail})")


# ---------------- dual-backend tests (MipOracle: cplex + highs) ----------------
# HiGHS (Apache-2.0) solves the SAME docplex model via
# LP export + name-mapped extraction, so both backends optimise one formulation.

HAS_HIGHS = importlib.util.find_spec("highspy") is not None
requires_highs = unittest.skipUnless(HAS_HIGHS,
                                     "highspy not available in this interpreter")
requires_both_solvers = unittest.skipUnless(
    HAS_DOCPLEX and HAS_HIGHS, "need both docplex/cplex and highspy")

from dagcd import MipOracle


def small_snapshot(seed):
    """Sparse sibling of ``random_snapshot`` (lam_peak=0.3 instead of 0.6):
    ~3-13 orders at tau=30 — small enough that both backends prove optimality
    within a test time limit (see consistency-test docstring for measurements)."""
    params = InstanceParams(sim_duration=60.0, lam_peak=0.3, n_clusters=2,
                            n_vehicles=2, n_drones=2, n_vertiports=2,
                            peak_start=10.0, peak_end=50.0, nfz_density=1.0,
                            mean_event_min=15.0)
    sc = sample_scenario(params, seed=seed)
    tau = 30.0
    snap = snapshot_at(sc, tau, sc.params.horizon)
    vertiports = sorted(n for n, nd in sc.nodes.items() if nd.kind == "vertiport")
    view = FleetView(
        vehicles={vid: VehicleView(v.start_depot, v.start_time, (), False)
                  for vid, v in snap.vehicles.items()},
        drones={did: DroneView(vertiports[0] if vertiports else 0, 0.0, None,
                               d.battery_wh)
                for did, d in snap.drones.items()},
        tau=tau, sim_end=sc.sim_end)
    return snap, view


class TestMipOracleHighsBruteForce(unittest.TestCase):
    @requires_highs
    def test_highs_matches_exhaustive_optimum(self):
        # identical protocol to the cplex brute-force test: same candidate space,
        # same hand-checked optimum 37.15913584443457 (relative tolerance 1e-6)
        snap, view = tiny_snapshot()
        cost = CostParams()
        oracle = MipOracle(cost, backend="highs", mip_gap=1e-9, time_limit=60.0)
        sol = oracle(snap, view, random.Random(0))
        self.assertIsNone(sol.oracle_note)
        rep = check_solution(snap, sol, cost)
        self.assertEqual(rep.violations, [])
        got = rep.objective
        want = brute_force_optimum(snap, view, cost)
        self.assertFalse(math.isinf(want))
        self.assertLessEqual(abs(got - want), 1e-6 * max(1.0, abs(want)),
                             msg=f"HiGHS oracle {got} != brute force {want}")
        self.assertAlmostEqual(want, 37.15913584443457, delta=1e-6)
        self.assertEqual(len(sol.drone_sorties), 0)
        # route-reversal degeneracy: depot->o1->o3->o2->depot and its reverse
        # depot->o2->o3->o1->depot are the SAME cycle length, so the stops set is
        # uniquely determined but its ORDER is a multiple optimum (module
        # docstring caveat) — HiGHS legitimately returns either direction.
        self.assertEqual(sorted(s.order for s in sol.ground_routes[0].stops),
                         ["o1", "o2", "o3"])
        # provenance: HiGHS must prove optimality on this tiny model
        self.assertEqual(sol.oracle_backend, "highs")
        self.assertTrue(sol.oracle_proven)
        self.assertTrue(oracle.last_epoch["proven"])
        self.assertAlmostEqual(oracle.last_epoch["objective"], want, delta=1e-6)

    @requires_highs
    def test_highs_drone_option_when_ground_infeasible(self):
        # identical to the cplex variant: ground shift-infeasible -> one 2-parcel
        # sortie, hand-checked optimum 12.551949889388258
        snap, view = tiny_snapshot()
        snap.vehicles["gv0"] = GroundVehicle("gv0", 10.0, 0, 0.0, 20.0)
        snap.nodes[2] = Node(2, 6000.0, 0.0, "customer")
        snap.nodes[3] = Node(3, 4200.0, 4200.0, "customer")
        snap.orders["o1"].node = 2
        snap.orders["o2"].node = 3
        del snap.orders["o3"]
        view = FleetView(vehicles={"gv0": VehicleView(0, 0.0, (), False)},
                         drones={"da0": DroneView(1, 0.0, None, 500.0)},
                         tau=0.0, sim_end=240.0)
        cost = CostParams()
        oracle = MipOracle(cost, backend="highs", mip_gap=1e-9, time_limit=60.0)
        sol = oracle(snap, view, random.Random(0))
        self.assertIsNone(sol.oracle_note)
        rep = check_solution(snap, sol, cost)
        self.assertEqual(rep.violations, [])
        got = rep.objective
        want = brute_force_optimum(snap, view, cost)
        self.assertLessEqual(abs(got - want), 1e-6 * max(1.0, abs(want)),
                             msg=f"HiGHS oracle {got} != brute force {want}")
        self.assertAlmostEqual(want, 12.551949889388258, delta=1e-6)
        self.assertEqual(sol.ground_routes, [])
        self.assertEqual(len(sol.drone_sorties), 1)
        self.assertEqual(len(sol.drone_sorties[0].stops), 2)
        self.assertEqual(sol.unserved, [])
        self.assertEqual(sol.oracle_backend, "highs")
        self.assertTrue(sol.oracle_proven)


class TestMipOracleBackends(unittest.TestCase):
    """Cross-backend safety, consistency and provenance (module docstring)."""

    @requires_both_solvers
    def test_checker_safety_20_random_snapshots_both_backends(self):
        cost = CostParams()
        for seed in range(20):
            snap, view = random_snapshot(1000 + seed)
            with self.subTest(seed=seed):
                for be in ("cplex", "highs"):
                    orc = MipOracle(cost, backend=be, time_limit=15.0)
                    sol = orc(snap, view, random.Random(seed))
                    rep = check_solution(snap, sol, cost)
                    self.assertEqual(rep.violations, [])
                    assert_earliest_claims(self, snap, sol)
                    self.assertEqual(sol.oracle_backend, be)
                    self.assertIsNotNone(orc.last_epoch)

    @requires_both_solvers
    def test_dual_backend_consistency_on_proven_snapshots(self):
        """Both backends optimise the same model: on every snapshot where BOTH
        prove optimality, objective values must agree to 1e-6 relative (unique
        optimal value; the paths may differ — multiple optima caveat).

        Population split (evidence-based): the DENSE
        random_snapshot set (~15-20 orders) is infeasible to prove within any
        unit-test budget — measured proven rates at 15 s: cplex 1/20, highs 0/20;
        at 60 s: 1/20 and 0/20 (weak big-M LP relaxation). It is therefore used
        here only to RECORD proven-rate statistics. The strict consistency
        assertion runs on a SPARSE population (lam_peak=0.3, ~3-13 orders) where
        proving is achievable: measured 8/12 mutually proven at 15 s with
        max|J_cplex - J_highs| = 3.7e-08.
        """
        cost = CostParams()
        stats = {"mutually_proven": 0, "cplex_proven": 0, "highs_proven": 0,
                 "max_abs_diff": 0.0, "dense_proven_cplex": 0,
                 "dense_proven_highs": 0}
        for seed in range(20):
            snap, view = random_snapshot(1000 + seed)
            oc = MipOracle(cost, backend="cplex", time_limit=15.0)
            oh = MipOracle(cost, backend="highs", time_limit=15.0)
            oc(snap, view, random.Random(seed))
            oh(snap, view, random.Random(seed))
            stats["dense_proven_cplex"] += bool(oc.last_epoch
                                                and oc.last_epoch["proven"])
            stats["dense_proven_highs"] += bool(oh.last_epoch
                                                and oh.last_epoch["proven"])
        for seed in range(2000, 2012):
            snap, view = small_snapshot(seed)
            with self.subTest(seed=seed):
                oc = MipOracle(cost, backend="cplex", time_limit=15.0)
                oh = MipOracle(cost, backend="highs", time_limit=15.0)
                oc(snap, view, random.Random(seed))
                oh(snap, view, random.Random(seed))
                pc = bool(oc.last_epoch and oc.last_epoch["proven"])
                ph = bool(oh.last_epoch and oh.last_epoch["proven"])
                stats["cplex_proven"] += pc
                stats["highs_proven"] += ph
                if not (pc and ph):
                    continue
                stats["mutually_proven"] += 1
                jc = oc.last_epoch["objective"]
                jh = oh.last_epoch["objective"]
                d = abs(jc - jh)
                stats["max_abs_diff"] = max(stats["max_abs_diff"], d)
                self.assertLessEqual(
                    d, 1e-6 * max(1.0, abs(jc)),
                    msg=f"objective mismatch seed={seed}: cplex={jc!r} highs={jh!r}")
        TestMipOracleBackends.last_consistency_stats = stats
        self.assertGreaterEqual(stats["mutually_proven"], 5,
                                msg=f"too few mutually proven snapshots: {stats}")

    @requires_highs
    def test_highs_degenerate_and_empty(self):
        nodes = {0: Node(0, 0, 0, "depot"), 2: Node(2, 1000, 0, "customer")}
        orders = {"o1": Order("o1", 2, 0.0, 1.0, 1.0, 0.0, 90.0)}
        snap = Instance(nodes=nodes, orders=orders,
                        vehicles={"gv0": GroundVehicle("gv0", 10.0, 0, 300.0, 240.0)},
                        drones={}, nfzs=[],
                        weather=[WeatherEvent(0.0, 300.0, "clear")], horizon=240.0,
                        congestion=[(0.0, 300.0, 1.0)])
        view = FleetView(vehicles={"gv0": VehicleView(0, 300.0, (), False)},
                         drones={}, tau=0.0, sim_end=300.0)
        orc = MipOracle(backend="highs")
        sol = orc(snap, view, random.Random(0))  # must not raise
        self.assertEqual(check_solution(snap, sol, CostParams()).violations, [])
        self.assertEqual(sol.unserved, ["o1"])
        snap.orders = {}
        sol = orc(snap, view, random.Random(0))
        self.assertEqual(check_solution(snap, sol, CostParams()).violations, [])
        self.assertEqual(sol.unserved, [])

    @requires_highs
    def test_highs_size_cap_fallback_annotation(self):
        snap, view = random_snapshot(7)
        while len(snap.orders) <= 3:
            snap, view = random_snapshot(7 + len(snap.orders))
        oracle = MipOracle(CostParams(), backend="highs", max_orders=2)
        sol = oracle(snap, view, random.Random(0))
        self.assertIsNotNone(sol.oracle_note)
        self.assertIn("max_orders", sol.oracle_note)
        self.assertFalse(sol.oracle_proven)  # fallback is not MIP-proven
        self.assertEqual(sol.oracle_backend, "highs")
        self.assertEqual(check_solution(snap, sol, CostParams()).violations, [])

    @requires_both_solvers
    def test_provenance_attributes_both_backends(self):
        # provenance semantics (module docstring): backend tag always present;
        # proven=True iff the backend returned an optimality proof
        snap, view = tiny_snapshot()
        cost = CostParams()
        for be in ("cplex", "highs"):
            orc = MipOracle(cost, backend=be, mip_gap=1e-9, time_limit=60.0)
            sol = orc(snap, view, random.Random(0))
            self.assertEqual(sol.oracle_backend, be)
            self.assertTrue(sol.oracle_proven)
            self.assertEqual(orc.last_epoch["backend"], be)
        # fallback path: backend tag kept, proven=False
        orc = MipOracle(cost, backend="highs", max_orders=1)
        sol = orc(snap, view, random.Random(0))
        self.assertEqual(sol.oracle_backend, "highs")
        self.assertFalse(sol.oracle_proven)


if __name__ == "__main__":
    unittest.main()

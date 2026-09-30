"""Smoke tests for the rolling-horizon baseline controllers (baselines.py).

1. Validity + determinism: both controllers pass the checker on random
   snapshots, run end-to-end under run_simulation, and are deterministic per
   (snapshot, view, configuration) (Invariant 2 of baselines.py).
2. LookaheadReserve reserve-period semantics: inside the reserve window a
   wide-window new order is deferred while a tight-window order is served;
   outside the window both are served (when capacity allows).
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dagcd import (CostParams, InstanceParams, check_solution,
                   run_simulation, sample_scenario, snapshot_at)
from dagcd.baselines import DynamicALNS, LookaheadReserve
from dagcd.simulator import FleetView, VehicleView, DroneView


def snapshot(seed):
    params = InstanceParams(sim_duration=60.0, lam_peak=0.4, n_clusters=2,
                            n_vehicles=2, n_drones=2, n_vertiports=2,
                            peak_start=10.0, peak_end=50.0, nfz_density=1.0,
                            mean_event_min=15.0)
    sc = sample_scenario(params, seed=seed)
    snap = snapshot_at(sc, 30.0, sc.params.horizon)
    vertiports = sorted(n for n, nd in sc.nodes.items() if nd.kind == "vertiport")
    view = FleetView(
        vehicles={vid: VehicleView(v.start_depot, v.start_time, (), False)
                  for vid, v in snap.vehicles.items()},
        drones={did: DroneView(vertiports[0] if vertiports else 0, 0.0, None,
                               d.battery_wh)
                for did, d in snap.drones.items()},
        tau=30.0, sim_end=sc.sim_end)
    return snap, view


class TestBaselinesSmoke(unittest.TestCase):
    def test_validity_and_determinism(self):
        cost = CostParams()
        snap, view = snapshot(11)
        for ctl in (DynamicALNS(cost, iterations=25),
                    LookaheadReserve(cost)):
            sol = ctl(snap, view, random.Random(0))
            self.assertEqual(check_solution(snap, sol, cost).violations, [])
            sol2 = ctl(snap, view, random.Random(999))  # sim rng must not matter
            self.assertEqual(
                [(r.vehicle, [s.order for s in r.stops])
                 for r in sol.ground_routes],
                [(r.vehicle, [s.order for s in r.stops])
                 for r in sol2.ground_routes])
            self.assertEqual(sorted(sol.unserved), sorted(sol2.unserved))
        # end-to-end smoke: no raise from the run_simulation guard
        params = InstanceParams(sim_duration=180.0, lam_peak=0.5, n_clusters=2,
                                n_vehicles=2, n_drones=1)
        sc = sample_scenario(params, seed=5)
        for ctl in (DynamicALNS(iterations=25), LookaheadReserve()):
            rep = run_simulation(sc, ctl, seed=3)
            self.assertGreater(rep.metrics["served"], 0)

    def test_lookahead_reserve_period_semantics(self):
        # hand-built snapshot: one wide-window order, one tight-window order,
        # plenty of capacity -> outside reserve both served; inside reserve the
        # wide-window order is deferred, the tight one served.
        from dagcd import (Instance, Node, Order, GroundVehicle, Drone,
                           WeatherEvent)
        nodes = {0: Node(0, 0, 0, "depot"), 1: Node(1, 500, 0, "vertiport"),
                 2: Node(2, 1200, 0, "customer"), 3: Node(3, 0, 1200, "customer")}
        orders = {
            "ow": Order("ow", 2, 0.0, 1.0, 1.0, 0.0, 200.0),    # window width 200
            "ot": Order("ot", 3, 0.0, 1.0, 1.0, 0.0, 20.0),     # window width 20
        }
        snap = Instance(nodes=nodes, orders=orders,
                        vehicles={"gv0": GroundVehicle("gv0", 10.0, 0, 0.0, 900.0)},
                        drones={"da0": Drone("da0", 5.0, 500.0, 600.0, 60.0, 3)},
                        nfzs=[], weather=[WeatherEvent(0.0, 900.0, "clear")],
                        horizon=900.0, congestion=[(0.0, 900.0, 1.0)])
        ctl = LookaheadReserve(CostParams(), peak_start=420.0, peak_end=660.0,
                               reserve_lead=60.0, tight_window_max=30.0)
        # tau=300: outside the reserve window -> both served
        view = FleetView(vehicles={"gv0": VehicleView(0, 0.0, (), False)},
                         drones={"da0": DroneView(1, 0.0, None, 500.0)},
                         tau=300.0, sim_end=900.0)
        sol = ctl(snap, view, random.Random(0))
        self.assertEqual(check_solution(snap, sol, CostParams()).violations, [])
        self.assertEqual(sol.unserved, [])
        # tau=400: inside [peak_start - lead, peak_end] -> wide deferred
        view = FleetView(vehicles={"gv0": VehicleView(0, 0.0, (), False)},
                         drones={"da0": DroneView(1, 0.0, None, 500.0)},
                         tau=400.0, sim_end=900.0)
        sol = ctl(snap, view, random.Random(0))
        self.assertEqual(check_solution(snap, sol, CostParams()).violations, [])
        self.assertEqual(sol.unserved, ["ow"])
        self.assertTrue(any(s.order == "ot"
                            for r in sol.ground_routes for s in r.stops))


if __name__ == "__main__":
    unittest.main()

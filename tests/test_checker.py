"""Unit tests for the DAGCD feasibility checker (Task package #2 acceptance tests).

Hand-checkable numbers: nodes in meters, v_g=500 m/min, drone cruise=600 m/min,
congestion 1.0 everywhere, weather clear over [0,240].
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dagcd import (Instance, Node, Order, GroundVehicle, Drone, NFZ, WeatherEvent,
                   GroundRoute, GroundStop, DroneSortie, DroneStop, Solution,
                   check_solution, CostParams)
from dagcd.checker import _seg_intersect


def make_instance(veh_capacity=10.0, drone_battery=500.0, weather=None, nfzs=None):
    nodes = {
        0: Node(0, 0, 0, "depot"),
        1: Node(1, 1000, 0, "vertiport"),
        2: Node(2, 0, 1000, "vertiport"),
        3: Node(3, 500, 100),
        4: Node(4, 300, 400),
        5: Node(5, 1200, 300),
    }
    orders = {
        "o1": Order("o1", 3, 0, 2, 2, 0, 60),
        "o2": Order("o2", 4, 0, 3, 2, 0, 60),
        "o3": Order("o3", 5, 0, 1, 1, 0, 60),
    }
    return Instance(
        nodes=nodes, orders=orders,
        vehicles={"gv0": GroundVehicle("gv0", veh_capacity, 0, 0, 240)},
        drones={"da0": Drone("da0", 5.0, drone_battery, 600.0, 60.0, 3)},
        nfzs=nfzs or [],
        weather=weather or [WeatherEvent(0, 240, "clear")],
        horizon=240.0,
        congestion=[(0, 240, 1.0)],
    )


def make_solution(ground_stops=None, sortie_stops=None, unserved=None,
                  energy_wh=0.0, o1_arrival=2.0):
    """Valid baseline plan: gv0 serves o1->o2, da0 sortie serves o3."""
    gs = ground_stops or [
        GroundStop("o1", 3, o1_arrival),
        GroundStop("o2", 4, 6.0),
    ]
    ss = sortie_stops if sortie_stops is not None else [DroneStop("o3", 5, 2.0)]
    return Solution(
        epoch=0,
        ground_routes=[GroundRoute("gv0", gs, 0, 9.5)],
        drone_sorties=[DroneSortie("da0", 1, 1.0, ss, 2, 5.5, energy_wh)],
        unserved=unserved or [],
    )


# hand-derived values (see module docstring):
D_DEPOT_O1 = 509.902  # (0,0)->(500,100)
D_O1_O2 = 360.555     # (500,100)->(300,400)
D_O2_DEPOT = 500.0
D_VP1_O3 = 360.555    # (1000,0)->(1200,300)
D_O3_VP2 = 1389.244   # (1200,300)->(0,1000)
GROUND_KM = (D_DEPOT_O1 + D_O1_O2 + D_O2_DEPOT) / 1000.0
DRONE_KM = (D_VP1_O3 + D_O3_VP2) / 1000.0
ENERGY_WH = (0.02 + 0.0005 * 1) * D_VP1_O3 + 0.02 * D_O3_VP2  # 7.392+27.786


class TestChecker(unittest.TestCase):
    def test_valid_solution(self):
        rep = check_solution(make_instance(), make_solution())
        self.assertEqual(rep.violations, [])
        self.assertTrue(rep.feasible)
        m = rep.metrics
        self.assertEqual(m["served"], 3)
        self.assertAlmostEqual(m["ground_km"], GROUND_KM, places=3)
        self.assertAlmostEqual(m["drone_km"], DRONE_KM, places=3)
        self.assertAlmostEqual(m["energy_wh"], ENERGY_WH, delta=0.5)
        # J = 30 (fixed ground) + 8 (fixed sortie) + 1.0*GROUND_KM + 0.3*DRONE_KM
        self.assertAlmostEqual(rep.objective, 38.0 + GROUND_KM + 0.3 * DRONE_KM,
                               delta=0.02)

    def test_capacity_violation_c2(self):
        inst = make_instance(veh_capacity=4.0)  # o1(2)+o2(3)=5 > 4
        rep = check_solution(inst, make_solution())
        self.assertFalse(rep.feasible)
        self.assertTrue(any("C2" in v for v in rep.violations))

    def test_early_arrival_c3(self):
        rep = check_solution(make_instance(), make_solution(o1_arrival=0.5))
        self.assertFalse(rep.feasible)
        self.assertTrue(any("C3" in v for v in rep.violations))

    def test_battery_violation_c6(self):
        inst = make_instance(drone_battery=10.0)  # needs ~35 Wh
        rep = check_solution(inst, make_solution())
        self.assertFalse(rep.feasible)
        self.assertTrue(any("C6" in v for v in rep.violations))

    def test_energy_claim_mismatch(self):
        rep = check_solution(make_instance(), make_solution(energy_wh=100.0))
        self.assertFalse(rep.feasible)
        self.assertTrue(any("C6" in v for v in rep.violations))

    def test_nfz_crossing_c8(self):
        # square blocking the leg (1000,0)->(1200,300)
        nfz = NFZ("z1", [[1000, 100], [1400, 100], [1400, 500], [1000, 500]])
        rep = check_solution(make_instance(nfzs=[nfz]), make_solution())
        self.assertFalse(rep.feasible)
        self.assertTrue(any("NFZ" in v for v in rep.violations))

    def test_duplicate_service_c1(self):
        sol = make_solution(sortie_stops=[DroneStop("o3", 5, 2.0),
                                          DroneStop("o1", 3, 4.0)])
        rep = check_solution(make_instance(), sol)
        self.assertFalse(rep.feasible)
        self.assertTrue(any("served twice" in v for v in rep.violations))

    def test_missing_order_c1(self):
        inst = make_instance()
        inst.orders["o9"] = Order("o9", 4, 0, 1, 1, 0, 60)  # neither served nor unserved
        rep = check_solution(inst, make_solution())
        self.assertFalse(rep.feasible)
        self.assertTrue(any("o9" in v for v in rep.violations))

    def test_storm_launch_c8(self):
        inst = make_instance(weather=[WeatherEvent(0, 240, "storm")])
        rep = check_solution(inst, make_solution())
        self.assertFalse(rep.feasible)
        self.assertTrue(any("storm" in v for v in rep.violations))

    def test_lateness_accounting_c9(self):
        inst = make_instance()
        inst.orders["o1"].l = 1.0  # arrival 2.0 -> u = 1.0
        rep = check_solution(inst, make_solution())
        self.assertTrue(rep.feasible)
        self.assertEqual(rep.metrics["late_orders"], 1)
        self.assertAlmostEqual(rep.metrics["cost_breakdown"]["lateness"], 1.0,
                               places=6)

    def test_drone_timeline_overlap_c11(self):
        # da0 is airborne 1.0-5.5; a second sortie launched at 4.5 overlaps -> C11
        sol = make_solution()
        sol.drone_sorties.append(DroneSortie(
            "da0", 2, 4.5, [DroneStop("o3b", 4, 5.7)], 1, 9.1, 0.0))
        inst = make_instance()
        inst.orders["o3b"] = Order("o3b", 4, 0, 1, 1, 0, 60)
        rep = check_solution(inst, sol)
        self.assertFalse(rep.feasible)
        self.assertTrue(any("C11" in v for v in rep.violations))

    def test_drone_timeline_chaining_ok(self):
        # chaining: second sortie launches from the landing vertiport at/after landing
        sol = make_solution()
        sol.drone_sorties.append(DroneSortie(
            "da0", 2, 6.0, [DroneStop("o3b", 4, 7.2)], 1, 9.6, 0.0))
        inst = make_instance()
        inst.orders["o3b"] = Order("o3b", 4, 0, 1, 1, 0, 60)
        rep = check_solution(inst, sol)
        self.assertTrue(rep.feasible)


class TestGeometryDegenerates(unittest.TestCase):
    """NFZ geometry degenerate cases: the collinear
    orientation-zero shortcut must not report false intersections."""

    def test_disjoint_collinear_segments_no_intersection(self):
        # (2,0)->(3,0) vs (0,0)->(1,0): collinear, disjoint
        self.assertFalse(_seg_intersect((2, 0), (3, 0), (0, 0), (1, 0)))

    def test_endpoint_touch_is_intersection(self):
        self.assertTrue(_seg_intersect((0, 0), (1, 0), (1, 0), (2, 0)))

    def test_overlapping_collinear_is_intersection(self):
        self.assertTrue(_seg_intersect((0, 0), (3, 0), (1, 0), (2, 0)))

    def test_crossing_is_intersection(self):
        self.assertTrue(_seg_intersect((0, 0), (2, 0), (1, -1), (1, 1)))

    def test_zero_length_segment(self):
        # degenerate point segment off the line: no intersection
        self.assertFalse(_seg_intersect((0, 0), (2, 0), (5, 5), (5, 5)))
        # point on the segment: intersection
        self.assertTrue(_seg_intersect((0, 0), (2, 0), (1, 0), (1, 0)))


if __name__ == "__main__":
    unittest.main()

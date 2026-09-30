"""Unit tests for the DAGCD instance generator (sampling + snapshot + JSON I/O).

Deterministic by construction: every test fixes its seed and derives expectations
analytically from InstanceParams (order-count integral) or from the sampling rules.
"""

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dagcd import (Instance, Node, Order, GroundVehicle, Drone, NFZ, WeatherEvent,
                   WEATHER_STATES, InstanceParams, Scenario, sample_instance,
                   sample_scenario, snapshot_at, scenario_to_dict,
                   scenario_from_dict, save_scenario, load_scenario)


def small_params(**kw) -> InstanceParams:
    base = dict(sim_duration=180.0, lam_peak=0.5, n_clusters=2,
                peak_start=60.0, peak_end=120.0, n_vertiports=2,
                n_vehicles=2, n_drones=1)
    base.update(kw)
    return InstanceParams(**base)


class TestGenerator(unittest.TestCase):
    def test_same_seed_reproducible(self):
        p = small_params()
        sc1 = sample_scenario(p, seed=42)
        sc2 = sample_scenario(p, seed=42)
        self.assertEqual(scenario_to_dict(sc1), scenario_to_dict(sc2))

    def test_different_seed_differs(self):
        p = small_params()
        d1 = scenario_to_dict(sample_scenario(p, seed=1))
        d2 = scenario_to_dict(sample_scenario(p, seed=2))
        self.assertNotEqual(d1, d2)

    def test_json_round_trip(self):
        p = small_params(nfz_density=1.0)
        sc = sample_scenario(p, seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "scenario.json")
            save_scenario(sc, path)
            loaded = load_scenario(path)
        self.assertEqual(loaded, sc)
        # raw json.dumps must also accept the dict (tuples already converted)
        again = scenario_from_dict(json.loads(json.dumps(scenario_to_dict(sc))))
        self.assertEqual(again, sc)

    def test_order_count_matches_integrated_rate(self):
        # peak window fully inside [0, sim_end]: E[N] = lam_peak*(ratio*trough + peak)
        p = InstanceParams(sim_duration=720.0, lam_peak=1.0, lam_trough_ratio=0.3,
                           peak_start=480.0, peak_end=720.0)
        expected = 1.0 * (0.3 * 480.0 + 240.0)  # 384
        sc = sample_scenario(p, seed=11)
        self.assertGreaterEqual(len(sc.orders), 1)
        self.assertLessEqual(abs(len(sc.orders) - expected), 0.25 * expected)
        # releases respect [0, sim_end) and ids are dense o1..oN
        rel = [o.release for o in sc.orders.values()]
        self.assertEqual(rel, sorted(rel))
        self.assertTrue(all(0.0 <= r < p.sim_duration for r in rel))
        self.assertEqual(set(sc.orders), {f"o{i + 1}" for i in range(len(rel))})

    def test_weather_chain_valid(self):
        sc = sample_scenario(small_params(mean_event_min=10.0), seed=5)
        self.assertGreater(len(sc.weather), 0)
        self.assertEqual(sc.weather[0].t_start, 0.0)
        self.assertEqual(sc.weather[0].state, "clear")
        self.assertAlmostEqual(sc.weather[-1].t_end, sc.sim_end, places=9)
        for a, b in zip(sc.weather, sc.weather[1:]):
            self.assertAlmostEqual(a.t_end, b.t_start, places=9)  # non-overlapping
            self.assertIn(b.state, WEATHER_STATES)

    def test_nfz_schedule_and_active_window(self):
        # hand-built NFZ active only on [50, 100)
        poly = [[0, 0], [10, 0], [10, 10], [0, 10]]
        inst = Instance(nodes={}, orders={}, vehicles={}, drones={},
                        nfzs=[NFZ("z_win", poly, 50.0, 100.0),
                              NFZ("z_all", poly)],  # None/None = always active
                        weather=[], horizon=240.0)
        self.assertEqual([z.id for z in inst.active_nfzs(30.0)], ["z_all"])
        self.assertEqual([z.id for z in inst.active_nfzs(75.0)],
                         ["z_win", "z_all"])
        self.assertEqual([z.id for z in inst.active_nfzs(120.0)], ["z_all"])
        # sampled NFZs carry legal schedules and axis-aligned 4-vertex rectangles
        sc = sample_scenario(small_params(nfz_density=2.0, sim_duration=240.0),
                             seed=3)
        for z in sc.nfzs:
            self.assertEqual(len(z.polygon), 4)
            self.assertIsNotNone(z.active_from)
            self.assertIsNotNone(z.active_to)
            self.assertGreaterEqual(z.active_from, 0.0)
            self.assertLess(z.active_from, z.active_to)
            self.assertLessEqual(z.active_to, sc.sim_end)

    def test_snapshot_filters_and_horizon(self):
        sc = sample_scenario(small_params(), seed=9)
        tau, horizon = 30.0, 60.0
        snap = snapshot_at(sc, tau, horizon)
        self.assertEqual(snap.horizon, tau + horizon)
        for o in snap.orders.values():
            self.assertLessEqual(o.release, tau)
        target = next(iter(snap.orders))
        snap2 = snapshot_at(sc, tau, horizon, exclude={target})
        self.assertNotIn(target, snap2.orders)
        self.assertEqual(len(snap2.orders), len(snap.orders) - 1)
        # weather/congestion/nfz tables pass through unchanged
        self.assertEqual(snap.weather, sc.weather)
        self.assertEqual(snap.congestion, sc.congestion)
        self.assertEqual(snap.nfzs, sc.nfzs)
        # model constants mirrored from the scenario
        self.assertEqual(snap.v_g, sc.v_g)
        self.assertEqual(snap.speed_factor, sc.speed_factor)

    def test_sample_instance_wrapper(self):
        p = small_params()
        inst = sample_instance(p, seed=13)
        self.assertIsInstance(inst, Instance)
        self.assertEqual(inst.horizon, p.horizon)  # tau=0 + params.horizon


class TestGeometryDeterminism(unittest.TestCase):
    def test_dist_formula_bit_exact(self):
        # Pin the distance formula: sqrt(dx*dx + dy*dy), NOT math.hypot.
        # math.hypot is libm-dependent and was verified to differ across CPython
        # minor versions (3.9 vs 3.10: 2700.0000000000005 vs 2700.0), silently
        # changing simulation geometry and invalidating archived J values.
        # IEEE-754 multiply/add/sqrt are correctly rounded -> bit-stable.
        # assertEqual on purpose: any formula drift must fail loudly.
        inst = Instance(
            nodes={0: Node(0, 0.0, 0.0, "depot"),
                   1: Node(1, 1530.0, 2258.0, "customer"),
                   2: Node(2, 6000.0, 5578.0, "customer")},
            orders={}, vehicles={}, drones={}, nfzs=[],
            weather=[WeatherEvent(0.0, 300.0, "clear")], horizon=240.0,
            congestion=[(0.0, 300.0, 1.0)])
        for i, j in ((0, 1), (0, 2), (1, 2), (1, 1)):
            a, b = inst.nodes[i], inst.nodes[j]
            dx, dy = a.x - b.x, a.y - b.y
            self.assertEqual(inst.dist(i, j), math.sqrt(dx * dx + dy * dy))
        # known-value anchor (Pythagorean triple, exact in binary)
        inst.nodes[3] = Node(3, 3000.0, 4000.0, "customer")
        self.assertEqual(inst.dist(0, 3), 5000.0)

    def test_snapshot_info_mode_full_unchanged(self):
        # default/full mode must be identical to the pre-info_mode semantics
        sc = sample_scenario(small_params(nfz_density=2.0), seed=17)
        for tau, h in ((0.0, 240.0), (30.0, 120.0), (215.5, 240.0)):
            ref = snapshot_at(sc, tau, h)
            got = snapshot_at(sc, tau, h, info_mode="full")
            self.assertEqual(ref, got)
            self.assertEqual(got.weather, sc.weather)      # clairvoyant tables
            self.assertEqual(got.nfzs, sc.nfzs)
        with self.assertRaises(ValueError):
            snapshot_at(sc, 10.0, 60.0, info_mode="clairvoyant")

    def test_snapshot_info_mode_forecast_truncation(self):
        # hand-built chain: clear [0,100), rain [100,200), windy [200,300);
        # NFZs: z_past ended, z_now active across tau, z_future not yet active
        nodes = {0: Node(0, 0, 0, "depot"), 1: Node(1, 500, 0, "vertiport"),
                 2: Node(2, 900, 0, "customer")}
        orders = {"o1": Order("o1", 2, 0.0, 1.0, 1.0, 0.0, 90.0)}
        poly = [[0, 0], [10, 0], [10, 10], [0, 10]]
        sc = Scenario(
            params=small_params(), nodes=nodes, orders=orders,
            vehicles={"gv0": GroundVehicle("gv0", 10.0, 0, 0.0, 300.0)},
            drones={"da0": Drone("da0", 5.0, 500.0, 600.0, 60.0, 3)},
            nfzs=[NFZ("z_past", poly, 0.0, 50.0),
                  NFZ("z_now", poly, 120.0, 250.0),
                  NFZ("z_future", poly, 400.0, 450.0)],
            weather=[WeatherEvent(0.0, 100.0, "clear"),
                     WeatherEvent(100.0, 200.0, "rain"),
                     WeatherEvent(200.0, 300.0, "windy")],
            congestion=[(0.0, 300.0, 1.0)], sim_end=300.0)
        snap = snapshot_at(sc, 150.0, 240.0, info_mode="forecast")
        # rain [100,200) has NOT fully ended at tau=150 -> truncated away, and
        # the persistence forecast continues the currently observed state (rain)
        self.assertEqual([(w.t_start, w.t_end, w.state) for w in snap.weather],
                         [(0.0, 100.0, "clear"), (150.0, 300.0, "rain")])
        self.assertEqual([z.id for z in snap.nfzs], ["z_now"])
        # boundary: tau exactly at an event end -> that event is fully observed,
        # current state comes from the next (dropped) event
        snap = snapshot_at(sc, 200.0, 240.0, info_mode="forecast")
        self.assertEqual([(w.t_start, w.t_end, w.state) for w in snap.weather],
                         [(0.0, 100.0, "clear"), (100.0, 200.0, "rain"),
                          (200.0, 300.0, "windy")])
        self.assertEqual([z.id for z in snap.nfzs], ["z_now"])
        # horizon and orders semantics unchanged
        self.assertEqual(snap.horizon, 200.0 + 240.0)
        self.assertEqual(list(snap.orders), ["o1"])
        # full mode stays clairvoyant
        full = snapshot_at(sc, 150.0, 240.0, info_mode="full")
        self.assertEqual(len(full.weather), 3)
        self.assertEqual(len(full.nfzs), 3)


if __name__ == "__main__":
    unittest.main()

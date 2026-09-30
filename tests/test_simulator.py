"""Unit tests for the DAGCD rolling-horizon simulator.

Smoke run under GreedyController plus hand-built scenarios for trigger semantics
and the foundation guard (infeasible epoch solutions must raise).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dagcd import (Node, Order, GroundVehicle, Drone, WeatherEvent, Solution,
                   GroundRoute, GroundStop, InstanceParams, Scenario, sample_scenario,
                   run_simulation, GreedyController)


def make_scenario(order_releases, sim_end=60.0, horizon=240.0):
    """Tiny hand-built scenario: depot 0, vertiport 1, customers 2.., one of each."""
    nodes = {0: Node(0, 0, 0, "depot"), 1: Node(1, 1000, 0, "vertiport")}
    orders = {}
    for i, rel in enumerate(order_releases):
        nid = 2 + i
        nodes[nid] = Node(nid, 500.0 + 100.0 * i, 0, "customer")
        orders[f"o{i + 1}"] = Order(f"o{i + 1}", nid, rel, 1.0, 1.0, rel, rel + 90.0)
    params = InstanceParams(sim_duration=sim_end, horizon=horizon,
                            n_vertiports=1, n_vehicles=1, n_drones=1)
    return Scenario(
        params=params, nodes=nodes, orders=orders,
        vehicles={"gv0": GroundVehicle("gv0", 10.0, 0, 0.0, sim_end)},
        drones={"da0": Drone("da0", 5.0, 500.0, 600.0, 60.0, 3)},
        nfzs=[], weather=[WeatherEvent(0.0, sim_end, "clear")],
        congestion=[(0.0, sim_end, 1.0)], sim_end=sim_end)


class TestSimulator(unittest.TestCase):
    def test_smoke_end_to_end(self):
        params = InstanceParams(sim_duration=180.0, lam_peak=0.3, n_clusters=2,
                                n_vehicles=2, n_drones=1)
        sc1 = sample_scenario(params, seed=21)
        sc2 = sample_scenario(params, seed=21)
        rep1 = run_simulation(sc1, GreedyController(), seed=4)
        rep2 = run_simulation(sc2, GreedyController(), seed=4)
        m = rep1.metrics
        # coverage: every order delivered or unserved, disjointly
        dlv = {d["order_id"] for d in rep1.deliveries}
        self.assertEqual(dlv | set(rep1.unserved), set(sc1.orders))
        self.assertFalse(dlv & set(rep1.unserved))
        # rates and epoch bookkeeping
        self.assertGreater(m["served"], 0)
        self.assertAlmostEqual(m["served_rate"], m["served"] / len(sc1.orders))
        self.assertTrue(0.0 <= m["served_rate"] <= 1.0)
        self.assertGreaterEqual(len(rep1.epochs), params.sim_duration / 30.0)
        self.assertTrue(all(e.violations == 0 for e in rep1.epochs))
        # J_sim self-consistency: breakdown sums to the objective
        self.assertAlmostEqual(sum(m["cost_breakdown"].values()),
                               m["objective"], delta=1e-6)
        # reproducibility: same scenario seed + same sim seed -> identical outcome
        self.assertEqual(rep1.deliveries, rep2.deliveries)
        self.assertEqual(rep1.unserved, rep2.unserved)
        self.assertEqual(m["objective"], rep2.metrics["objective"])
        self.assertEqual(rep1.params["sim_duration"], params.sim_duration)

    def test_order_batch_trigger(self):
        # 3 orders released together at t=5; n_trig=2 -> epoch at t=5 must be
        # triggered by the order batch, not the 30-min timer.
        sc = make_scenario([5.0, 5.0, 5.0])
        rep = run_simulation(sc, GreedyController(), epoch_delta=30.0, n_trig=2)
        self.assertIn("order_batch", [e.trigger for e in rep.epochs])
        batch = [e for e in rep.epochs if e.trigger == "order_batch"]
        self.assertEqual(batch[0].tau, 5.0)
        self.assertEqual(batch[0].released, 3)
        # all three orders fit one vehicle and get served
        self.assertEqual(rep.metrics["served"], 3)
        self.assertEqual(rep.unserved, [])

    def test_timer_trigger_when_no_events(self):
        sc = make_scenario([1.0])  # single order, below n_trig
        rep = run_simulation(sc, GreedyController(), epoch_delta=15.0, n_trig=2)
        taus = [e.tau for e in rep.epochs]
        self.assertEqual(taus[0], 0.0)
        self.assertIn(15.0, taus)   # pure timer epoch
        self.assertNotIn("order_batch", [e.trigger for e in rep.epochs])
        self.assertEqual(rep.metrics["served"], 1)

    def test_infeasible_controller_raises(self):
        sc = make_scenario([0.0])  # released at tau=0 so it must be covered

        def bad_controller(snap, view, rng):
            return Solution(epoch=0,
                            ground_routes=[GroundRoute("gv0", [
                                GroundStop("o1", 2, 0.001)], 0, 2.0)],
                            drone_sorties=[], unserved=[])

        with self.assertRaises(RuntimeError) as ctx:
            run_simulation(sc, bad_controller)
        self.assertIn("C3", str(ctx.exception))  # claimed 0.001 < earliest feasible


if __name__ == "__main__":
    unittest.main()

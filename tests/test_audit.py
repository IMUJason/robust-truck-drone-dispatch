"""Unit tests for the adversarial regret audit harness (docs/SPECIFICATION.md Sec. 5.1).

Invariants (mirror src/dagcd/audit.py; see its module docstring for the full
statements — the quantile_ub formula, the scenario-approach basis, and the
determinism preconditions):
  * reproducibility — same inputs + same base_seed give identical reports
    (asserted via audit_to_dict equality across two full runs);
  * regret definition — regret == J_heuristic - J_oracle exactly, per sample;
  * quantile_ub formula — _quantile_ub_index checked point-by-point against a
    brute-force binomial sum, both branches (k <= N, k > N), plus monotonicity
    in alpha and quantile;
  * factory freshness — every simulation instantiates via factory() (code
    review of audit.py; a shared stateful controller would void determinism);
  * inversions transparent — any regret < -TOL lands in summary["inversions"]
    with its numbers;
  * JSON round-trip — save/load preserves the report dict.
CPLEX-dependent tests skip when docplex is not importable.
"""

import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dagcd import (InstanceParams, GreedyController, InsertionOracle, CplexOracle,
                   adversarial_regret_audit, audit_to_dict, audit_from_dict,
                   save_audit, load_audit)
from dagcd.audit import _quantile_ub_index, _binomial_cdf, TOL

HAS_DOCPLEX = importlib.util.find_spec("docplex") is not None
requires_docplex = unittest.skipUnless(HAS_DOCPLEX,
                                       "docplex/cplex not available in this interpreter")

SMALL = InstanceParams(sim_duration=180.0, lam_peak=0.4, n_clusters=2,
                       n_vehicles=2, n_drones=1, n_vertiports=2)
EPS = {"lam": 0.3, "cong": 0.2}


def brute_cdf(k_max, n, p):
    return sum(math.comb(n, j) * p ** j * (1.0 - p) ** (n - j)
               for j in range(0, min(k_max, n) + 1))


class TestQuantileUbIndex(unittest.TestCase):
    def test_matches_brute_force_grid(self):
        for n in (5, 16, 50, 200):
            for q in (0.9, 0.95, 0.99):
                for alpha in (0.01, 0.05, 0.1):
                    with self.subTest(n=n, q=q, alpha=alpha):
                        k = _quantile_ub_index(n, q, alpha)
                        if k is not None:
                            self.assertGreaterEqual(k, 1)
                            self.assertLessEqual(k, n)
                            # k is the SMALLEST index meeting the condition
                            self.assertGreaterEqual(brute_cdf(k - 1, n, q),
                                                    1.0 - alpha - 1e-12)
                            if k > 1:
                                self.assertLess(brute_cdf(k - 2, n, q),
                                                1.0 - alpha - 1e-12)
                        else:
                            # no k <= N works: even k = N fails
                            self.assertLess(brute_cdf(n - 1, n, q),
                                            1.0 - alpha - 1e-12)

    def test_cdf_helper_matches_brute(self):
        for n in (5, 16, 50):
            for q in (0.9, 0.95):
                for k in range(-1, n + 2):
                    self.assertAlmostEqual(_binomial_cdf(k, n, q),
                                           brute_cdf(k, n, q), places=12)

    def test_k_exceeds_n_branch(self):
        # N=5, q=0.99, alpha=0.01: even P(Bin <= 4) = 1 - 0.99^5 = 0.049 < 0.99
        self.assertIsNone(_quantile_ub_index(5, 0.99, 0.01))
        # even alpha=0.1 fails (0.049 < 0.9): the bound needs 1-alpha <= 0.049
        self.assertIsNone(_quantile_ub_index(5, 0.99, 0.1))
        # ... i.e. only alpha >= 1 - 0.049 admits k; verify that boundary branch
        self.assertIsNotNone(_quantile_ub_index(5, 0.99, 0.99))

    def test_monotone_alpha_and_quantile(self):
        for n in (16, 50, 200):
            for q in (0.9, 0.95, 0.99):
                ks = [_quantile_ub_index(n, q, a) for a in (0.1, 0.05, 0.01)]
                present = [k for k in ks if k is not None]
                # alpha down => k up (weakly): sequence is non-decreasing
                for a, b in zip(present, present[1:]):
                    self.assertLessEqual(a, b)
            for a in (0.05, 0.1):
                kq = [_quantile_ub_index(n, q, a) for q in (0.9, 0.95, 0.99)]
                present = [k for k in kq if k is not None]
                # quantile up => k up (weakly), where defined
                for x, y in zip(present, present[1:]):
                    self.assertGreaterEqual(y, x)

    def test_n_zero(self):
        self.assertIsNone(_quantile_ub_index(0, 0.95, 0.05))


class TestAuditEndToEnd(unittest.TestCase):
    def test_small_audit_runs(self):
        rep = adversarial_regret_audit(
            SMALL, lambda: GreedyController(), lambda: InsertionOracle(),
            eps=EPS, n_adversarial=3, nominal_seeds=2, base_seed=7)
        d = audit_to_dict(rep)
        # segment sizes and exact regret identity per sample
        self.assertEqual(len(d["nominal"]), 2)
        self.assertEqual(len(d["adversarial"]), 3)
        for seg in ("nominal", "adversarial"):
            for row in d[seg]:
                self.assertAlmostEqual(row["regret"],
                                       row["J_heuristic"] - row["J_oracle"],
                                       delta=1e-9)
                self.assertGreater(row["J_heuristic"], 0.0)
                self.assertGreater(row["J_oracle"], 0.0)
        # summary fields complete
        for key in ("nominal", "adversarial", "quantile_ub",
                    "quantile_ub_index", "inversions", "notes"):
            self.assertIn(key, d["summary"])
        for seg in ("nominal", "adversarial"):
            for stat in ("mean", "std", "max", "q95"):
                self.assertIn(stat, d["summary"][seg]["regret"])
        # perturbation took effect: every theta_prime differs from theta
        for row in d["adversarial"]:
            self.assertNotEqual(row["theta_prime"], d["theta"])
        # quantile_ub: None (N too small) or a valid order statistic >= q95
        k = d["summary"]["quantile_ub_index"]
        ub = d["summary"]["quantile_ub"]
        regrets = sorted(r["regret"] for r in d["adversarial"])
        if k is None:
            self.assertIsNone(ub)
        else:
            self.assertEqual(ub, regrets[k - 1])
            self.assertGreaterEqual(ub, d["summary"]["adversarial"]["regret"]["q95"])
        # adversarial samples carry the full (scrubbed) heuristic metrics
        m = d["adversarial"][0]["metrics_heuristic"]
        for key in ("served", "served_rate", "avg_lateness", "p95_lateness",
                    "ground_km", "drone_km", "energy_wh", "cost_breakdown",
                    "objective", "cost_per_served"):
            self.assertIn(key, m)
        self.assertAlmostEqual(sum(m["cost_breakdown"].values()),
                               m["objective"], delta=1e-6)

    def test_determinism_two_runs_identical(self):
        def run():
            return adversarial_regret_audit(
                SMALL, lambda: GreedyController(), lambda: InsertionOracle(),
                eps=EPS, n_adversarial=3, nominal_seeds=2, base_seed=7)
        d1 = audit_to_dict(run())
        d2 = audit_to_dict(run())
        self.assertEqual(d1, d2)

    def test_theta_prime_sequence_reproducible(self):
        rep = adversarial_regret_audit(
            SMALL, lambda: GreedyController(), lambda: InsertionOracle(),
            eps=EPS, n_adversarial=3, nominal_seeds=1, base_seed=11)
        reps = adversarial_regret_audit(
            SMALL, lambda: GreedyController(), lambda: InsertionOracle(),
            eps=EPS, n_adversarial=3, nominal_seeds=1, base_seed=11)
        t1 = [r["theta_prime"] for r in audit_to_dict(rep)["adversarial"]]
        t2 = [r["theta_prime"] for r in audit_to_dict(reps)["adversarial"]]
        self.assertEqual(t1, t2)
        # different base_seed must give a different perturbation sequence
        rep3 = adversarial_regret_audit(
            SMALL, lambda: GreedyController(), lambda: InsertionOracle(),
            eps=EPS, n_adversarial=3, nominal_seeds=1, base_seed=12)
        t3 = [r["theta_prime"] for r in audit_to_dict(rep3)["adversarial"]]
        self.assertNotEqual(t1, t3)

    def test_factory_descriptor_rebuildable(self):
        rep = adversarial_regret_audit(
            SMALL, lambda: GreedyController(), lambda: InsertionOracle(),
            eps=EPS, n_adversarial=1, nominal_seeds=1, base_seed=3)
        d = audit_to_dict(rep)
        for key, cls in (("heuristic", GreedyController), ("oracle", InsertionOracle)):
            self.assertEqual(d[key]["class"], cls.__name__)
            self.assertIn("config", d[key])
            self.assertIn("module", d[key])

    def test_json_round_trip(self):
        rep = adversarial_regret_audit(
            SMALL, lambda: GreedyController(), lambda: InsertionOracle(),
            eps=EPS, n_adversarial=3, nominal_seeds=2, base_seed=7)
        d = audit_to_dict(rep)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "audit.json")
            save_audit(rep, path)
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.assertEqual(raw, d)
            rep2 = load_audit(path)
            self.assertEqual(audit_to_dict(rep2), d)
        # dict-level round trip as well
        self.assertEqual(audit_to_dict(audit_from_dict(json.loads(json.dumps(d)))), d)


class TestAuditCplexSpot(unittest.TestCase):
    @requires_docplex
    def test_cplex_oracle_inside_harness(self):
        rep = adversarial_regret_audit(
            SMALL, lambda: GreedyController(),
            lambda: CplexOracle(time_limit=5.0, max_orders=25),
            eps=EPS, n_adversarial=1, nominal_seeds=1, base_seed=5)
        d = audit_to_dict(rep)
        # harness ran end to end; the MILP must not have fallen back
        fallback_notes = [n for n in d["summary"]["notes"] if "oracle_note" in n]
        self.assertEqual(fallback_notes, [])
        for row in d["nominal"] + d["adversarial"]:
            self.assertAlmostEqual(row["regret"],
                                   row["J_heuristic"] - row["J_oracle"],
                                   delta=1e-9)


if __name__ == "__main__":
    unittest.main()

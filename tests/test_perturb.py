"""Unit tests for the DAGCD epsilon-ball perturbation operator (docs/SPECIFICATION.md Sec. 6).

Every bound is asserted numerically (log-bounds / additive bound / TV bound), the
no-mutation and no-aliasing guarantees are checked explicitly, seeds are archived,
and extreme eps values (all-zero, very large) must not crash and must yield valid
parameters. TV distance is defined exactly as in generator._perturb_wx:
TV(P', P) = max_state 0.5 * sum_next |P'[s][t] - P[s][t]|.
"""

import copy
import math
import os
import random
import sys
import unittest
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dagcd import InstanceParams, WEATHER_STATES, perturb_params, sample_scenario

EPS = {"lam": 0.25, "clusters": 0.20, "cong": 0.15, "nfz": 0.30, "wx": 0.12}


def tv_distance(p, q):
    return max(0.5 * sum(abs(p[s][i] - q[s][i]) for i in range(len(WEATHER_STATES)))
               for s in WEATHER_STATES)


class TestPerturbBounds(unittest.TestCase):
    def test_seed_reproducible(self):
        p = InstanceParams()
        p1 = perturb_params(p, EPS, random.Random(42))
        p2 = perturb_params(p, EPS, random.Random(42))
        self.assertEqual(asdict(p1), asdict(p2))
        p3 = perturb_params(p, EPS, random.Random(43))
        self.assertNotEqual(asdict(p1), asdict(p3))

    def test_bounds_hold_all_keys(self):
        p = InstanceParams()
        for seed in range(10):
            with self.subTest(seed=seed):
                q = perturb_params(p, EPS, random.Random(seed))
                self.assertLessEqual(abs(math.log(q.lam_peak / p.lam_peak)),
                                     EPS["lam"] + 1e-12)
                self.assertLessEqual(abs(math.log(q.cluster_spread_m
                                                  / p.cluster_spread_m)),
                                     EPS["clusters"] + 1e-12)
                self.assertGreaterEqual(q.cluster_spread_m, 1.0)  # legality floor
                self.assertLessEqual(abs(q.cong_level - p.cong_level),
                                     EPS["cong"] + 1e-12)
                self.assertTrue(0.0 <= q.cong_level <= 1.0)
                self.assertLessEqual(abs(math.log(q.nfz_density / p.nfz_density)),
                                     EPS["nfz"] + 1e-12)
                self.assertGreaterEqual(q.nfz_density, 0.01)        # spec floor
                self.assertLessEqual(tv_distance(p.wx_chain, q.wx_chain),
                                     EPS["wx"] + 1e-12)
                # wx stays a stochastic matrix
                for s in WEATHER_STATES:
                    row = q.wx_chain[s]
                    self.assertTrue(all(v >= 0.0 for v in row))
                    self.assertAlmostEqual(sum(row), 1.0, places=9)

    def test_keys_independent_subset(self):
        p = InstanceParams()
        q = perturb_params(p, {"lam": 0.3}, random.Random(1))
        self.assertNotEqual(q.lam_peak, p.lam_peak)
        self.assertEqual(q.cluster_spread_m, p.cluster_spread_m)
        self.assertEqual(q.cong_level, p.cong_level)
        self.assertEqual(q.nfz_density, p.nfz_density)
        self.assertEqual(q.wx_chain, p.wx_chain)
        # perturbing only "clusters" leaves lam untouched (asdict for deep equality)
        q2 = perturb_params(p, {"clusters": 0.3}, random.Random(1))
        self.assertEqual(q2.lam_peak, p.lam_peak)
        self.assertNotEqual(q2.cluster_spread_m, p.cluster_spread_m)

    def test_tv_shrink_lands_on_boundary_when_exceeded(self):
        # force the raw perturbation far outside the ball with a scripted rng and
        # verify the projection lands exactly on the TV=eps boundary
        from dagcd.generator import _perturb_wx

        class ScriptedRng:
            def __init__(self):
                self.calls = 0

            def uniform(self, a, b):
                self.calls += 1
                return b  # maximal push for every entry

        chain = {s: [0.7, 0.1, 0.1, 0.1] for s in WEATHER_STATES}
        eps = 0.05
        out = _perturb_wx(chain, eps, ScriptedRng())
        # raw TV of the clipped+renormalized matrix exceeds eps -> shrink fired
        self.assertLessEqual(tv_distance(chain, out), eps + 1e-12)
        # ... and sits on the boundary (linear shrink: TV(lam) = lam * TV_raw)
        raw_tv = 0.075  # 0.5*(0.075 + 3*0.025) for the scripted maximal push
        self.assertGreater(raw_tv, eps)
        self.assertAlmostEqual(tv_distance(chain, out), eps, delta=1e-12)
        for s in WEATHER_STATES:
            self.assertAlmostEqual(sum(out[s]), 1.0, places=12)

    def test_tv_bound_many_trials(self):
        p = InstanceParams()
        for seed in range(30):
            q = perturb_params(p, {"wx": 0.10}, random.Random(1000 + seed))
            self.assertLessEqual(tv_distance(p.wx_chain, q.wx_chain), 0.10 + 1e-12)


class TestPerturbIntegrity(unittest.TestCase):
    def test_input_not_mutated_no_alias(self):
        p = InstanceParams()
        snapshot_before = asdict(p)
        q = perturb_params(p, EPS, random.Random(0))
        self.assertEqual(asdict(p), snapshot_before)   # input untouched
        self.assertIsNot(q, p)
        self.assertIsNot(q.wx_chain, p.wx_chain)
        # mutating the result must not leak into the original (deep copy)
        q.wx_chain["clear"][0] = -999.0
        q.lam_peak = -1.0
        self.assertEqual(p.wx_chain["clear"][0], 0.90)
        self.assertEqual(p.lam_peak, 3.0)

    def test_empty_eps_identity_copy(self):
        p = InstanceParams()
        for eps in (None, {}):
            q = perturb_params(p, eps, random.Random(0))
            self.assertEqual(asdict(q), asdict(p))
            self.assertIsNot(q, p)
            self.assertIsNot(q.wx_chain, p.wx_chain)

    def test_zero_eps_exact_identity(self):
        p = InstanceParams(cong_level=0.5)
        zero = {"lam": 0.0, "clusters": 0.0, "cong": 0.0, "nfz": 0.0, "wx": 0.0}
        q = perturb_params(p, zero, random.Random(5))
        self.assertEqual(asdict(q), asdict(p))

    def test_extreme_eps_legal_and_no_crash(self):
        p = InstanceParams()
        big = {"lam": 1e6, "clusters": 1e6, "cong": 1e6, "nfz": 1e6, "wx": 1e6}
        q = perturb_params(p, big, random.Random(0))  # must not raise/overflow
        self.assertTrue(math.isfinite(q.lam_peak) and q.lam_peak > 0.0)
        self.assertTrue(math.isfinite(q.cluster_spread_m)
                        and 1.0 <= q.cluster_spread_m <= 1e9)
        self.assertTrue(0.0 <= q.cong_level <= 1.0)
        self.assertTrue(math.isfinite(q.nfz_density) and q.nfz_density >= 0.01)
        for s in WEATHER_STATES:
            self.assertTrue(all(v >= 0.0 and math.isfinite(v)
                                for v in q.wx_chain[s]))
            self.assertAlmostEqual(sum(q.wx_chain[s]), 1.0, places=9)
        # sanity: moderately perturbed parameters still define a samplable
        # distribution (the huge-lam case above is legality-only, not sampled)
        p_small = InstanceParams(sim_duration=60.0, peak_start=10.0, peak_end=50.0)
        q2 = perturb_params(p_small, {"lam": 0.5, "clusters": 0.5, "cong": 0.5,
                                      "nfz": 0.5, "wx": 0.5}, random.Random(1))
        sc = sample_scenario(q2, seed=3)
        self.assertGreater(len(sc.orders), 0)
        self.assertGreater(len(sc.weather), 0)


class TestCoupledPerturb(unittest.TestCase):
    """Coupled-mode (couple=True): with probability 1/2
    a coherent 'bad-day' axis aligns every requested component's perturbation
    sign (lam up, nfz up, cong up, wx storm-directed); magnitudes stay
    independent inside each eps bound. couple=False must remain bit-identical
    to the uncoupled reference implementation (kept in-test below)."""

    EPS_C = {"lam": 0.3, "cong": 0.2, "nfz": 0.3, "wx": 0.1, "clusters": 0.4}

    def _uncoupled_reference(self, params, eps, rng):
        """Copy of the uncoupled (independent-box) perturb_params draw logic
        (kept in-test as the anti-regression oracle for couple=False)."""
        from dagcd.generator import _multiplicative, _perturb_wx
        out = copy.deepcopy(params)
        eps = eps or {}
        if "lam" in eps:
            out.lam_peak = params.lam_peak * _multiplicative(eps["lam"], rng)
        if "clusters" in eps:
            out.cluster_spread_m = min(max(params.cluster_spread_m
                                           * _multiplicative(eps["clusters"], rng),
                                           1.0), 1e9)
        if "cong" in eps:
            out.cong_level = min(max(params.cong_level
                                     + rng.uniform(-eps["cong"], eps["cong"]),
                                     0.0), 1.0)
        if "nfz" in eps:
            out.nfz_density = max(params.nfz_density
                                  * _multiplicative(eps["nfz"], rng), 0.01)
        if "wx" in eps:
            out.wx_chain = _perturb_wx(params.wx_chain, eps["wx"], rng)
        return out

    def test_couple_false_bit_identical_to_uncoupled(self):
        p = InstanceParams()
        for seed in range(25):
            with self.subTest(seed=seed):
                q_new = perturb_params(p, EPS, random.Random(seed), couple=False)
                q_ref = self._uncoupled_reference(p, EPS, random.Random(seed))
                self.assertEqual(asdict(q_new), asdict(q_ref))

    def test_bounds_hold_coupled(self):
        p = InstanceParams()
        for seed in range(40):
            q = perturb_params(p, EPS, random.Random(seed))  # couple default
            self.assertLessEqual(abs(math.log(q.lam_peak / p.lam_peak)),
                                 EPS["lam"] + 1e-12)
            self.assertLessEqual(abs(math.log(q.cluster_spread_m
                                              / p.cluster_spread_m)),
                                 EPS["clusters"] + 1e-12)
            self.assertLessEqual(abs(q.cong_level - p.cong_level),
                                 EPS["cong"] + 1e-12)
            self.assertTrue(0.0 <= q.cong_level <= 1.0)
            self.assertLessEqual(abs(math.log(q.nfz_density / p.nfz_density)),
                                 EPS["nfz"] + 1e-12)
            self.assertGreaterEqual(q.nfz_density, 0.01)
            self.assertLessEqual(tv_distance(p.wx_chain, q.wx_chain),
                                 EPS["wx"] + 1e-12)
            for s in WEATHER_STATES:
                row = q.wx_chain[s]
                self.assertTrue(all(v >= 0.0 for v in row))
                self.assertAlmostEqual(sum(row), 1.0, places=9)

    def test_sign_coherence_exceeds_independent_baseline(self):
        p = InstanceParams()
        n = 2000
        eps3 = {"lam": 0.3, "cong": 0.2, "nfz": 0.3}
        bad = 0
        for seed in range(n):
            q = perturb_params(p, eps3, random.Random(10000 + seed))
            bad += (q.lam_peak > p.lam_peak
                    and q.nfz_density > p.nfz_density
                    and q.cong_level > p.cong_level)
        frac = bad / n
        # coupled: P(bad-day axis)=0.5, signs then always align -> ~0.5
        # independent: 1/8 = 0.125; binomial std ~0.011 -> margins are >20 sigma
        self.assertGreaterEqual(frac, 0.40)
        indep = 0
        for seed in range(n):
            q = perturb_params(p, eps3, random.Random(10000 + seed),
                               couple=False)
            indep += (q.lam_peak > p.lam_peak
                      and q.nfz_density > p.nfz_density
                      and q.cong_level > p.cong_level)
        self.assertLessEqual(indep / n, 0.20)

    def test_wx_stormward_drift_coupled(self):
        p = InstanceParams()
        n = 800
        storm = WEATHER_STATES.index("storm")

        def inbound(chain):
            return sum(chain[s][storm] for s in WEATHER_STATES if s != "storm") / 3.0

        coupled_sum = 0.0
        for seed in range(n):
            q = perturb_params(p, {"wx": 0.15}, random.Random(500 + seed))
            coupled_sum += inbound(q.wx_chain)
        indep_sum = 0.0
        for seed in range(n):
            q = perturb_params(p, {"wx": 0.15}, random.Random(500 + seed),
                               couple=False)
            indep_sum += inbound(q.wx_chain)
        # bad-day rows push non-storm -> storm transitions up; independent is
        # symmetric around the baseline (drift ~ 0). Margin is many sigmas.
        self.assertGreater(coupled_sum / n - indep_sum / n, 0.002)

    def test_reproducible_coupled(self):
        p = InstanceParams()
        q1 = perturb_params(p, EPS, random.Random(42))
        q2 = perturb_params(p, EPS, random.Random(42))
        self.assertEqual(asdict(q1), asdict(q2))
        q3 = perturb_params(p, EPS, random.Random(43))
        self.assertNotEqual(asdict(q1), asdict(q3))


if __name__ == "__main__":
    unittest.main()

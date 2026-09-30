"""Unit tests for the LLM evolution outer loop (src/dagcd/evolve.py).

Invariants under test (mirror evolve.py's module docstring):
  * compile_controller: valid template passes probe; syntax error / missing
    create_controller / wrong return type each raise ProgramError; the
    namespace of one program cannot pollute another's globals;
  * MockLLM end-to-end loop (population=4, generations=2, tiny fitness
    scenario): archives exist with one row per individual per generation,
    the best program compiles/probes and survives a full run_simulation,
    llm_calls stay within budget, the trajectory is recorded and monotone
    non-decreasing in best-fitness, and a cache hit never re-runs the audit
    (a MockLLM-injected broken source is penalized identically on repeat,
    with no simulation work);
  * OpenAICompatibleClient with a fake transport: request body format,
    content parsing, reasoning_content fallback, HTTP 500 -> EvolutionError,
    from_env_file priority explicit args > environment > env file.

No test touches the network (MockLLM / injected fake transports only).

"""

import hashlib
import json
import math
import os
import sys
from dataclasses import replace
import tempfile
import unittest
from dataclasses import replace
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dagcd import (InstanceParams, Solution, run_simulation, sample_scenario,
                   EvolutionError)
import dagcd.evolve as evolve
from dagcd.evolve import (AnthropicClient, MockLLM, OpenAICompatibleClient,
                          ProgramError, client_from_env_file,
                          FitnessConfig, EvalResult, EvolutionConfig,
                          EvolutionResult, SEED_TEMPLATE, compile_controller,
                          probe_controller, evaluate_program, clear_eval_cache,
                          run_evolution)

SMALL = InstanceParams(sim_duration=120.0, lam_peak=0.3, n_clusters=2,
                       n_vehicles=1, n_drones=1, n_vertiports=2)
FIT = FitnessConfig(params=SMALL, eps={"lam": 0.2}, nominal_seeds=1,
                    n_adversarial=2, base_seed=0,
                    sim_kwargs={"epoch_delta": 30.0, "n_trig": 3})

BAD_SYNTAX = "def create_controller(cost=None):\n    return lambda s, v, r\n"

NO_FACTORY = "X = 1\n"

WRONG_TYPE = ("def create_controller(cost=None):\n"
              "    return 42\n")

MARK_A = ('MARK = "a"\n'
          'def create_controller(cost=None):\n'
          '    def c(snap, view, rng):\n'
          '        return Solution(0 if MARK == "a" else 1, [], [],\n'
          '                      [o.id for o in snap.orders.values()])\n'
          '    return c\n')

MARK_B = ('MARK = "b"\n'
          'def create_controller(cost=None):\n'
          '    def c(snap, view, rng):\n'
          '        return Solution(0 if MARK == "a" else 1, [], [],\n'
          '                      [o.id for o in snap.orders.values()])\n'
          '    return c\n')


def _json_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestCompileController(unittest.TestCase):
    def test_valid_template_passes_probe(self):
        factory = compile_controller(SEED_TEMPLATE)
        ok, fb = probe_controller(factory)
        self.assertTrue(ok, fb)
        ctrl = factory()
        snap, view = evolve._synthetic_snapshot()
        sol = ctrl(snap, view, __import__("random").Random(0))
        self.assertIsInstance(sol, Solution)

    def test_syntax_error_is_program_error(self):
        with self.assertRaises(ProgramError):
            compile_controller(BAD_SYNTAX)

    def test_missing_create_controller_is_program_error(self):
        with self.assertRaises(ProgramError):
            compile_controller(NO_FACTORY)

    def test_wrong_return_type_is_program_error(self):
        factory = compile_controller(WRONG_TYPE)
        with self.assertRaises(ProgramError):
            factory()
        # and the probe surfaces it as a failure, not a crash
        ok, fb = probe_controller(compile_controller(WRONG_TYPE))
        self.assertFalse(ok)

    def test_namespace_isolation_between_programs(self):
        fa = compile_controller(MARK_A)
        fb_ = compile_controller(MARK_B)   # same global name MARK, other value
        rng = __import__("random").Random(0)
        snap, view = evolve._synthetic_snapshot()
        sol_a = fa()(snap, view, rng)
        sol_b = fb_()(snap, view, rng)
        # if namespaces leaked, MARK would have been overwritten by "b"
        self.assertEqual(sol_a.epoch, 0)
        self.assertEqual(sol_b.epoch, 1)


class TestEvaluateProgram(unittest.TestCase):
    def setUp(self):
        clear_eval_cache()

    def test_penalty_on_syntax_error(self):
        res = evaluate_program(BAD_SYNTAX, FIT)
        self.assertEqual(res.fitness, FIT.big_penalty)
        self.assertIn("compile error", res.feedback)
        self.assertFalse(res.cache_hit)

    def test_penalty_on_probe_violation(self):
        src = ("def create_controller(cost=None):\n"
               "    def c(snap, view, rng):\n"
               "        return Solution(0, [], [], [])  # covers nothing (C1)\n"
               "    return c\n")
        res = evaluate_program(src, FIT)
        self.assertEqual(res.fitness, FIT.big_penalty)
        self.assertIn("probe failed", res.feedback)

    def test_valid_program_gets_real_fitness_and_feedback(self):
        res = evaluate_program(SEED_TEMPLATE, FIT)
        self.assertFalse(res.cache_hit)
        self.assertGreater(res.fitness, -1e8)   # not the big penalty
        self.assertTrue(math.isfinite(res.fitness))
        for token in ("mean=", "max=", "inversions=", "served_rate_avg="):
            self.assertIn(token, res.feedback)
        adv = res.audit_summary["adversarial"]["regret"]
        self.assertIsNotNone(adv["max"])
        # fitness == -(1.0*mean + 0.5*max) of the adversarial regret stats
        expect = -(FIT.w_mean * adv["mean"] + FIT.w_max * adv["max"])
        self.assertAlmostEqual(res.fitness, expect, places=9)

    def test_served_rate_floor_penalizes_conservative_program(self):
        # an all-unserved program games regret-vs-weak-oracle; the
        # soft floor must make it measurably worse than its floor-free score
        lazy = ("def create_controller(cost=None):\n"
                "    def c(snap, view, rng):\n"
                "        return Solution(epoch=0, ground_routes=[],"
                " drone_sorties=[], unserved=sorted(snap.orders))\n"
                "    return c\n")
        r_plain = evaluate_program(lazy, FIT)
        r_floor = evaluate_program(lazy, replace(FIT, min_served_rate=0.9))
        self.assertLess(r_floor.fitness, r_plain.fitness - 800.0)
        self.assertIn("served-rate floor violated", r_floor.feedback)

    def test_served_rate_floor_inactive_when_met(self):
        r_plain = evaluate_program(SEED_TEMPLATE, FIT)
        r_floor = evaluate_program(SEED_TEMPLATE,
                                   replace(FIT, min_served_rate=0.3))
        self.assertAlmostEqual(r_plain.fitness, r_floor.fitness, places=9)

    def test_absolute_mode_fitness_formula_and_feedback(self):
        import statistics
        captured = {}
        original = evolve.adversarial_regret_audit

        def spy(*a, **k):
            rep = original(*a, **k)
            captured["rep"] = rep
            return rep

        with mock.patch.object(evolve, "adversarial_regret_audit", spy):
            res = evaluate_program(SEED_TEMPLATE,
                                   replace(FIT, fitness_mode="absolute"))
        self.assertFalse(res.cache_hit)
        js = [r["metrics_heuristic"]["objective"]
              for r in captured["rep"].adversarial]
        expect = -(FIT.w_mean * statistics.fmean(js) + FIT.w_max * max(js))
        self.assertAlmostEqual(res.fitness, expect, places=9)
        self.assertIn("absolute J mean=", res.feedback)
        self.assertIn("nominal_J_mean=", res.feedback)
        self.assertIn("served_rate_avg=", res.feedback)
        # regret-mode wording must not leak into absolute-mode feedback
        self.assertNotIn("regret mean=", res.feedback)

    def test_fitness_mode_cache_isolation(self):
        calls = {"n": 0}
        original = evolve.adversarial_regret_audit

        def counting(*a, **k):
            calls["n"] += 1
            return original(*a, **k)

        with mock.patch.object(evolve, "adversarial_regret_audit", counting):
            evaluate_program(SEED_TEMPLATE, replace(FIT, fitness_mode="regret"))
            self.assertEqual(calls["n"], 1)
            # same source, different mode -> NOT a cache hit (asdict covers
            # fitness_mode), the audit re-runs
            r_abs = evaluate_program(SEED_TEMPLATE,
                                     replace(FIT, fitness_mode="absolute"))
            self.assertEqual(calls["n"], 2)
            r_abs2 = evaluate_program(SEED_TEMPLATE,
                                      replace(FIT, fitness_mode="absolute"))
            self.assertEqual(calls["n"], 2)          # now cached
            self.assertTrue(r_abs2.cache_hit)
            self.assertFalse(r_abs.cache_hit)
            self.assertNotEqual(
                evaluate_program(SEED_TEMPLATE,
                                 replace(FIT, fitness_mode="regret")).fitness,
                r_abs.fitness)

    def test_cache_hit_never_reruns_audit(self):
        calls = {"n": 0}
        original = evolve.adversarial_regret_audit

        def counting(*a, **k):
            calls["n"] += 1
            return original(*a, **k)

        with mock.patch.object(evolve, "adversarial_regret_audit", counting):
            r1 = evaluate_program(SEED_TEMPLATE, FIT)
            self.assertEqual(calls["n"], 1)
            r2 = evaluate_program(SEED_TEMPLATE, FIT)
            self.assertEqual(calls["n"], 1)     # no re-simulation
        self.assertTrue(r2.cache_hit)
        self.assertFalse(r1.cache_hit)
        self.assertEqual(r1.fitness, r2.fitness)
        # distinct cfg -> distinct cache entry
        other = FitnessConfig(params=SMALL, eps={"lam": 0.2}, nominal_seeds=1,
                              n_adversarial=2, base_seed=1,
                              sim_kwargs={"epoch_delta": 30.0, "n_trig": 3})
        with mock.patch.object(evolve, "adversarial_regret_audit", counting):
            evaluate_program(SEED_TEMPLATE, other)
            self.assertEqual(calls["n"], 2)

    def test_mockllm_bad_source_penalized_without_sims(self):
        # MockLLM injected broken source: penalized, and repeat evaluation
        # adds no simulation work at all.
        calls = {"n": 0}
        original = evolve.adversarial_regret_audit

        def counting(*a, **k):
            calls["n"] += 1
            return original(*a, **k)

        client = MockLLM(responses=["```python\n" + BAD_SYNTAX + "\n```"])
        bad = client.chat("s", "u", max_tokens=10, temperature=0.0)
        with mock.patch.object(evolve, "adversarial_regret_audit", counting):
            r1 = evaluate_program(evolve._extract_python_block(bad), FIT)
            r2 = evaluate_program(evolve._extract_python_block(bad), FIT)
        self.assertEqual(r1.fitness, FIT.big_penalty)
        self.assertEqual(r2.fitness, FIT.big_penalty)
        self.assertTrue(r2.cache_hit)
        self.assertEqual(calls["n"], 0)         # compile failed: no sims ever


class TestMockLLM(unittest.TestCase):
    def test_sequential_responses_then_rule_based(self):
        client = MockLLM(responses=["first", "second"], seed=3)
        self.assertEqual(client.chat("s", "u", max_tokens=8,
                                     temperature=0.0), "first")
        self.assertEqual(client.chat("s", "u", max_tokens=8,
                                     temperature=0.0), "second")
        third = client.chat("s", "u", max_tokens=8, temperature=0.0)
        self.assertIn("```python", third)       # rule-based fallback
        self.assertEqual(client.call_count, 3)
        self.assertGreater(client.last_usage["completion_tokens"], 0)

    def test_rule_based_is_deterministic_and_varies_source(self):
        c1, c2 = MockLLM(seed=5), MockLLM(seed=5)
        prompt = "mutate this:\n```python\n" + SEED_TEMPLATE + "\n```"
        r1 = c1.chat("s", prompt, max_tokens=8, temperature=0.0)
        r2 = c2.chat("s", prompt, max_tokens=8, temperature=0.0)
        self.assertEqual(r1, r2)                # deterministic
        self.assertNotEqual(r1, SEED_TEMPLATE)  # but perturbed
        b1 = evolve._extract_python_block(r1)
        compile_controller(b1)                  # still valid Python

    def test_rule_based_without_block_returns_seed_variant(self):
        client = MockLLM(seed=1)
        out = client.chat("s", "no code here", max_tokens=8, temperature=0.0)
        self.assertIn("create_controller", out)


class TestOpenAICompatibleClient(unittest.TestCase):
    def _fake_transport(self, status=200, payload=None):
        calls = {}

        def transport(url, headers, body, timeout):
            calls["url"] = url
            calls["headers"] = headers
            calls["body"] = json.loads(body.decode("utf-8"))
            calls["timeout"] = timeout
            return status, json.dumps(payload if payload is not None else
                                      {"choices": [{"message":
                                                    {"content": "hi"},
                                                    "finish_reason": "stop"}],
                                       "usage": {"prompt_tokens": 3,
                                                 "completion_tokens": 2}})
        return transport, calls

    def test_request_format_and_content_parse(self):
        transport, calls = self._fake_transport()
        client = OpenAICompatibleClient("https://api.example.com/",
                                        "sk-test", "m1", timeout=5.0,
                                        transport=transport)
        out = client.chat("SYS", "USER", max_tokens=128, temperature=0.5)
        self.assertEqual(out, "hi")
        self.assertEqual(calls["url"],
                         "https://api.example.com/chat/completions")
        self.assertEqual(calls["headers"]["Authorization"], "Bearer sk-test")
        body = calls["body"]
        self.assertEqual(body["model"], "m1")
        self.assertEqual(body["max_tokens"], 128)
        self.assertEqual(body["temperature"], 0.5)
        self.assertEqual(body["messages"],
                         [{"role": "system", "content": "SYS"},
                          {"role": "user", "content": "USER"}])
        self.assertEqual(client.last_usage["prompt_tokens"], 3)
        self.assertEqual(client.last_finish, "stop")

    def test_reasoning_content_fallback(self):
        payload = {"choices": [{"message":
                                {"content": "",
                                 "reasoning_content": "the reasoning"},
                                "finish_reason": "stop"}]}
        transport, _ = self._fake_transport(payload=payload)
        client = OpenAICompatibleClient("https://x", "k", "m",
                                        transport=transport)
        self.assertEqual(client.chat("s", "u", max_tokens=8,
                                     temperature=0.0), "the reasoning")

    def test_extra_body_merged_into_request(self):
        transport, calls = self._fake_transport()
        client = OpenAICompatibleClient("https://x", "k", "m",
                                        transport=transport)
        client.chat("s", "u", max_tokens=8, temperature=0.0,
                    extra_body={"effort": "low",
                                "thinking": {"effort": "low"}})
        body = calls["body"]
        self.assertEqual(body["effort"], "low")
        self.assertEqual(body["thinking"], {"effort": "low"})
        # standard fields still intact alongside the extensions
        self.assertEqual(body["model"], "m")
        self.assertEqual(body["max_tokens"], 8)

    def test_extra_body_none_absent_from_request(self):
        transport, calls = self._fake_transport()
        client = OpenAICompatibleClient("https://x", "k", "m",
                                        transport=transport)
        client.chat("s", "u", max_tokens=8, temperature=0.0)
        client.chat("s", "u", max_tokens=8, temperature=0.0, extra_body=None)
        body = calls["body"]        # last call: extra_body=None
        self.assertNotIn("effort", body)
        self.assertNotIn("thinking", body)
        self.assertEqual(set(body),
                         {"model", "messages", "max_tokens", "temperature"})

    def test_http_error_raises_evolution_error(self):
        def transport(url, headers, body, timeout):
            return 500, b'{"error": "boom"}'
        client = OpenAICompatibleClient("https://x", "k", "m",
                                        transport=transport)
        with self.assertRaises(EvolutionError) as cm:
            client.chat("s", "u", max_tokens=8, temperature=0.0)
        self.assertIn("500", str(cm.exception))
        self.assertIn("boom", str(cm.exception))

    def test_from_env_file_priority(self):
        transport, calls = self._fake_transport()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "llm.env")
            with open(path, "w", encoding="utf-8") as f:
                f.write('DAGCD_LLM_BASE_URL="https://from-file"\n'
                        'DAGCD_LLM_API_KEY=file-key\n'
                        'DAGCD_LLM_MODEL=file-model\n')
            # file only
            c = OpenAICompatibleClient.from_env_file(path,
                                                     transport=transport)
            self.assertEqual(c.base_url, "https://from-file")
            self.assertEqual(c.model, "file-model")
            # env beats file
            with mock.patch.dict(os.environ,
                                 {"DAGCD_LLM_BASE_URL": "https://from-env"}):
                c = OpenAICompatibleClient.from_env_file(path,
                                                         transport=transport)
                self.assertEqual(c.base_url, "https://from-env")
                self.assertEqual(c.api_key, "file-key")
            # explicit beats env beats file
            with mock.patch.dict(os.environ,
                                 {"DAGCD_LLM_API_KEY": "env-key"}):
                c = OpenAICompatibleClient.from_env_file(
                    path, api_key="explicit-key", transport=transport)
                self.assertEqual(c.api_key, "explicit-key")
            # missing everything -> EvolutionError
            with self.assertRaises(EvolutionError):
                OpenAICompatibleClient.from_env_file(
                    os.path.join(tmp, "nope.env"), transport=transport)


class TestAnthropicClient(unittest.TestCase):
    def _fake_transport(self, status=200, payload=None):
        calls = {}

        def transport(url, headers, body, timeout):
            calls["url"] = url
            calls["headers"] = headers
            calls["body"] = json.loads(body.decode("utf-8"))
            calls["timeout"] = timeout
            return status, json.dumps(payload if payload is not None else {
                "content": [{"type": "text", "text": "hello "},
                            {"type": "thinking", "thinking": "ignored"},
                            {"type": "text", "text": "world"}],
                "usage": {"input_tokens": 11, "output_tokens": 7},
                "stop_reason": "end_turn",
            })
        return transport, calls

    def test_request_format_headers_and_multi_text_join(self):
        transport, calls = self._fake_transport()
        client = AnthropicClient("https://open.bigmodel.cn/api/anthropic",
                                 "ak-test", "glm-5.3-flash", timeout=9.0,
                                 transport=transport)
        out = client.chat("SYS", "USER", max_tokens=512, temperature=0.3)
        self.assertEqual(out, "hello world")     # thinking block skipped
        self.assertEqual(
            calls["url"],
            "https://open.bigmodel.cn/api/anthropic/v1/messages")
        self.assertEqual(calls["headers"]["x-api-key"], "ak-test")
        self.assertEqual(calls["headers"]["anthropic-version"],
                         "2023-06-01")
        self.assertEqual(calls["headers"]["content-type"],
                         "application/json")
        body = calls["body"]
        self.assertEqual(body["model"], "glm-5.3-flash")
        self.assertEqual(body["max_tokens"], 512)
        self.assertEqual(body["system"], "SYS")
        self.assertEqual(body["messages"],
                         [{"role": "user", "content": "USER"}])
        self.assertEqual(body["temperature"], 0.3)
        # usage mapped to the loop convention; stop_reason surfaced
        self.assertEqual(client.last_usage,
                         {"prompt_tokens": 11, "completion_tokens": 7})
        self.assertEqual(client.last_finish, "end_turn")

    def test_base_url_ending_v1_not_doubled(self):
        transport, calls = self._fake_transport()
        client = AnthropicClient("https://example.com/api/anthropic/v1",
                                 "k", "m", transport=transport)
        client.chat("s", "u", max_tokens=8, temperature=None)
        self.assertEqual(calls["url"],
                         "https://example.com/api/anthropic/v1/messages")
        self.assertNotIn("temperature", calls["body"])   # None -> omitted

    def test_extra_body_rejected(self):
        def transport(url, headers, body, timeout):
            raise AssertionError("transport must not be invoked")

        client = AnthropicClient("https://x/anthropic", "k", "m",
                                 transport=transport)
        with self.assertRaises(EvolutionError) as cm:
            client.chat("s", "u", max_tokens=8, temperature=0.0,
                        extra_body={"effort": "low"})
        self.assertIn("extra_body", str(cm.exception))

    def test_zero_text_blocks_returns_empty(self):
        payload = {"content": [{"type": "thinking", "thinking": "..."}],
                   "usage": {"input_tokens": 3, "output_tokens": 2},
                   "stop_reason": "max_tokens"}
        transport, _ = self._fake_transport(payload=payload)
        client = AnthropicClient("https://x/anthropic", "k", "m",
                                 transport=transport)
        self.assertEqual(client.chat("s", "u", max_tokens=8,
                                     temperature=0.0), "")
        self.assertEqual(client.last_finish, "max_tokens")

    def test_http_error_raises_evolution_error(self):
        def transport(url, headers, body, timeout):
            return 500, b'{"error": "boom"}'
        client = AnthropicClient("https://x/anthropic", "k", "m",
                                 transport=transport)
        with self.assertRaises(EvolutionError) as cm:
            client.chat("s", "u", max_tokens=8, temperature=0.0)
        self.assertIn("500", str(cm.exception))
        self.assertIn("boom", str(cm.exception))


class TestClientFactory(unittest.TestCase):
    def _write_env(self, tmp, base_url):
        path = os.path.join(tmp, "llm.env")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f'DAGCD_LLM_BASE_URL="{base_url}"\n'
                    'DAGCD_LLM_API_KEY=test-key\n'
                    'DAGCD_LLM_MODEL=test-model\n')
        return path

    def test_anthropic_url_selects_anthropic_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_env(tmp,
                                   "https://open.bigmodel.cn/api/anthropic")
            client = client_from_env_file(path, timeout=42.0)
            self.assertIsInstance(client, AnthropicClient)
            self.assertNotIsInstance(client, OpenAICompatibleClient)
            self.assertEqual(client.model, "test-model")
            self.assertEqual(client.api_key, "test-key")
            self.assertEqual(client.timeout, 42.0)
            self.assertEqual(
                client.endpoint_url(),
                "https://open.bigmodel.cn/api/anthropic/v1/messages")

    def test_openai_url_selects_openai_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_env(tmp, "https://api.deepseek.com")
            client = client_from_env_file(path)
            self.assertIsInstance(client, OpenAICompatibleClient)
            self.assertNotIsInstance(client, AnthropicClient)
            self.assertEqual(client.base_url, "https://api.deepseek.com")

    def test_explicit_base_url_beats_env_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write_env(tmp, "https://api.deepseek.com")
            client = client_from_env_file(
                path, base_url="https://open.bigmodel.cn/api/anthropic")
            self.assertIsInstance(client, AnthropicClient)

    def test_missing_config_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(EvolutionError):
                client_from_env_file(os.path.join(tmp, "nope.env"))


class TestEvolutionLoopEndToEnd(unittest.TestCase):
    def setUp(self):
        clear_eval_cache()

    def test_mockllm_small_loop(self):
        client = MockLLM(seed=0)
        cfg = EvolutionConfig(population_size=4, n_generations=2,
                              fitness_cfg=FIT, llm_call_budget=40,
                              archive_dir=None, seed_programs=[SEED_TEMPLATE])
        with tempfile.TemporaryDirectory() as tmp:
            cfg.archive_dir = os.path.join(tmp, "evo")
            result = run_evolution(client, cfg, seed=0)

            self.assertIsInstance(result, EvolutionResult)
            # archives: one file per generation, one row per individual
            for g in (0, 1):
                path = os.path.join(cfg.archive_dir, f"gen_{g:03d}.jsonl")
                self.assertTrue(os.path.exists(path), path)
                rows = _json_lines(path)
                self.assertEqual(len(rows), 4)
                for row in rows:
                    self.assertEqual(
                        row["sha"],
                        hashlib.sha256(row["source"].encode("utf-8"))
                        .hexdigest())
                    self.assertIn("fitness", row)
                    self.assertIn("feedback", row)
                    self.assertIn("cached", row)
            llm_log = _json_lines(os.path.join(cfg.archive_dir,
                                               "llm_calls.jsonl"))
            self.assertEqual(len(llm_log), result.llm_calls)

        # budget respected, calls actually happened
        self.assertGreater(result.llm_calls, 0)
        self.assertLessEqual(result.llm_calls, 40)
        self.assertEqual(client.call_count, result.llm_calls)
        self.assertGreaterEqual(result.prompt_tokens, 1)
        self.assertGreaterEqual(result.completion_tokens, 1)

        # trajectory: recorded per generation; best (real fitness, penalties
        # excluded) monotone non-decreasing; raw mean/worst ordered
        self.assertEqual(len(result.trajectory), 2)
        for row in result.trajectory:
            self.assertEqual(set(row), {"gen", "best", "mean", "worst"})
            self.assertLessEqual(row["worst"], row["mean"])
            self.assertLess(row["best"], FIT.big_penalty)
        best_seq = [r["best"] for r in result.trajectory]
        self.assertEqual(best_seq, sorted(best_seq))

        # best program: compiles, probes, and survives a full simulation
        self.assertIsNotNone(result.best_source)
        factory = compile_controller(result.best_source)
        ok, fb = probe_controller(factory)
        self.assertTrue(ok, fb)
        sc = sample_scenario(SMALL, seed=5)
        rep = run_simulation(sc, factory(), seed=0)
        self.assertGreaterEqual(rep.metrics["served_rate"], 0.0)
        self.assertIn("objective", rep.metrics)
        self.assertTrue(any("stopped after" in n for n in result.notes))

    def test_probe_failing_program_never_becomes_best(self):
        # compiles but violates C1 on the probe snapshot -> big penalty;
        # the penalty must not win the generation max() (regression: a
        # positive big_penalty otherwise dominates every feasible fitness).
        bad = ("def create_controller(cost=None):\n"
               "    def c(snap, view, rng):\n"
               "        return Solution(0, [], [], [])  # covers nothing\n"
               "    return c\n")
        client = MockLLM(responses=["```python\n" + bad + "\n```"], seed=0)
        cfg = EvolutionConfig(population_size=2, n_generations=1,
                              fitness_cfg=FIT, llm_call_budget=10,
                              archive_dir=None)
        result = run_evolution(client, cfg, seed=0)
        self.assertLess(result.best_fitness, FIT.big_penalty)
        self.assertIn("create_controller", result.best_source)
        self.assertNotEqual(result.best_source.strip(), bad.strip())

    def test_seed_template_evolution_is_deterministic(self):
        cfg = EvolutionConfig(population_size=3, n_generations=1,
                              fitness_cfg=FIT, llm_call_budget=10,
                              archive_dir=None)
        r1 = run_evolution(MockLLM(seed=0), cfg, seed=0)
        r2 = run_evolution(MockLLM(seed=0), cfg, seed=0)
        self.assertEqual(r1.trajectory, r2.trajectory)
        self.assertEqual(r1.best_fitness, r2.best_fitness)
        self.assertEqual(r1.best_source, r2.best_source)
        self.assertEqual(r1.llm_calls, r2.llm_calls)


if __name__ == "__main__":
    unittest.main()

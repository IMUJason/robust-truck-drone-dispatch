"""LLM evolutionary outer loop for DAGCD (EoH-style, self-contained).

This module implements the heuristic-evolution loop: an
Evolution-of-Heuristics-style loop that
evolves dispatch *programs* (controller source code) against the adversarial
regret audit of ``dagcd.audit`` (docs/SPECIFICATION.md Sec. 5.1).  A program is a Python
source string defining ``create_controller(cost=None)``; the loop compiles it,
probes it on a synthetic snapshot, scores it with
``adversarial_regret_audit`` (fitness = -(w_mean*mean + w_max*max) of the
adversarial regret statistics), archives every individual, and asks an LLM
for mutations with SDFF-style failure feedback injected into the prompt.

The prompting scheme follows the EoH/OpenEvolve conventions (parent source +
failure feedback + operator description + "output exactly one fenced python
block"); the loop itself is deliberately self-contained (stdlib only) instead
of a hard dependency on an external evolution-framework installation.

Invariants (enforced by construction; asserted by tests/test_evolve.py)
-----------------------------------------------------------------------
1. Standard library only.  HTTP goes through ``urllib.request``; the
   transports of :class:`OpenAICompatibleClient` and :class:`AnthropicClient`
   are injectable, and the unit tests never touch the network (MockLLM /
   fake transports only).
2. Fitness cache.  ``evaluate_program`` memoizes on
   ``sha256(source + json(fitness_cfg.asdict(), sort_keys=True))``; a cache
   hit never re-runs simulations and never counts LLM calls (the returned
   ``EvalResult.cache_hit`` is True).
3. Archive completeness.  Every individual of every generation — parents and
   offspring alike — is archived to ``archive_dir/gen_{g:03d}.jsonl`` with
   ``{source, fitness, feedback, sha, cached}``; every LLM call (usage,
   finish_reason, success/failure) is archived to ``llm_calls.jsonl``.
4. Namespace isolation.  Each compiled program gets a fresh globals dict with
   the injected dagcd symbols; two programs defining the same module-level
   name cannot pollute each other.
5. SDFF closure.  Compile/probe/audit failures map to
   ``fitness_cfg.big_penalty`` and the failure summary re-enters the next
   generation's mutation prompts, so the population learns from broken code
   instead of silently discarding it.  Because big_penalty (positive) would
   otherwise dominate the max() over feasible programs (fitness <= 0),
   penalty individuals are (a) kept out of the population by an
   accept-at-birth gate (compile + protocol probe before an offspring slot
   accepts a candidate) and (b) ranked below every feasible program in
   elite/parent/best selection.
6. Budget and plateau guards.  ``llm_call_budget`` caps LLM invocations and
   ``patience`` caps generations without best-fitness improvement; both stops
   are recorded in ``EvolutionResult.notes``.
7. Reproducibility.  Given a deterministic client, same (config, seed)
   reproduces the same archives and trajectory.  Real LLMs are not
   deterministic — Invariant 3 archives everything (source/usage/
   finish_reason) so runs stay auditable anyway.
"""

from __future__ import annotations

import abc
import hashlib
import json
import math
import os
import random
import re
import socket
import statistics
import urllib.error
import urllib.request
from urllib.parse import urljoin
from dataclasses import dataclass, field, is_dataclass, asdict as _dc_asdict

from .instance import (Instance, Solution, Node, Order, GroundVehicle, Drone,
                       WeatherEvent, GroundRoute, GroundStop, DroneSortie,
                       DroneStop, TOL)
from .checker import CostParams, check_solution
from .generator import InstanceParams
from .simulator import (FleetView, VehicleView, DroneView, _propagate_ground,
                        _leg_blocked)
from .oracle import InsertionOracle
from .audit import adversarial_regret_audit

__all__ = ["EvolutionError", "ProgramError", "LLMClient",
           "OpenAICompatibleClient", "AnthropicClient", "client_from_env_file",
           "MockLLM", "compile_controller",
           "probe_controller", "FitnessConfig", "EvalResult",
           "evaluate_program", "clear_eval_cache", "EvolutionConfig",
           "EvolutionResult", "run_evolution", "SEED_TEMPLATE",
           "DEFAULT_MUTATION_OPERATORS"]


# =====================================================================
# errors
# =====================================================================

class EvolutionError(Exception):
    """LLM/HTTP-level failure of the evolution outer loop."""


class ProgramError(EvolutionError):
    """The evolved source failed to compile or violates the controller protocol."""


# =====================================================================
# LLM client abstraction
# =====================================================================

class LLMClient(abc.ABC):
    """Chat-completion abstraction used by ``run_evolution``.

    Implementations must return the assistant text.  Implementations that can
    report usage SHOULD set ``self.last_usage`` (dict with prompt_tokens /
    completion_tokens) and ``self.last_finish`` (finish_reason) after each
    call so the loop can archive them (Invariant 3).
    """

    @abc.abstractmethod
    def chat(self, system: str, user: str, *, max_tokens: int,
             temperature: float, extra_body: dict | None = None) -> str:
        """Return the assistant reply for a (system, user) message pair.

        ``extra_body`` (OpenAI-compatible extension) is merged into the
        request JSON top level when given — provider-private fields such as
        DeepSeek/Kimi ``effort`` or GLM thinking controls.
        """


def _urllib_transport(url: str, headers: dict, body: bytes,
                      timeout: float) -> tuple[int, bytes]:
    """Default transport: POST via urllib.request (Invariant 1).

    Returns (status, body_bytes); HTTP errors are returned as statuses, not
    raised, so the client can build a uniform EvolutionError.
    """
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (TimeoutError, socket.timeout) as e:
        # socket.timeout is a plain OSError on CPython < 3.10 (only aliased
        # to TimeoutError from 3.10 on) — catch both so every timeout
        # surfaces as an EvolutionError, never a raw socket exception.
        raise EvolutionError(f"LLM request timed out after {timeout}s") from e
    except urllib.error.URLError as e:
        reason = getattr(e, "reason", e)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise EvolutionError(
                f"LLM request timed out after {timeout}s") from e
        raise EvolutionError(f"LLM request failed: {e}") from e


class OpenAICompatibleClient(LLMClient):
    """OpenAI-compatible chat-completions client (stdlib urllib only).

    POSTs {base_url}/chat/completions with {"model", "messages",
    "max_tokens", "temperature"} plus any ``extra_body`` fields merged into
    the top level.  Parses choices[0].message.content; when
    content is empty and reasoning_content is present the reasoning text is
    used instead (DeepSeek reasoner-style responses).  Non-200 statuses and
    malformed payloads raise EvolutionError with the status code / a response
    summary.  ``transport`` is injectable for tests (see Invariant 1).
    """

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 120.0, transport=None):
        if not base_url or not base_url.startswith(("http://", "https://")):
            raise EvolutionError(f"invalid base_url {base_url!r}")
        if not api_key:
            raise EvolutionError("api_key is required")
        if not model:
            raise EvolutionError("model is required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = float(timeout)
        self.transport = transport or _urllib_transport
        self.last_usage: dict = {}
        self.last_finish: str | None = None

    def chat(self, system: str, user: str, *, max_tokens: int,
             temperature: float, extra_body: dict | None = None) -> str:
        url = self.base_url + "/chat/completions"
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra_body:
            payload.update(extra_body)
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json",
                   "Authorization": f"Bearer {self.api_key}"}
        status, raw = self.transport(url, headers, body, self.timeout)
        if status != 200:
            snippet = raw[:400].decode("utf-8", "replace") \
                if isinstance(raw, (bytes, bytearray)) else str(raw)[:400]
            raise EvolutionError(
                f"LLM request failed: HTTP {status}: {snippet}")
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as e:
            raise EvolutionError(
                f"LLM response is not valid JSON: {e}: {str(raw)[:200]}") from e
        try:
            choice = data["choices"][0]
            message = choice.get("message") or {}
        except (KeyError, IndexError, TypeError) as e:
            raise EvolutionError(
                f"LLM response missing choices: {str(raw)[:200]}") from e
        content = message.get("content")
        if not content or not str(content).strip():
            # DeepSeek reasoner models put the answer in reasoning_content
            # when content is empty.
            content = message.get("reasoning_content") or ""
        self.last_usage = data.get("usage") or {}
        self.last_finish = choice.get("finish_reason")
        return str(content)

    @classmethod
    def from_env_file(cls, path: str | None = None,
                      env_prefix: str = "DAGCD_LLM_", *,
                      base_url: str | None = None,
                      api_key: str | None = None,
                      model: str | None = None,
                      timeout: float = 120.0, transport=None
                      ) -> "OpenAICompatibleClient":
        """Build a client with priority: explicit args > environment > env file.

        ``path`` defaults to ``~/.dagcd/llm_key.env`` (chmod 600, simple
        KEY=VALUE lines).  Variable names are ``{env_prefix}BASE_URL``,
        ``{env_prefix}API_KEY``, ``{env_prefix}MODEL``.  With the default
        prefix this is ``DAGCD_LLM_*``; the smoke runner uses prefix
        ``DEEPSEEK_`` against the same file layout.  See also
        :func:`client_from_env_file` for protocol-dispatching construction.
        """
        bu, key, mdl = _resolve_llm_creds(path, env_prefix, base_url, api_key,
                                          model)
        return cls(base_url=bu, api_key=key, model=mdl, timeout=timeout,
                   transport=transport)


class AnthropicClient(LLMClient):
    """Anthropic Messages-protocol client (stdlib urllib only).

    Speaks the Anthropic Messages API — the GLM Coding Plan endpoint
    (https://open.bigmodel.cn/api/anthropic) and genuine Anthropic alike.
    POSTs ``{base_url}/v1/messages``: a base_url already ending in "/v1" is
    used as-is (``.../v1/messages``, joined with :func:`urllib.parse.urljoin`
    so no segment is duplicated); otherwise "/v1/messages" is appended.
    Headers: ``x-api-key`` + ``anthropic-version: 2023-06-01``.  Body:
    ``{model, max_tokens, system, messages: [{role, content}]}`` with
    ``temperature`` added only when not None (the API accepts 0..1).

    ``extra_body`` is deliberately unsupported: the Anthropic API rejects
    unknown body fields, so silently forwarding OpenAI-style extensions
    (``effort``, ``thinking``, ...) would fail every request — ``chat``
    raises EvolutionError when a non-empty extra_body is passed.  Use
    :class:`OpenAICompatibleClient` for providers that accept extensions.

    Response handling: joins the ``text`` of every content block with
    ``type == "text"`` (thinking/signature blocks are skipped).  When the
    response carries zero text blocks the result is "" — the evolution
    loop's parse gate (``_extract_python_block``) then treats it as an
    unparseable response and retries/falls back.  Non-200 statuses and
    malformed payloads raise EvolutionError with the status code / a
    response summary, mirroring :class:`OpenAICompatibleClient`.
    ``transport`` is injectable for tests (Invariant 1).
    """

    API_VERSION = "2023-06-01"

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 120.0, transport=None):
        if not base_url or not base_url.startswith(("http://", "https://")):
            raise EvolutionError(f"invalid base_url {base_url!r}")
        if not api_key:
            raise EvolutionError("api_key is required")
        if not model:
            raise EvolutionError("model is required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = float(timeout)
        self.transport = transport or _urllib_transport
        self.last_usage: dict = {}
        self.last_finish: str | None = None

    def endpoint_url(self) -> str:
        """Resolved messages URL (base_url ending in /v1 is not re-appended)."""
        if self.base_url.endswith("/v1"):
            return urljoin(self.base_url + "/", "messages")
        return urljoin(self.base_url + "/", "v1/messages")

    def chat(self, system: str, user: str, *, max_tokens: int,
             temperature: float | None, extra_body: dict | None = None) -> str:
        if extra_body:
            raise EvolutionError(
                "AnthropicClient does not support extra_body — the "
                "Anthropic API rejects unknown body fields (drop "
                "llm_extra_body, or use OpenAICompatibleClient for "
                f"OpenAI-style extensions): {sorted(extra_body)}")
        body = {"model": self.model,
                "max_tokens": max_tokens,
                "system": system,
                "messages": [{"role": "user", "content": user}]}
        if temperature is not None:
            body["temperature"] = temperature
        headers = {"content-type": "application/json",
                   "x-api-key": self.api_key,
                   "anthropic-version": self.API_VERSION}
        status, raw = self.transport(self.endpoint_url(), headers,
                                     json.dumps(body).encode("utf-8"),
                                     self.timeout)
        if status != 200:
            snippet = raw[:400].decode("utf-8", "replace") \
                if isinstance(raw, (bytes, bytearray)) else str(raw)[:400]
            raise EvolutionError(
                f"LLM request failed: HTTP {status}: {snippet}")
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError) as e:
            raise EvolutionError(
                f"LLM response is not valid JSON: {e}: {str(raw)[:200]}") from e
        if not isinstance(data, dict):
            raise EvolutionError(
                f"LLM response is not an object: {str(raw)[:200]}")
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks
                       if isinstance(b, dict) and b.get("type") == "text")
        usage = data.get("usage") or {}
        # anthropic field names -> loop convention (LLMClient contract)
        self.last_usage = {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
        }
        self.last_finish = data.get("stop_reason")
        return text


def _parse_env_file(path: str) -> dict:
    """Parse KEY=VALUE lines (comments/blank lines skipped, quotes stripped)."""
    vals = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
    return vals


def _resolve_llm_creds(path: str | None, env_prefix: str,
                       base_url: str | None, api_key: str | None,
                       model: str | None) -> tuple[str, str, str]:
    """Resolve (BASE_URL, API_KEY, MODEL): explicit args > environment > env
    file.  Shared by :meth:`OpenAICompatibleClient.from_env_file` and
    :func:`client_from_env_file`."""
    if path is None:
        path = os.path.join(os.path.expanduser("~"), ".dagcd",
                            "llm_key.env")
    else:
        path = os.path.expanduser(path)
    file_vals = _parse_env_file(path) if os.path.exists(path) else {}
    explicit = {"BASE_URL": base_url, "API_KEY": api_key, "MODEL": model}

    def pick(name: str) -> str | None:
        if explicit[name]:
            return explicit[name]
        env_name = env_prefix + name
        if os.environ.get(env_name):
            return os.environ[env_name]
        return file_vals.get(env_name)

    bu, key, mdl = pick("BASE_URL"), pick("API_KEY"), pick("MODEL")
    missing = [n for n, v in (("BASE_URL", bu), ("API_KEY", key),
                              ("MODEL", mdl)) if not v]
    if missing:
        raise EvolutionError(
            f"missing LLM configuration for {missing}: pass explicit "
            f"arguments, set {env_prefix}* environment variables, or "
            f"provide an env file at {path}")
    return bu, key, mdl


def client_from_env_file(path: str | None = None,
                         env_prefix: str = "DAGCD_LLM_", *,
                         timeout: float = 120.0, transport=None,
                         base_url: str | None = None,
                         api_key: str | None = None,
                         model: str | None = None) -> LLMClient:
    """Protocol-dispatching client factory (explicit > environment > env file).

    Base URLs containing ``/anthropic`` (e.g. the GLM Coding Plan endpoint
    ``https://open.bigmodel.cn/api/anthropic``) speak the Anthropic Messages
    protocol and get an :class:`AnthropicClient`; anything else is treated
    as OpenAI-compatible (:class:`OpenAICompatibleClient`).  ``timeout`` and
    the injectable ``transport`` are forwarded to the chosen client.
    """
    bu, key, mdl = _resolve_llm_creds(path, env_prefix, base_url, api_key,
                                      model)
    if "/anthropic" in bu:
        return AnthropicClient(bu, key, mdl, timeout=timeout,
                               transport=transport)
    return OpenAICompatibleClient(bu, key, mdl, timeout=timeout,
                                  transport=transport)


_BLOCK_RE = re.compile(r"```python[ \t]*\n(.*?)```", re.DOTALL)
_NUMERIC_RE = re.compile(r"(?<![A-Za-z0-9_.])(\d+\.\d+|\d+)(?![A-Za-z0-9_.])")
_FACTORS = (0.5, 0.75, 1.25, 1.5, 2.0)


def _extract_python_block(text: str | None) -> str | None:
    """Return the last fenced ```python block of ``text`` (stripped), or None."""
    if not text:
        return None
    blocks = _BLOCK_RE.findall(text)
    if not blocks:
        return None
    cand = blocks[-1].strip("\n")
    return cand if cand.strip() else None


def _perturb_source(source: str, rng: random.Random) -> str:
    """Deterministically perturb one numeric literal of ``source``.

    Used by MockLLM's rule-based generation and by the loop's deterministic
    fallback fill: renames nothing (syntax stays valid) and tweaks exactly
    one literal, so offspring are near-copies of their parent.  If no bare
    numeric literal exists, a harmless marker comment is appended.
    """
    matches = list(_NUMERIC_RE.finditer(source))
    if matches:
        m = matches[rng.randrange(len(matches))]
        lit = m.group(1)
        if "." in lit:
            rep = f"{float(lit) * rng.choice(_FACTORS):.4f}"
        else:
            rep = str(max(0, int(lit) + rng.choice((-1, 1))))
        return source[:m.start(1)] + rep + source[m.end(1):]
    return source + f"\n\n# mutation tag {rng.random():.6f}\n"


class MockLLM(LLMClient):
    """Deterministic LLM stand-in for tests (no network, Invariant 1).

    With ``responses`` given, replies are returned in order; once exhausted
    (or without ``responses``) a rule-based generator kicks in: the last
    fenced python block of the prompt is perturbed via
    :func:`_perturb_source` (falling back to SEED_TEMPLATE when the prompt
    has no code block), so a population genuinely "evolves" in tests.
    ``call_count`` records invocations; token usage is estimated (words).
    """

    def __init__(self, responses: list | None = None, seed: int = 0):
        self._responses = list(responses) if responses is not None else None
        self._idx = 0
        self._seed = seed
        self.call_count = 0
        self.last_usage: dict = {}
        self.last_finish: str | None = "stop"

    def chat(self, system: str, user: str, *, max_tokens: int,
             temperature: float, extra_body: dict | None = None) -> str:
        self.call_count += 1
        if self._responses is not None and self._idx < len(self._responses):
            text = self._responses[self._idx]
            self._idx += 1
        else:
            r = random.Random(f"{self._seed}:{self.call_count}")
            block = _extract_python_block(user)
            cand = _perturb_source(block, r) if block is not None \
                else SEED_TEMPLATE
            text = f"```python\n{cand}\n```"
        self.last_usage = {
            "prompt_tokens": len((system + " " + user).split()),
            "completion_tokens": len(text.split()),
        }
        return text


# =====================================================================
# program compilation & protocol probe
# =====================================================================

def _base_namespace() -> dict:
    """Fresh globals dict for one evolved program (Invariant 4)."""
    return {
        "__name__": "evolved_program",
        "Instance": Instance, "Solution": Solution,
        "GroundRoute": GroundRoute, "GroundStop": GroundStop,
        "DroneSortie": DroneSortie, "DroneStop": DroneStop,
        "TOL": TOL, "CostParams": CostParams,
        "check_solution": check_solution, "FleetView": FleetView,
        "_propagate_ground": _propagate_ground, "_leg_blocked": _leg_blocked,
        "math": math, "random": random,
    }


def compile_controller(source: str):
    """Compile an evolved program into a controller factory.

    ``source`` is executed in a fresh namespace pre-injected with the dagcd
    public symbols (Instance/Solution/route/sortie/stop dataclasses, TOL,
    CostParams, check_solution, FleetView, _propagate_ground, _leg_blocked,
    math, random — Invariant 4) and must define
    ``create_controller(cost=None)`` returning a callable
    ``(snap, view, rng) -> Solution``.  Returns a zero-argument
    ``factory(cost=None)`` producing fresh controller instances (audit
    Invariant 3).  Syntax errors, missing ``create_controller`` and wrong
    return types raise ProgramError.
    """
    if not isinstance(source, str) or not source.strip():
        raise ProgramError("empty program source")
    namespace = _base_namespace()
    try:
        code = compile(source, "<evolved_program>", "exec")
    except SyntaxError as e:
        raise ProgramError(f"syntax error: {e}") from e
    try:
        exec(code, namespace)
    except Exception as e:
        raise ProgramError(f"exec failed: {type(e).__name__}: {e}") from e
    fn = namespace.get("create_controller")
    if fn is None:
        raise ProgramError("source must define create_controller(cost=None)")
    if not callable(fn):
        raise ProgramError("create_controller is not callable")

    def factory(cost=None):
        ctrl = fn(cost)
        if not callable(ctrl):
            raise ProgramError(
                f"create_controller returned {type(ctrl).__name__}, expected "
                "a callable (snap, view, rng) -> Solution")
        return ctrl

    factory.source = source
    return factory


def _synthetic_snapshot() -> tuple[Instance, FleetView]:
    """Tiny hand-built (snap, view) pair for the protocol probe.

    Fully deterministic (no sampling): one depot, two vertiports, two
    released orders, one vehicle, one drone, clear weather, no NFZ.
    """
    nodes = {0: Node(0, 0.0, 0.0, "depot"),
             1: Node(1, 3000.0, 0.0, "vertiport"),
             2: Node(2, -3000.0, 0.0, "vertiport"),
             3: Node(3, 800.0, 600.0, "customer"),
             4: Node(4, -800.0, -600.0, "customer")}
    orders = {"o1": Order("o1", 3, 0.0, 1.0, 1.0, 0.0, 60.0),
              "o2": Order("o2", 4, 0.0, 1.2, 1.0, 0.0, 60.0)}
    vehicles = {"gv0": GroundVehicle("gv0", 40.0, 0, 0.0, 240.0)}
    drones = {"da0": Drone("da0", 5.0, 500.0, 600.0, 60.0, 3)}
    snap = Instance(nodes=nodes, orders=orders, vehicles=vehicles,
                    drones=drones, nfzs=[],
                    weather=[WeatherEvent(0.0, 240.0, "clear")],
                    horizon=240.0, congestion=[(0.0, 240.0, 1.0)])
    view = FleetView(vehicles={"gv0": VehicleView(0, 0.0, (), False)},
                     drones={"da0": DroneView(1, 0.0, None, 500.0)},
                     tau=0.0, sim_end=240.0)
    return snap, view


def probe_controller(factory) -> tuple[bool, str]:
    """Protocol probe: one controller call on a synthetic snapshot + check_solution.

    Returns (True, "") on success; (False, feedback) when instantiation or
    the call raises, the return value is not a Solution, or check_solution
    reports violations.  The feedback string is what SDFF re-injects into
    mutation prompts (Invariant 5).
    """
    try:
        ctrl = factory()
        if not callable(ctrl):
            return False, f"factory returned non-callable {type(ctrl).__name__}"
        snap, view = _synthetic_snapshot()
        sol = ctrl(snap, view, random.Random(0))
        if not isinstance(sol, Solution):
            return False, (f"controller returned {type(sol).__name__}, "
                           "expected Solution")
        rep = check_solution(snap, sol)
        if rep.violations:
            return False, "violations: " + "; ".join(rep.violations[:5])
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# =====================================================================
# fitness
# =====================================================================

def _jsonable(obj):
    """Deep-convert dataclasses to JSON-safe structures (cache keys)."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _jsonable(_dc_asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return repr(obj)


@dataclass
class FitnessConfig:
    """Configuration of one program's adversarial-regret fitness.

    Defaults are deliberately small so tests and smoke runs stay cheap; the
    audit protocol parameters mirror ``adversarial_regret_audit``.
    ``oracle_factory`` defaults to InsertionOracle (deterministic); sim
    timing fields are scrubbed by the audit harness as usual.

    ``min_served_rate`` (optional soft floor, default None = disabled): the
    mean ``served_rate_heuristic`` over the adversarial audit segment must
    stay at or above this value, otherwise a soft penalty is subtracted from
    the fitness (see ``evaluate_program``).  Motivation: with regret against
    a weak oracle as the raw objective, a pathologically conservative
    controller that declares most orders unserved pays regret almost nothing
    (the weak oracle also leaves orders unserved) while its real served rate
    collapses — a reward-hacking failure mode of regret objectives.  The
    floor prices that collapse back into the fitness without hard-rejecting
    borderline programs.

    ``fitness_mode``: "regret" (default) scores fitness = -(w_mean*mean +
    w_max*max) over the adversarial regret statistics; "absolute" (ablation
    arm (a) — isolate the regret objective from the absolute-cost objective)
    scores fitness = -(w_mean*mean_J + w_max*max_J) over the adversarial
    absolute objectives (``metrics_heuristic["objective"]`` of the SAME audit
    call; the nominal segment's J is recorded in the feedback only).
    """

    params: InstanceParams = field(default_factory=lambda: InstanceParams(
        sim_duration=240.0, lam_peak=1.0, n_clusters=2, n_vehicles=2,
        n_drones=1, n_vertiports=2))
    eps: dict = field(default_factory=lambda: {"lam": 0.3, "cong": 0.2})
    nominal_seeds: int = 2
    n_adversarial: int = 4
    base_seed: int = 0
    sim_kwargs: dict | None = None
    w_mean: float = 1.0
    w_max: float = 0.5
    big_penalty: float = 1e9
    min_served_rate: float | None = None   # soft served-rate floor (None = off)
    fitness_mode: str = "regret"           # "regret" | "absolute"
    oracle_factory: object = None          # None -> InsertionOracle()

    def resolved_oracle_factory(self):
        if self.oracle_factory is not None:
            return self.oracle_factory
        return lambda: InsertionOracle()

    def asdict(self) -> dict:
        """JSON-safe config snapshot (cache key; excludes the callable)."""
        return {
            "params": _jsonable(self.params),
            "eps": _jsonable(self.eps),
            "nominal_seeds": self.nominal_seeds,
            "n_adversarial": self.n_adversarial,
            "base_seed": self.base_seed,
            "sim_kwargs": _jsonable(self.sim_kwargs or {}),
            "w_mean": self.w_mean,
            "w_max": self.w_max,
            "big_penalty": self.big_penalty,
            "min_served_rate": self.min_served_rate,
            "fitness_mode": self.fitness_mode,
            "oracle": "custom" if self.oracle_factory is not None
                      else "insertion",
        }


@dataclass
class EvalResult:
    """Outcome of one program evaluation (fitness + feedback + summary)."""

    fitness: float
    feedback: str
    audit_summary: dict
    cache_hit: bool = False


_EVAL_CACHE: dict = {}


def clear_eval_cache() -> None:
    """Drop all memoized evaluations (test hygiene; Invariant 2)."""
    _EVAL_CACHE.clear()


def _cache_key(source: str, cfg: FitnessConfig) -> str:
    payload = source + json.dumps(cfg.asdict(), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate_program(source: str, fitness_cfg: FitnessConfig) -> EvalResult:
    """Evaluate one evolved program -> EvalResult (fitness, feedback, summary).

    Pipeline: compile (ProgramError -> big_penalty), probe (fail ->
    big_penalty), then ``adversarial_regret_audit`` with
    ``fitness_cfg``'s params/eps/sizes/base_seed/sim_kwargs and oracle.
    fitness depends on ``fitness_cfg.fitness_mode``: "regret" (default)
    computes -(w_mean*mean + w_max*max) over the adversarial regret
    statistics; "absolute" (ablation arm (a)) computes -(w_mean*mean_J +
    w_max*max_J) over the adversarial absolute objectives of the SAME audit
    call, reporting the nominal segment's J mean in the feedback.  In both
    modes the feedback carries the headline statistics, the inversion count
    and the average served rate for the SDFF prompt injection.  Results are
    memoized on sha256(source + json(cfg)) — a cache hit never re-runs the
    audit (Invariant 2) and is flagged with ``cache_hit=True``; the cache
    key covers fitness_mode, so the two ablation arms never collide.
    """
    if fitness_cfg.fitness_mode not in ("regret", "absolute"):
        raise ValueError(f"unknown fitness_mode {fitness_cfg.fitness_mode!r};"
                         " expected 'regret' or 'absolute'")
    key = _cache_key(source, fitness_cfg)
    hit = _EVAL_CACHE.get(key)
    if hit is not None:
        return EvalResult(fitness=hit.fitness, feedback=hit.feedback,
                          audit_summary=hit.audit_summary, cache_hit=True)
    penalty = fitness_cfg.big_penalty
    try:
        factory = compile_controller(source)
    except ProgramError as e:
        res = EvalResult(penalty, f"compile error: {e}", {})
        _EVAL_CACHE[key] = res
        return res
    ok, fb = probe_controller(factory)
    if not ok:
        res = EvalResult(penalty, f"probe failed: {fb}", {})
        _EVAL_CACHE[key] = res
        return res
    try:
        report = adversarial_regret_audit(
            fitness_cfg.params, factory,
            fitness_cfg.resolved_oracle_factory(),
            eps=fitness_cfg.eps, n_adversarial=fitness_cfg.n_adversarial,
            nominal_seeds=fitness_cfg.nominal_seeds,
            base_seed=fitness_cfg.base_seed,
            sim_kwargs=fitness_cfg.sim_kwargs)
    except Exception as e:
        res = EvalResult(penalty,
                         f"audit error: {type(e).__name__}: {e}", {})
        _EVAL_CACHE[key] = res
        return res
    s = report.summary
    n_inv = len(s["inversions"])
    sr = statistics.fmean(r["served_rate_heuristic"]
                          for r in report.adversarial) if report.adversarial \
        else 0.0
    if fitness_cfg.fitness_mode == "absolute":
        # ablation arm (a): same audit call, absolute-J objective.  The
        # adversarial segment drives the fitness; nominal J is recorded.
        js = [r["metrics_heuristic"]["objective"] for r in report.adversarial]
        if not js:
            raise ValueError("absolute fitness_mode needs >=1 adversarial "
                             "sample")
        mean_j, max_j = statistics.fmean(js), max(js)
        fitness = -(fitness_cfg.w_mean * mean_j
                    + fitness_cfg.w_max * max_j)
        head = (f"absolute J mean={mean_j:.4f} max={max_j:.4f} "
                f"inversions={n_inv}")
        nom_js = [r["J_heuristic"] for r in report.nominal]
        if nom_js:
            head += f" nominal_J_mean={statistics.fmean(nom_js):.4f}"
    else:  # "regret"
        adv = s["adversarial"]["regret"]
        mean, mx = adv["mean"], adv["max"]
        q95 = adv["q95"] if adv["q95"] is not None else float("nan")
        fitness = -(fitness_cfg.w_mean * mean + fitness_cfg.w_max * mx)
        head = (f"adversarial regret mean={mean:.4f} max={mx:.4f} "
                f"q95={q95:.4f} inversions={n_inv}")
    floor_note = ""
    if fitness_cfg.min_served_rate is not None \
            and sr < fitness_cfg.min_served_rate - 1e-9:
        # soft served-rate floor: a conservative program that games the weak
        # oracle by refusing orders pays for its collapsed served rate
        fitness -= 1000.0 * (fitness_cfg.min_served_rate - sr)
        floor_note = (f" served-rate floor violated: {sr:.3f} < "
                      f"{fitness_cfg.min_served_rate:.2f}")
    feedback = head + f" served_rate_avg={sr:.3f}" + floor_note
    res = EvalResult(fitness, feedback, s)
    _EVAL_CACHE[key] = res
    return res


# =====================================================================
# seed template
# =====================================================================

SEED_TEMPLATE = '''"""Greedy nearest-dispatch controller — seed program for evolution.

Semantics equivalent to dagcd.simulator.GreedyController (docs/SPECIFICATION.md
Sec. 7 baseline (i)): for every released order compare (a) a single-parcel drone
sortie launched from the drone's chain position and (b) the cheapest
insertion into the nearest ground vehicle with spare capacity; the cheaper
marginal-cost option wins. Orders feasible by neither are declared unserved
in this epoch's solution and retried at later epochs. Claimed arrival times
are earliest-feasible under the snapshot's tau_g/tau_a.
"""


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
                energy = (snap.energy_alpha + snap.energy_beta * o.weight) \
                    * d1 * snap.energy_mu.get(wx1, 1.0) \
                    + snap.energy_alpha * d2 * snap.energy_mu.get(wx2, 1.0)
                if energy > plan_energy[did] + TOL:
                    continue
                u = max(0.0, max(arr, o.e) - o.l)
                c = cost.fixed_sortie \
                    + cost.var_drone_per_km * (d1 + d2) / 1000.0 \
                    + u * o.lateness_penalty
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
            for vid in sorted(view.vehicles,
                              key=lambda x: (snap.dist(
                                  view.vehicles[x].node, o.node), x)):
                veh = snap.vehicles[vid]
                if load[vid] + o.weight > veh.capacity + TOL:
                    continue
                base = routes[vid]
                cur = _propagate_ground(snap, veh, base, T, depots,
                                        enforce_horizon=True) if base else None
                cur_late = cur[3] if cur else 0.0
                cur_km = cur[4] if cur else 0.0
                best_pos = None
                for pos in range(prefix[vid], len(base) + 1):
                    seq = base[:pos] + [o.id] + base[pos:]
                    prop = _propagate_ground(snap, veh, seq, T, depots,
                                             enforce_horizon=True)
                    if prop is None:
                        continue
                    delta = cost.var_ground_per_km * (prop[4] - cur_km) \
                        + (prop[3] - cur_late)
                    if not base and not view.vehicles[vid].used:
                        delta += cost.fixed_ground
                    if best_pos is None or delta < best_pos[0]:
                        best_pos = (delta, pos)
                if best_pos is None:
                    return None
                delta, pos = best_pos

                def apply(vid=vid, pos=pos, oid=o.id):
                    routes[vid].insert(pos, oid)
                    load[vid] += snap.orders[oid].weight
                return (delta, GROUND, apply)
            return None

        for o in sorted(snap.orders.values(),
                        key=lambda x: (x.e, x.release, x.id)):
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
'''


# =====================================================================
# prompts
# =====================================================================

_SYSTEM_PROMPT = """You are evolving a dispatch controller for DAGCD (dynamic
truck-drone delivery dispatch; the controller is re-invoked at every decision
epoch of a rolling-horizon simulation).

Controller contract (mandatory):
- The program must define create_controller(cost=None) returning a callable
  controller(snap, view, rng) -> Solution.
- snap (Instance): the epoch snapshot — orders released up to the epoch time
  (release <= tau) not yet delivered or in flight; planning horizon
  snap.horizon. Helpers: snap.dist(i, j), snap.tau_g(i, j, t),
  snap.tau_a(i, j, t, cruise_speed), snap.weather_at(t), snap.active_nfzs(t),
  snap.orders, snap.vehicles, snap.drones, snap.nodes.
- view (FleetView): fleet state at the epoch — view.vehicles[v] has
  (node, available_at, onboard, used); view.drones[d] has (vertiport,
  available_at, busy_until, energy_wh); plus view.tau and view.sim_end.
- rng is the simulator's random.Random; deterministic controllers may ignore
  it.
- Parcels already onboard a ground vehicle (view.vehicles[v].onboard) are
  committed: they keep their vehicle and must stay first on its route.
- Claimed arrival times must be earliest-feasible; the simulator re-derives
  times and rejects optimistic claims.

Solution construction (all imported): Solution(epoch, ground_routes,
drone_sorties, unserved); GroundRoute(vehicle, stops, return_depot,
return_time); GroundStop(order, node, arrival); DroneSortie(drone,
launch_vertiport, launch_time, stops, land_vertiport, land_time, energy_wh);
DroneStop(order, node, arrival).

Hard constraints C1-C11 (any violation makes the solution infeasible and the
program's fitness collapses to a big penalty):
- C1: every released order is served exactly once or declared unserved.
- C2: ground vehicle capacity. C3: ground travel-time propagation with
  time-dependent congestion. C4: shift-end (start and return).
- C5: drone sorties launch/land at vertiports, at most M_max parcels per
  sortie, payload and per-leg load within Q_a.
- C6: battery energy over all legs. C7: sortie wall-clock <= L_max.
- C8: drone time propagation, NFZ leg check at departure time, no launch in
  storm. C9: lateness accounting u = max(0, start - l).
- C10: service starts within [0, T]. C11: a drone flies at most one sortie
  at a time (chaining after landing is allowed).

Available helpers (already imported for you): math, random, TOL, CostParams,
check_solution, FleetView, _propagate_ground(snap, veh, seq, T, depots,
enforce_horizon) -> (stops, return_depot, return_time, lateness_penalty, km)
or None, and _leg_blocked(snap, i, j, t) -> bool.

Fitness: adversarial regret against a rolling re-optimization insertion
oracle inside an epsilon-ball of environment parameters — over perturbed
environments, fitness = -(1.0 * mean_regret + 0.5 * max_regret). Lower
regret is better (higher fitness is better); unserved orders and lateness
are expensive, and robustness across perturbed environments matters more
than nominal brilliance.

Output format: exactly one fenced python code block and no other text. The
block must be self-contained (only the injected names above) and define
create_controller(cost=None)."""

DEFAULT_MUTATION_OPERATORS = (
    "Tweak numeric constants (cost weights, thresholds, margins) in the "
    "parent and explain none of it — code only.",
    "Change the order in which orders are considered (window start, release "
    "time, slack, distance, or a mix).",
    "Change the ground insertion rule (nearest vehicle vs cheapest "
    "regret-based insertion, different insertion positions).",
    "Improve drone usage: better launch/vertiport choice, battery-aware "
    "decisions, cheaper sortie leg pairing.",
    "Add tie-breaking or fallback rules so that orders feasible by neither "
    "mode are handled deliberately.",
)


def _seed_user_prompt() -> str:
    return """Write ONE initial controller program for the DAGCD problem
(described in the system message). Keep it simple but feasible under C1-C11.
You may start from this seed template (equivalent to the greedy
nearest-dispatch baseline) and modify it, or write something better:

```python
""" + SEED_TEMPLATE + """
```

Output exactly one fenced python code block and no other text."""


def _build_mutation_user_prompt(parent_source: str, parent_fitness: float,
                                parent_feedback: str, worst_feedback: str,
                                operators: list) -> str:
    """SDFF-style mutation prompt: parent source + failure feedback + worst-of-
    generation audit summary + operator menu.  The parent's code block is the
    LAST fenced block in the message (MockLLM relies on this)."""
    ops = "\n".join(f"- {op}" for op in operators)
    fb = parent_feedback.strip() or "(none)"
    wfb = worst_feedback.strip() or "(none)"
    return f"""Parent program (fitness {parent_fitness:.4f}, higher is better):

```python
{parent_source}
```

Parent evaluation feedback: {fb}

Worst individual of this generation (avoid its failure mode): {wfb}

Produce ONE mutated offspring. Apply one (or at most two) of these operators:
{ops}

The offspring must keep the create_controller(cost=None) contract and stay
feasible under C1-C11. Output exactly one fenced python code block and no
other text."""


# =====================================================================
# evolution loop
# =====================================================================

@dataclass
class EvolutionConfig:
    """Configuration of one evolution run (all sizes small by default)."""

    population_size: int = 8
    n_generations: int = 5
    elite_frac: float = 0.25
    n_offspring: int | None = None        # None -> population_size - n_elite
    fitness_cfg: FitnessConfig = field(default_factory=FitnessConfig)
    max_tokens: int = 4096
    temperature: float = 0.7
    mutation_operators: list = field(
        default_factory=lambda: list(DEFAULT_MUTATION_OPERATORS))
    archive_dir: str | None = None
    seed_programs: list = field(default_factory=lambda: [SEED_TEMPLATE])
    llm_call_budget: int = 64
    patience: int = 2                     # gens without best improvement
    strict_llm: bool = False              # propagate EvolutionError to caller
    llm_extra_body: dict | None = None    # merged into every chat request JSON


@dataclass
class EvolutionResult:
    """Outcome of one evolution run (JSON-serializable)."""

    best_source: str
    best_fitness: float
    trajectory: list                     # [{gen, best, mean, worst}]
    archive_dir: str | None
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    notes: list


def run_evolution(client: LLMClient, config: EvolutionConfig,
                  seed: int = 0) -> EvolutionResult:
    """Run the EoH-style evolution loop (self-contained, reproducible).

    gen0 = seed_programs + LLM-generated fillers (within the call budget);
    every generation evaluates the population (cached — Invariant 2),
    archives it to gen_{g:03d}.jsonl (Invariant 3), keeps the elite fraction,
    selects parents by fitness, and generates offspring with the SDFF mutation
    prompt.  Parse/compile/protocol-probe failures retry at most twice per
    offspring slot, then the slot falls back to a deterministic numeric
    perturbation of an elite (recorded in notes); accepted candidates are
    compile- and probe-clean, so penalty fitness can only appear through
    audit-time failures and never ranks above a feasible program in
    elite/parent/best selection.  Stops on patience plateau or budget
    exhaustion (Invariant 6).  With ``strict_llm=True`` an EvolutionError
    from the client propagates instead of falling back (used by the real-LLM
    smoke runner to detect model-level errors and switch to the fallback
    model).
    """
    rng = random.Random(seed)
    notes: list[str] = []
    counters = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    archive_dir = config.archive_dir
    llm_log = None
    if archive_dir:
        os.makedirs(archive_dir, exist_ok=True)
        llm_log = open(os.path.join(archive_dir, "llm_calls.jsonl"), "w",
                       encoding="utf-8")

    def log_llm(row: dict) -> None:
        if llm_log is not None:
            llm_log.write(json.dumps(row) + "\n")
            llm_log.flush()

    def call_llm(system: str, user: str, purpose: str, gen: int) -> str | None:
        """One budgeted LLM call; None when the budget is exhausted or (in
        non-strict mode) the call failed."""
        if counters["llm_calls"] >= config.llm_call_budget:
            return None
        counters["llm_calls"] += 1
        try:
            text = client.chat(system, user, max_tokens=config.max_tokens,
                               temperature=config.temperature,
                               extra_body=config.llm_extra_body)
        except EvolutionError:
            log_llm({"gen": gen, "purpose": purpose, "ok": False,
                     "error": "EvolutionError"})
            if config.strict_llm:
                raise
            return None
        usage = getattr(client, "last_usage", None) or {}
        counters["prompt_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        counters["completion_tokens"] += \
            int(usage.get("completion_tokens", 0) or 0)
        log_llm({"gen": gen, "purpose": purpose, "ok": True,
                 "usage": usage,
                 "finish_reason": getattr(client, "last_finish", None),
                 "response_sha256": hashlib.sha256(
                     text.encode("utf-8")).hexdigest()})
        return text

    try:
        # ---- acceptance gate: compile + protocol probe (no simulations) ----
        def accepts(cand: str) -> bool:
            try:
                factory = compile_controller(cand)
            except ProgramError:
                return False
            ok, _fb = probe_controller(factory)
            return ok

        # ---------------- generation 0 population ----------------
        pop_sources = list(dict.fromkeys(config.seed_programs))[
            :config.population_size]
        deterministic_seeds = 0
        while len(pop_sources) < config.population_size:
            child = None
            for _ in range(3):            # parse retries, <=2 per slot
                text = call_llm(_SYSTEM_PROMPT, _seed_user_prompt(),
                                "seed_fill", 0)
                if text is None:
                    break
                cand = _extract_python_block(text)
                if cand is None or not accepts(cand):
                    continue
                child = cand
                break
            if child is None:
                base = rng.choice(pop_sources) if pop_sources else SEED_TEMPLATE
                child = _perturb_source(base, rng)
                deterministic_seeds += 1
            pop_sources.append(child)
        if deterministic_seeds:
            notes.append(f"gen0: {deterministic_seeds} individual(s) filled "
                         "deterministically (LLM budget exhausted or "
                         "unparseable responses)")

        penalty = config.fitness_cfg.big_penalty

        def rank(f: float) -> float:
            """Selection/best fitness: penalties rank below every real fitness.

            big_penalty (1e9, positive) would otherwise dominate the max() over
            a generation whose feasible programs all have fitness <= 0."""
            return f if f < penalty else -math.inf

        trajectory: list = []
        best_source: str | None = None
        best_fitness = -math.inf
        raw_best = (-math.inf, None)     # fallback if no feasible individual
        stall = 0
        n_evals = 0

        for gen in range(config.n_generations):
            evals = [(src, evaluate_program(src, config.fitness_cfg))
                     for src in pop_sources]
            n_evals += len(evals)
            fits = [r.fitness for _, r in evals]
            gen_worst = min(fits)
            gen_mean = statistics.fmean(fits)
            gen_best_real = max(rank(f) for f in fits)
            raw_gen_best = max(fits)
            if raw_gen_best > raw_best[0]:
                raw_best = (raw_gen_best,
                            max(evals, key=lambda p: p[1].fitness)[0])
            if gen_best_real > best_fitness + 1e-12:
                best_fitness = gen_best_real
                best_source = max(evals,
                                  key=lambda p: rank(p[1].fitness))[0]
                stall = 0
            else:
                stall += 1
            trajectory.append({"gen": gen, "best": best_fitness,
                               "mean": gen_mean, "worst": gen_worst})

            # ---- archive this generation (Invariant 3) ----
            if archive_dir:
                path = os.path.join(archive_dir, f"gen_{gen:03d}.jsonl")
                with open(path, "w", encoding="utf-8") as f:
                    for src, res in evals:
                        f.write(json.dumps({
                            "source": src,
                            "fitness": res.fitness,
                            "feedback": res.feedback,
                            "sha": hashlib.sha256(
                                src.encode("utf-8")).hexdigest(),
                            "cached": res.cache_hit,
                        }) + "\n")

            if stall >= config.patience and gen < config.n_generations - 1:
                notes.append(f"early stop at generation {gen}: no best-"
                             f"fitness improvement for {stall} generation(s) "
                             f"(patience={config.patience})")
                break
            if gen == config.n_generations - 1:
                break

            # ---- selection (penalty programs rank last, Invariant 5) ----
            unique = {src: res for src, res in evals}
            ranked = sorted(unique,
                            key=lambda s: (rank(unique[s].fitness),
                                           unique[s].fitness),
                            reverse=True)
            n_elite = max(1, int(round(config.elite_frac
                                       * config.population_size)))
            elites = ranked[:n_elite]
            n_offspring = config.n_offspring if config.n_offspring \
                is not None else config.population_size - n_elite
            worst_feedback = unique[ranked[-1]].feedback

            def pick_parent() -> str:
                a, b = rng.choice(ranked), rng.choice(ranked)
                return a if rank(unique[a].fitness) >= rank(unique[b].fitness) \
                    else b

            offsprings: list = []
            n_fallback = 0
            for _ in range(n_offspring):
                parent = pick_parent()
                prompt = _build_mutation_user_prompt(
                    parent, unique[parent].fitness, unique[parent].feedback,
                    worst_feedback, config.mutation_operators)
                child = None
                for _try in range(3):     # parse retries, <=2 per slot
                    text = call_llm(_SYSTEM_PROMPT, prompt, "offspring", gen)
                    if text is None:
                        break
                    cand = _extract_python_block(text)
                    if cand is None or not accepts(cand):
                        continue
                    child = cand
                    break
                if child is None:
                    child = _perturb_source(rng.choice(elites), rng)
                    n_fallback += 1
                offsprings.append(child)
            if n_fallback:
                notes.append(f"generation {gen}: {n_fallback} offspring "
                             "slot(s) filled deterministically (parse/"
                             "compile failure or LLM budget)")
            pop_sources = elites + offsprings

        if best_source is None:
            # no feasible individual ever evaluated: fall back to the top
            # raw-fitness program (a penalty program) and say so loudly
            best_fitness, best_source = raw_best
            notes.append("WARNING: every evaluated individual carried the "
                         "big penalty (compile/probe/audit failures); "
                         "best_source is the top raw-fitness program, i.e. "
                         "broken code — do not use without inspection")
        notes.append(f"stopped after {len(trajectory)} generation(s), "
                     f"{n_evals} evaluations, llm_calls="
                     f"{counters['llm_calls']}/{config.llm_call_budget}")
        notes.append("LLM nondeterminism: this run's sources, per-call usage "
                     "and finish_reasons are archived (gen_*.jsonl / "
                     "llm_calls.jsonl); reruns with the same seed reproduce "
                     "only given identical LLM responses")
        return EvolutionResult(
            best_source=best_source, best_fitness=best_fitness,
            trajectory=trajectory, archive_dir=archive_dir,
            llm_calls=counters["llm_calls"],
            prompt_tokens=counters["prompt_tokens"],
            completion_tokens=counters["completion_tokens"], notes=notes)
    finally:
        if llm_log is not None:
            llm_log.close()

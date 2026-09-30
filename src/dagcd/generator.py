"""Instance generator for DAGCD — parameter vector theta defines distribution D(theta).

Implements docs/SPECIFICATION.md Sec. 6: full ground-truth sampling (Scenario,
seed-reproducible) plus the static snapshot entry sample_instance and
perturb_params, the epsilon-ball operator of the adversarial inner loop. The
parameter schema is normative: robustness experiments perturb theta within
epsilon-balls.
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass, field, asdict

from .instance import (Instance, Node, Order, GroundVehicle, Drone, NFZ,
                       WeatherEvent, WEATHER_STATES, TOL)

# fleet constants not carried by InstanceParams (mirrored by the test fixtures)
_DRONE_CRUISE_M_MIN = 600.0
_DRONE_MAX_SORTIE_MIN = 60.0
_DRONE_MAX_PARCEL = 3


@dataclass
class InstanceParams:
    """theta — fully defines the instance distribution (docs/SPECIFICATION.md Sec. 6)."""

    # demand
    lam_peak: float = 3.0          # orders per minute at peak
    lam_trough_ratio: float = 0.3  # trough rate = ratio * peak
    peak_start: float = 480.0      # peak window [min]
    peak_end: float = 720.0
    n_clusters: int = 3
    cluster_spread_m: float = 1500.0

    # network / congestion
    n_customers_area: float = 6000.0   # service area side [m]
    cong_level: float = 0.5            # 0 free-flow .. 1 severe; scales congestion curve depth
    n_depots: int = 1
    n_vertiports: int = 2

    # disturbances
    wx_chain: dict = field(default_factory=lambda: {
        # rows: from-state; cols: to-state (clear, windy, rain, storm)
        "clear": [0.90, 0.08, 0.02, 0.00],
        "windy": [0.40, 0.50, 0.09, 0.01],
        "rain":  [0.20, 0.35, 0.40, 0.05],
        "storm": [0.30, 0.30, 0.30, 0.10],
    })
    mean_event_min: float = 45.0   # mean weather episode length
    nfz_density: float = 0.5       # expected active NFZs per hour

    # fleet
    n_vehicles: int = 4
    vehicle_capacity: float = 40.0
    n_drones: int = 3
    drone_payload: float = 5.0
    drone_battery_wh: float = 500.0

    horizon: float = 240.0         # planning horizon T of one epoch snapshot
    # ground-truth simulation length [min]; distinct from the epoch planning horizon
    sim_duration: float = 720.0


@dataclass
class Scenario:
    """Full ground truth sampled from D(theta) — JSON-serializable, seed-archived."""
    params: InstanceParams
    nodes: dict[int, Node]
    orders: dict[str, Order]      # full stream, releases over [0, sim_end]
    vehicles: dict[str, GroundVehicle]   # fleet definition (original depot/start/shift)
    drones: dict[str, Drone]
    nfzs: list[NFZ]               # with activation schedules
    weather: list[WeatherEvent]   # full chain over [0, sim_end]
    congestion: list[tuple[float, float, float]]
    sim_end: float
    # model constants mirrored from Instance defaults
    v_g: float = 500.0
    speed_factor: dict = field(default_factory=lambda:
        {"clear": 1.0, "windy": 0.9, "rain": 0.75, "storm": 0.0})
    energy_alpha: float = 0.02
    energy_beta: float = 0.0005
    energy_mu: dict = field(default_factory=lambda:
        {"clear": 1.0, "windy": 1.2, "rain": 1.5, "storm": 1.0})


# ---------------- sampling helpers ----------------

def _poisson(rng: random.Random, lam: float) -> int:
    """Knuth sampler, exact for small Poisson means (NFZ counts)."""
    if lam <= 0.0:
        return 0
    limit = math.exp(-lam)
    k, p = 0, 1.0
    while p > limit:
        k += 1
        p *= rng.random()
    return k - 1


def _sample_arrival_times(rng: random.Random, params: InstanceParams,
                          sim_end: float) -> list[float]:
    """Piecewise-homogeneous Poisson process (trough/peak/trough), exact via
    inversion of the cumulative intensity within each constant-rate segment."""
    lam_peak = params.lam_peak
    lam_trough = params.lam_peak * params.lam_trough_ratio
    ps = min(max(params.peak_start, 0.0), sim_end)
    pe = min(max(params.peak_end, 0.0), sim_end)
    if pe < ps:
        ps, pe = pe, ps
    segs = [(0.0, ps, lam_trough), (ps, pe, lam_peak), (pe, sim_end, lam_trough)]

    times = []
    t = 0.0
    si = 0
    resid = -math.log(1.0 - rng.random())
    while t < sim_end - TOL:
        while si < len(segs) and t >= segs[si][1] - TOL:
            si += 1
        if si >= len(segs):
            break
        a, b, lam = segs[si]
        if lam <= 0.0:
            t = b
            si += 1
            continue
        capacity = lam * (b - t)
        if resid <= capacity + TOL:
            t += resid / lam
            resid = -math.log(1.0 - rng.random())
            if t < sim_end - TOL:
                times.append(t)
        else:
            resid -= capacity
            t = b
            si += 1
    return times


def _wx_step(rng: random.Random, chain: dict, state: str) -> str:
    row = chain.get(state)
    if not row:
        return state
    u = rng.random()
    acc = 0.0
    for nxt, p in zip(WEATHER_STATES, row):
        acc += p
        if u <= acc:
            return nxt
    return WEATHER_STATES[-1]


def _sample_weather(rng: random.Random, params: InstanceParams,
                    sim_end: float) -> list[WeatherEvent]:
    """Markov chain over WEATHER_STATES starting from 'clear'; episode lengths are
    exponential(mean_event_min); the last episode is truncated at sim_end."""
    events = []
    t = 0.0
    state = "clear"
    while t < sim_end - TOL:
        dur = rng.expovariate(1.0 / params.mean_event_min)
        end = min(t + dur, sim_end)
        events.append(WeatherEvent(t, end, state))
        state = _wx_step(rng, params.wx_chain, state)
        t = end
    return events


def _congestion_table(params: InstanceParams,
                      sim_end: float) -> list[tuple[float, float, float]]:
    """Piecewise-constant congestion curve (docs/SPECIFICATION.md Sec. 3): c = 1 - 0.2*cong_level
    inside the peak window, c = 1 - 0.4*cong_level in the 60-min shoulders around it,
    c = 1.0 otherwise; contiguous coverage of [0, sim_end]."""
    c_peak = 1.0 - 0.2 * params.cong_level
    c_shoulder = 1.0 - 0.4 * params.cong_level
    ps, pe = params.peak_start, params.peak_end
    cuts = {0.0, sim_end,
            min(max(ps - 60.0, 0.0), sim_end), min(max(ps, 0.0), sim_end),
            min(max(pe, 0.0), sim_end), min(max(pe + 60.0, 0.0), sim_end)}
    bounds = sorted(cuts)
    table = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a < TOL:
            continue
        mid = 0.5 * (a + b)
        if ps <= mid < pe:
            c = c_peak
        elif ps - 60.0 <= mid < ps or pe <= mid < pe + 60.0:
            c = c_shoulder
        else:
            c = 1.0
        table.append((a, b, c))
    return table


# ---------------- main entry ----------------

def sample_scenario(params: InstanceParams, seed: int) -> Scenario:
    """Sample a full ground-truth scenario from D(params) with a dedicated
    random.Random(seed), so scenarios are byte-for-byte reproducible."""
    rng = random.Random(seed)
    sim_end = params.sim_duration
    side = params.n_customers_area
    cx = cy = side / 2.0

    # network: depot 0 at the area centre, vertiports on a ring, node ids dense
    nodes: dict[int, Node] = {}
    for i in range(max(1, params.n_depots)):
        if i == 0:
            nodes[0] = Node(0, cx, cy, "depot")
        else:
            ang = 2.0 * math.pi * (i - 1) / max(1, params.n_depots - 1)
            nodes[i] = Node(i, cx + 0.2 * side * math.cos(ang),
                            cy + 0.2 * side * math.sin(ang), "depot")
    n_vert = max(0, params.n_vertiports)
    ring_r = 0.45 * side
    phase = rng.uniform(0.0, 2.0 * math.pi)
    for j in range(n_vert):
        nid = len(nodes)
        ang = phase + 2.0 * math.pi * j / max(1, n_vert)
        nodes[nid] = Node(nid, cx + ring_r * math.cos(ang), cy + ring_r * math.sin(ang),
                          "vertiport")

    # cluster centres for order destinations, uniform over the service area
    clusters = [(rng.uniform(0.0, side), rng.uniform(0.0, side))
                for _ in range(max(1, params.n_clusters))]

    orders: dict[str, Order] = {}
    for i, t in enumerate(_sample_arrival_times(rng, params, sim_end)):
        ccx, ccy = clusters[rng.randrange(len(clusters))]
        x = min(max(ccx + rng.gauss(0.0, params.cluster_spread_m), 0.0), side)
        y = min(max(ccy + rng.gauss(0.0, params.cluster_spread_m), 0.0), side)
        nid = len(nodes)
        nodes[nid] = Node(nid, x, y, "customer")
        oid = f"o{i + 1}"
        release = t
        weight = rng.uniform(0.5, max(0.9 * params.drone_payload, 0.5 + 1e-9))
        service = rng.uniform(1.0, 3.0)
        e = release + rng.uniform(0.0, 15.0)
        l = e + rng.uniform(30.0, 90.0)
        orders[oid] = Order(oid, nid, release, weight, service, e, l)

    weather = _sample_weather(rng, params, sim_end)

    nfzs: list[NFZ] = []
    n_nfz = _poisson(rng, params.nfz_density * sim_end / 60.0)
    for i in range(n_nfz):
        w = rng.uniform(300.0, 1200.0)
        h = rng.uniform(300.0, 1200.0)
        ccx = rng.uniform(0.0, side)
        ccy = rng.uniform(0.0, side)
        poly = [[ccx - w / 2, ccy - h / 2], [ccx + w / 2, ccy - h / 2],
                [ccx + w / 2, ccy + h / 2], [ccx - w / 2, ccy + h / 2]]
        active_from = rng.uniform(0.0, max(sim_end - 60.0, 0.0))
        active_to = min(active_from + rng.uniform(30.0, 180.0), sim_end)
        nfzs.append(NFZ(f"z{i + 1}", poly, active_from=active_from, active_to=active_to))

    congestion = _congestion_table(params, sim_end)

    vehicles = {f"gv{i}": GroundVehicle(f"gv{i}", params.vehicle_capacity, 0, 0.0, sim_end)
                for i in range(params.n_vehicles)}
    drones = {f"da{i}": Drone(f"da{i}", params.drone_payload, params.drone_battery_wh,
                              _DRONE_CRUISE_M_MIN, _DRONE_MAX_SORTIE_MIN, _DRONE_MAX_PARCEL)
              for i in range(params.n_drones)}

    return Scenario(params=params, nodes=nodes, orders=orders, vehicles=vehicles,
                    drones=drones, nfzs=nfzs, weather=weather,
                    congestion=congestion, sim_end=sim_end)


def snapshot_at(sc: Scenario, tau: float, horizon: float,
                exclude: set[str] | None = None,
                info_mode: str = "full") -> Instance:
    """Epoch snapshot: released orders (release <= tau, minus `exclude`), full network,
    original weather/congestion tables and NFZ schedules (filtered via active_nfzs),
    planning horizon T = tau + horizon, same model constants as the scenario.

    ``info_mode`` controls the controllers' information pattern:
      * ``"full"`` (default) — clairvoyant: the complete weather chain and all
        NFZ activation schedules are handed over (historic behaviour, unchanged).
      * ``"forecast"`` — non-clairvoyant: the weather chain is truncated to
        events that have fully ended by tau and extended with a persistence
        forecast [tau, sim_end] that continues the state currently observed at
        tau; only NFZs active at tau are kept (future activations removed). The
        congestion table is a published schedule and stays full in both modes.
    """
    exclude = exclude or set()
    orders = {oid: o for oid, o in sc.orders.items()
              if o.release <= tau and oid not in exclude}
    if info_mode == "full":
        weather = sc.weather
        nfzs = sc.nfzs
    elif info_mode == "forecast":
        # current state at tau read from the FULL chain before truncation
        state = "clear"
        for w in sc.weather:
            if w.t_start - TOL <= tau < w.t_end:
                state = w.state
                break
        weather = [w for w in sc.weather if w.t_end <= tau + TOL]
        weather.append(WeatherEvent(tau, sc.sim_end, state))
        nfzs = [z for z in sc.nfzs
                if (z.active_from is None or z.active_from <= tau)
                and (z.active_to is None or tau < z.active_to)]
    else:
        raise ValueError(f"unknown info_mode {info_mode!r}; "
                         "expected 'full' or 'forecast'")
    return Instance(
        nodes=sc.nodes,
        orders=orders,
        vehicles=sc.vehicles,
        drones=sc.drones,
        nfzs=nfzs,
        weather=weather,
        horizon=tau + horizon,
        v_g=sc.v_g,
        congestion=sc.congestion,
        speed_factor=sc.speed_factor,
        energy_alpha=sc.energy_alpha,
        energy_beta=sc.energy_beta,
        energy_mu=sc.energy_mu,
    )


def sample_instance(params: InstanceParams, seed: int | None = None) -> Instance:
    """Static snapshot entry point: the epoch-0 view of one sampled
    scenario — orders already released at t=0 within the planning horizon."""
    if seed is None:
        seed = random.randrange(2 ** 31)
    scenario = sample_scenario(params, seed)
    return snapshot_at(scenario, 0.0, params.horizon)


# ---------------- JSON I/O ----------------

def scenario_to_dict(sc: Scenario) -> dict:
    d = asdict(sc)
    d["congestion"] = [list(x) for x in sc.congestion]
    return d


def scenario_from_dict(d: dict) -> Scenario:
    return Scenario(
        params=InstanceParams(**d["params"]),
        nodes={int(k): Node(**v) for k, v in d["nodes"].items()},
        orders={k: Order(**v) for k, v in d["orders"].items()},
        vehicles={k: GroundVehicle(**v) for k, v in d["vehicles"].items()},
        drones={k: Drone(**v) for k, v in d["drones"].items()},
        nfzs=[NFZ(**z) for z in d.get("nfzs", [])],
        weather=[WeatherEvent(**w) for w in d.get("weather", [])],
        congestion=[tuple(x) for x in d.get("congestion", [])],
        sim_end=d["sim_end"],
        v_g=d.get("v_g", 500.0),
        speed_factor=d.get("speed_factor"),
        energy_alpha=d.get("energy_alpha", 0.02),
        energy_beta=d.get("energy_beta", 0.0005),
        energy_mu=d.get("energy_mu"),
    )


def save_scenario(sc: Scenario, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scenario_to_dict(sc), f, indent=2)


def load_scenario(path: str) -> Scenario:
    with open(path, "r", encoding="utf-8") as f:
        return scenario_from_dict(json.load(f))


def _multiplicative(eps: float, rng: random.Random) -> float:
    """Sample r > 0 with |log r| <= eps (uniform in log-space within the ball)."""
    log_r = min(max(rng.uniform(-1.0, 1.0) * eps, -700.0), 700.0)
    return math.exp(log_r)


def _bad_day_multiplicative(eps: float, rng: random.Random) -> float:
    """Coupled-mode magnitude: same log-space distribution as ``_multiplicative``
    (|log r| <= eps) but the direction is fixed to r > 1 — the 'bad' direction."""
    log_r = min(max(abs(rng.uniform(-1.0, 1.0)) * eps, -700.0), 700.0)
    return math.exp(log_r)


def _perturb_wx_bad_day(chain: dict, eps: float, rng: random.Random) -> dict:
    """Storm-directed WX perturbation with projection onto the TV ball.

    Same projection as ``_perturb_wx`` (clip non-negative, renormalize rows,
    shrink toward the original matrix to land exactly on TV = eps when the raw
    matrix exceeds it); the ONLY difference is the direction: a symmetric
    per-entry perturbation is given a storm-ward bias — every non-storm row's
    transition TO 'storm' is pushed up and the storm row's exits to other
    states are pushed down, each by an independent magnitude <= eps. The TV
    bound TV(P', P) <= eps therefore holds exactly as in ``_perturb_wx``.
    """
    n = len(WEATHER_STATES)
    si = WEATHER_STATES.index("storm")
    raw = {}
    for s in WEATHER_STATES:
        row = chain.get(s)
        if row is None:
            continue
        pert = [float(p) + rng.uniform(-eps, eps) for p in row]
        if s != "storm":
            pert[si] += abs(rng.uniform(-eps, eps))        # toward storm
        else:
            for i in range(n):
                if i != si:
                    pert[i] -= abs(rng.uniform(-eps, eps))  # linger in storm
        pert = [max(0.0, p) for p in pert]
        tot = sum(pert)
        raw[s] = ([float(p) for p in row] if tot <= 0.0
                  else [p / tot for p in pert])
    tv = max((0.5 * sum(abs(float(chain[s][i]) - raw[s][i])
                        for i in range(n)) for s in raw), default=0.0)
    if tv > eps and tv > 0.0:
        lam = eps / tv
        for s in raw:
            raw[s] = [(1.0 - lam) * float(chain[s][i]) + lam * raw[s][i]
                      for i in range(n)]
    out = {s: [float(p) for p in row] for s, row in chain.items()}
    out.update(raw)
    return out


def _perturb_wx(chain: dict, eps: float, rng: random.Random) -> dict:
    """Symmetric entry-wise perturbation of a Markov transition matrix with
    projection back into the TV ball.

    Steps (docs/SPECIFICATION.md Sec. 6): sample a symmetric perturbation of magnitude eps per
    entry, clip each row non-negative, renormalize rows to sum 1; if the total-
    variation distance TV(P', P) = max_state 0.5 * sum_next |P' - P| exceeds eps,
    shrink toward the original matrix, P'_lam = (1-lam) P + lam P', with lam = eps/TV
    (TV is linear in lam, so this lands exactly on the boundary). Rows that clip to
    zero mass keep their original distribution.
    """
    n = len(WEATHER_STATES)
    raw = {}
    for s in WEATHER_STATES:
        row = chain.get(s)
        if row is None:
            continue
        pert = [max(0.0, float(p) + rng.uniform(-eps, eps)) for p in row]
        tot = sum(pert)
        raw[s] = ([float(p) for p in row] if tot <= 0.0
                  else [p / tot for p in pert])
    tv = max((0.5 * sum(abs(float(chain[s][i]) - raw[s][i])
                        for i in range(n)) for s in raw), default=0.0)
    if tv > eps and tv > 0.0:
        lam = eps / tv
        for s in raw:
            raw[s] = [(1.0 - lam) * float(chain[s][i]) + lam * raw[s][i]
                      for i in range(n)]
    out = {s: [float(p) for p in row] for s, row in chain.items()}
    out.update(raw)
    return out


def perturb_params(params: InstanceParams, eps: dict | None,
                   rng: random.Random, *, couple: bool = True) -> InstanceParams:
    """Adversarial inner-loop operator: sample theta' inside the epsilon-ball
    (docs/SPECIFICATION.md Sec. 6, RAISE-style constrained sampling with projection).

    ``eps`` keys (all optional; missing keys leave the component unchanged,
    ``eps=None``/``{}`` return an unchanged deep copy):

      "lam"      multiplicative on ``lam_peak`` with |log(lam'/lam)| <= eps (lam' > 0)
      "clusters" multiplicative on ``cluster_spread_m`` with |log r| <= eps — this
                 operator moves ONLY the spread; cluster-centre displacement is
                 not possible because centres are not part of
                 InstanceParams (documented limitation, docs/SPECIFICATION.md
                 Sec. 6)
      "cong"     additive on ``cong_level``, clipped to [0, 1]: |cong' - cong| <= eps
      "nfz"      multiplicative on ``nfz_density`` with |log r| <= eps and floor 0.01
      "wx"       symmetric transition-matrix perturbation, TV(P', P) <= eps with
                 projection onto the boundary (see ``_perturb_wx`` for the TV norm)

    ``couple`` (default True) — multi-dimensional correlated perturbation:
      - True: with probability 1/2 the sampler draws a **coherent bad-day axis**:
        the sign of every requested component's perturbation points "bad-ward"
        — ``lam`` up (busier), ``nfz`` up (denser), ``wx`` storm-directed
        (see ``_perturb_wx_bad_day``), ``cong`` up (more congested) — while
        each magnitude is still sampled independently inside its own eps
        bound. With probability 1/2 (and always for ``clusters``, whose
        spread direction carries no intrinsic "bad" sense) the components are
        sampled from the independent box exactly as in the uncoupled mode.
      - False: uncoupled (independent-box) semantics, bit-identical draw order
        and values — scenarios produced with uncoupled sampling stay
        reproducible.

    Invariants (asserted by tests/test_perturb.py):
      1. the input ``params`` is never mutated; a new deep-copied InstanceParams is
         returned (no aliasing of ``wx_chain`` or other mutable fields);
      2. every requested bound holds exactly in BOTH modes (log-bounds / additive
         bound / TV bound); whenever a legality clip is tighter than the eps-ball
         (only possible at extreme eps), legality wins and the bound may be
         violated — such cases are asserted as legality-only in tests. The result
         is always a valid parameter vector (lam_peak > 0, cluster_spread_m > 0,
         cong_level in [0, 1], nfz_density >= 0.01, wx rows non-negative and
         summing to 1);
      3. reproducible: same (params, eps, couple) and same-seeded rng give
         identical output; ``couple=True`` consumes exactly one leading
         ``random()`` for the axis draw before the component draws.
    """
    eps = eps or {}
    out = copy.deepcopy(params)
    coupled = couple and rng.random() < 0.5
    if "lam" in eps:
        out.lam_peak = params.lam_peak * (
            _bad_day_multiplicative(eps["lam"], rng) if coupled
            else _multiplicative(eps["lam"], rng))
    if "clusters" in eps:
        # spread-only, independent box in both modes (see docstring)
        out.cluster_spread_m = min(max(params.cluster_spread_m
                                       * _multiplicative(eps["clusters"], rng),
                                       1.0), 1e9)
    if "cong" in eps:
        delta = abs(rng.uniform(-eps["cong"], eps["cong"])) if coupled \
            else rng.uniform(-eps["cong"], eps["cong"])
        out.cong_level = min(max(params.cong_level + delta, 0.0), 1.0)
    if "nfz" in eps:
        out.nfz_density = max(params.nfz_density
                              * (_bad_day_multiplicative(eps["nfz"], rng) if coupled
                                 else _multiplicative(eps["nfz"], rng)), 0.01)
    if "wx" in eps:
        out.wx_chain = (_perturb_wx_bad_day(params.wx_chain, eps["wx"], rng)
                        if coupled else _perturb_wx(params.wx_chain, eps["wx"], rng))
    return out

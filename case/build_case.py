"""Build the Shenzhen urban low-altitude logistics case study.

One-click reconstruction of ``scenario.json`` from embedded real-world
coordinates, fixed seeds and cached Amap (Gaode) driving-distance responses::

    python3 build_case.py            # offline: reuse amap_raw/ cache (0 API calls)
    python3 build_case.py --refetch  # re-call the Amap driving API (53 calls)
    python3 build_case.py --order-seed 1 --wx-seed 1 --out scenario_day1.json
                                     # variant operating day (the default seeds
                                     # reproduce scenario.json byte-identically)

Design (see README.md of this directory for the full documentation):

* Geography: Nanshan-Bao'an built-up area of Shenzhen. Coordinates are
  approximate real locations (logistics park, vertiport candidates, business
  districts) hand-picked on the public map; listed in COORDS below and
  archived in ``case_nodes.json``. Local metric frame: equirectangular
  anchored at (22.55 N, 113.93 E), WGS84 spherical R = 6378137 m.
* Road distances: circuity fitting (a standard approximation under API quota).
  45 OD pairs (stratified short/mid/long) are sampled with a fixed
  seed and queried against the Amap driving-directions API; a single scalar
  ``circuity = mean(road / euclidean)`` is fitted and applied to ALL node
  pairs by scaling every coordinate and NFZ polygon by it. 8 held-out pairs
  report the fit error ``mean |pred - real| / real``.  The Amap-reported
  durations are recorded as a sanity statistic only and never enter the model.
* Ground speed: v_g = 350 m/min in the road-equivalent metric (about 21 km/h
  real urban delivery speed at circuity ~ 1.35), compressed by the
  piecewise-constant Shenzhen rush-hour congestion table (7:30-9:30 and
  17:30-20:00, cong_level = 0.6).
* Drone energy coefficients are divided by the circuity so that reported Wh
  stay physical (leg lengths are road-equivalent metres); drone cruise speed
  stays the package default 600 m/min model units (a conservative urban
  multirotor figure once the metric is road-equivalent).
* Weather: rainy-season Markov chain (rain/storm probabilities clearly above
  the package default; Shenzhen subtropical monsoon climate common knowledge -
  no fabricated citations), mean_event_min = 40, sim_duration = 720 min
  (sim day starts 07:00 local).
* NFZ: simplified 5-vertex approximation of the Shenzhen Bao'an airport
  (terminal + runway strip, ~22.61-22.66 N, 113.79-113.84 E), activated
  15:00-18:00 (temporary afternoon restriction), plus one small temporary NFZ
  near Bao'an Stadium (active 17:00-19:00).
* Orders: hand-built Order objects (sampler entry intentionally NOT used).
  Releases follow a piecewise-homogeneous Poisson double peak
  (morning 7:30-9:30 lam=0.09/min, midday lam=0.03/min, evening 17:30-19:00
  lam=0.14/min, seeded); destinations uniform over the 23 customer nodes;
  weight/service/window distributions mirror ``generator.sample_scenario``
  defaults (weight U(0.5, 4.5) kg, service U(1, 3) min, e = release + U(0, 15),
  l = e + U(30, 90)).

Key hygiene: the Amap key is read from ``~/.dagcd/amap_key.env`` at run time,
never printed, and never written into any file of this repository; raw API
responses (which contain no key material) are cached under ``amap_raw/``;
API usage is tracked in ``amap_raw/_ledger.json`` and hard-capped.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dagcd.generator import Scenario, InstanceParams, save_scenario, _sample_weather
from dagcd.instance import Node, Order, GroundVehicle, Drone, NFZ

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(HERE, "amap_raw")
LEDGER = os.path.join(RAW_DIR, "_ledger.json")
KEY_ENV = os.path.expanduser("~/.dagcd/amap_key.env")
SCENARIO_JSON = os.path.join(HERE, "scenario.json")
CIRC_JSON = os.path.join(HERE, "case_circuity.json")
NODES_JSON = os.path.join(HERE, "case_nodes.json")

# ---------------- fixed seeds (archived) ----------------
PAIR_SEED = 2026092301      # OD-pair sampling for circuity fit/validation
ORDER_SEED = 2026092302     # order stream (releases, destinations, weights, windows)
WEATHER_SEED = 2026092303   # weather Markov chain

# ---------------- geography ----------------
# (key, name, lat, lon) — approximate real locations, hand-picked on the map.
LAT0, LON0 = 22.55, 113.93          # equirectangular anchor
R_EARTH = 6378137.0                 # WGS84 spherical radius [m]
M_PER_DEG_LAT = math.pi * R_EARTH / 180.0
M_PER_DEG_LON = M_PER_DEG_LAT * math.cos(math.radians(LAT0))

COORDS = [
    # depot
    ("depot_qianhai", "Depot 前海综合保税区物流园", 22.5320, 113.8950),
    # vertiports
    ("vp_techpark", "Vertiport 南山科技园(深大北门)", 22.5385, 113.9510),
    ("vp_xixiang", "Vertiport 西乡(西乡地铁站周边空地)", 22.5910, 113.8640),
    ("vp_baoancenter", "Vertiport 宝安中心(宝安体育馆东侧广场)", 22.5535, 113.8840),
    # customers
    ("cu_szu", "深圳大学粤海校区", 22.5327, 113.9382),
    ("cu_tech_south", "科技园南区(虚拟大学园)", 22.5308, 113.9560),
    ("cu_xili", "西丽(留仙大道沿线)", 22.5860, 113.9650),
    ("cu_fanshen", "翻身片区", 22.5655, 113.8870),
    ("cu_pingzhou", "坪洲", 22.5690, 113.8600),
    ("cu_gushu", "固戍", 22.6070, 113.8450),
    ("cu_coastcity", "南山海岸城", 22.5210, 113.9400),
    ("cu_shekou", "蛇口网谷", 22.4955, 113.9180),
    ("cu_guiwan", "前海桂湾(深港基金小镇)", 22.5380, 113.8920),
    ("cu_binhai", "宝安滨海文化公园", 22.5475, 113.8900),
    ("cu_liuxiandong", "留仙洞总部基地", 22.5780, 113.9430),
    ("cu_vankely", "万科云城", 22.5760, 113.9520),
    ("cu_szbayeco", "深圳湾科技生态园", 22.5260, 113.9490),
    ("cu_daxin", "大新(大新地铁站)", 22.5360, 113.9180),
    ("cu_nantou", "南头古城", 22.5410, 113.9200),
    ("cu_shajing", "沙井(沙井地铁站)", 22.6300, 113.8520),
    ("cu_lingzhi", "灵芝(灵芝公园)", 22.5710, 113.9050),
    ("cu_xingdong", "兴东(兴东地铁站)", 22.5800, 113.9220),
    ("cu_tanglang", "塘朗(塘朗地铁站)", 22.5950, 113.9780),
    ("cu_univtown", "深圳大学城", 22.5900, 113.9700),
    ("cu_talentpark", "深圳湾口岸/人才公园", 22.5120, 113.9320),
    ("cu_bihaiwan", "碧海湾", 22.5600, 113.8550),
    ("cu_nanyou", "南油(南油地铁站)", 22.5070, 113.9270),
]

# no-fly zones, polygons as (lat, lon) vertices, converted below
AIRPORT_NFZ_LATLON = [                  # Shenzhen Bao'an airport, simplified 5-vertex
    (22.612, 113.792), (22.658, 113.795), (22.662, 113.830),
    (22.640, 113.842), (22.613, 113.825),
]
TEMP_NFZ_CENTER = (22.5485, 113.8930)   # temporary event restriction near Bao'an Stadium
TEMP_NFZ_HALF_M = 120.0                 # axis-aligned square, half side [m]
AIRPORT_NFZ_WINDOW = (480.0, 660.0)     # 15:00-18:00, temporary afternoon activation
TEMP_NFZ_WINDOW = (600.0, 720.0)        # 17:00-19:00 (sim ends 19:00)

# ---------------- environment parameters ----------------
SIM_END = 720.0                 # 07:00 -> 19:00
HORIZON = 240.0
CONG_LEVEL = 0.6
V_G = 350.0                     # ground base speed, road-equivalent m/min
DRONE_CRUISE = 600.0            # m/min (package default)
DRONE_PAYLOAD, DRONE_BATTERY, DRONE_SORTIE_MAX, DRONE_MAX_PARCEL = 5.0, 500.0, 60.0, 3
N_VEHICLES, VEH_CAPACITY = 4, 40.0

WX_CHAIN_RAINY = {              # rows: clear/windy/rain/storm; rainy-season tendency
    "clear": [0.55, 0.20, 0.20, 0.05],
    "windy": [0.30, 0.35, 0.25, 0.10],
    "rain":  [0.25, 0.25, 0.35, 0.15],
    "storm": [0.25, 0.30, 0.30, 0.15],
}
MEAN_EVENT_MIN = 40.0

# order release process: (start_min, end_min, lambda per min), minutes from 07:00
RELEASE_SEGMENTS = [(30.0, 150.0, 0.09), (150.0, 570.0, 0.03), (570.0, 720.0, 0.14)]

# Amap quota guard
QUOTA_BUDGET = 120              # conservative daily call budget
MAX_ATTEMPTS = 3                # exponential-backoff retries per OD pair
HTTP_TIMEOUT = 15.0
CALL_SPACING_S = 0.25


def to_local(lat: float, lon: float) -> tuple[float, float]:
    """(lat, lon) -> local metric (x east, y north) [m], equirectangular."""
    return ((lon - LON0) * M_PER_DEG_LON, (lat - LAT0) * M_PER_DEG_LAT)


# ---------------- Amap client ----------------

def _load_key() -> str:
    key = None
    with open(KEY_ENV, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("AMAP_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise RuntimeError(f"AMAP_KEY not found in {KEY_ENV}")
    return key


def _read_ledger() -> dict:
    if os.path.exists(LEDGER):
        with open(LEDGER, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"total_calls": 0, "runs": []}


def _write_ledger(led: dict) -> None:
    tmp = LEDGER + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(led, f, indent=2)
    os.replace(tmp, LEDGER)


_RETRY_INFOCODES = {"10001", "10003", "10004", "10009", "10021", "10044", "10046"}


def amap_driving(olat: float, olon: float, dlat: float, dlon: float, key: str,
                 tag: str, ledger: dict, calls: dict) -> dict | None:
    """One OD query against the Amap driving-directions API (v3).

    Returns a record dict with road distance / duration on success, ``None``
    on failure (after <= MAX_ATTEMPTS attempts).  Every HTTP request is
    counted in ``calls`` and the run ledger; the key is never persisted.
    """
    base = "https://restapi.amap.com/v3/direction/driving"
    qs = urllib.parse.urlencode({
        "origin": f"{olon:.6f},{olat:.6f}",
        "destination": f"{dlon:.6f},{dlat:.6f}",
        "key": key, "output": "json",
    })
    rec = {"tag": tag, "origin_ll": [olat, olon], "destination_ll": [dlat, dlon]}
    last_err = "unknown"
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if ledger["total_calls"] + calls["n"] >= QUOTA_BUDGET:
            last_err = "quota budget exhausted"
            break
        req = urllib.request.Request(base + "?" + qs, headers={"User-Agent": "dagcd-case-study/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:                      # network/HTTP-level failure
            last_err = f"{type(exc).__name__}: {exc}"
            calls["n"] += 1
            time.sleep(2 ** (attempt - 1))
            continue
        calls["n"] += 1
        rec["amap_status"] = body.get("status")
        rec["amap_infocode"] = body.get("infocode")
        rec["amap_info"] = body.get("info")
        if body.get("status") == "1" and body.get("route", {}).get("paths"):
            path = body["route"]["paths"][0]
            rec["distance_m"] = float(path["distance"])
            rec["duration_s"] = float(path["duration"])
            rec["attempts"] = attempt
            rec["fetched_at"] = datetime.now(timezone.utc).isoformat()
            with open(os.path.join(RAW_DIR, tag + ".json"), "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2)
            return rec
        last_err = f"infocode={body.get('infocode')} info={body.get('info')}"
        if str(body.get("infocode")) in _RETRY_INFOCODES:
            time.sleep(2 ** (attempt - 1))
            continue
        break                                          # deterministic API error (e.g. no route)
    rec["error"] = last_err
    rec["attempts"] = attempt
    rec["fetched_at"] = datetime.now(timezone.utc).isoformat()
    with open(os.path.join(RAW_DIR, tag + ".json"), "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2)
    return None


# ---------------- circuity fitting ----------------

def _sample_pairs(nodes_xy: dict[int, tuple[float, float]]):
    """Stratified random OD-pair sample: 45 fit pairs (22 short / 15 mid / 8 long)
    + 8 validation pairs (3 / 3 / 2), disjoint, seeded by PAIR_SEED."""
    ids = sorted(nodes_xy)
    pairs = [(i, j) for a, i in enumerate(ids) for j in ids[a + 1:]]
    rng = random.Random(PAIR_SEED)

    def d2(p):
        (x1, y1), (x2, y2) = nodes_xy[p[0]], nodes_xy[p[1]]
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    pools = {"short": [], "mid": [], "long": []}
    for p in pairs:
        d = d2(p)
        pools["short" if d < 3000.0 else "mid" if d <= 8000.0 else "long"].append(p)
    for k in pools:
        rng.shuffle(pools[k])

    def take(pool, n_fit, n_val):
        return pool[:n_fit], pool[n_fit:n_fit + n_val]

    fit, val = [], []
    for name, n_fit, n_val in (("short", 22, 3), ("mid", 15, 3), ("long", 8, 2)):
        f, v = take(pools[name], n_fit, n_val)
        fit.extend(f)
        val.extend(v)
    rng.shuffle(fit)
    rng.shuffle(val)
    return fit, val


def fit_circuity(nodes_ll: dict[int, tuple[float, float]], refetch: bool) -> dict:
    """Fit the single circuity scalar; validate on held-out pairs. Uses the
    amap_raw/ cache unless ``refetch`` is set."""
    os.makedirs(RAW_DIR, exist_ok=True)
    nodes_xy = {nid: to_local(*ll) for nid, ll in nodes_ll.items()}
    fit_pairs, val_pairs = _sample_pairs(nodes_xy)
    ledger = _read_ledger()
    key = None
    if refetch:
        key = _load_key()
    calls = {"n": 0}

    def get(pair) -> dict | None:
        i, j = pair
        tag = f"od_{i:02d}_{j:02d}"
        cached = os.path.join(RAW_DIR, tag + ".json")
        if not refetch and os.path.exists(cached):
            with open(cached, "r", encoding="utf-8") as f:
                rec = json.load(f)
            return rec if "distance_m" in rec else None
        rec = amap_driving(nodes_ll[i][0], nodes_ll[i][1], nodes_ll[j][0],
                           nodes_ll[j][1], key, tag, ledger, calls)
        time.sleep(CALL_SPACING_S)
        return rec

    rows = []
    for split, pairs in (("fit", fit_pairs), ("validation", val_pairs)):
        for pair in pairs:
            rec = get(pair)
            (x1, y1), (x2, y2) = nodes_xy[pair[0]], nodes_xy[pair[1]]
            euclid = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            row = {"split": split, "i": pair[0], "j": pair[1],
                   "euclid_m": round(euclid, 3)}
            if rec and "distance_m" in rec:
                row.update(distance_m=rec["distance_m"], duration_s=rec["duration_s"])
                row["circuity"] = rec["distance_m"] / euclid
            else:
                row["failed"] = True
            rows.append(row)

    ledger["total_calls"] += calls["n"]
    if calls["n"]:
        ledger["runs"].append({"at": datetime.now(timezone.utc).isoformat(),
                               "calls": calls["n"], "refetch": refetch})
        _write_ledger(ledger)

    fit_rows = [r for r in rows if r["split"] == "fit" and not r.get("failed")]
    n_failed = sum(1 for r in rows if r.get("failed"))
    circs = [r["circuity"] for r in fit_rows]
    n = len(circs)
    if n < 30:
        raise RuntimeError(
            f"only {n} usable fit OD pairs (< 30 minimum); "
            "stopping per protocol — investigate failures in amap_raw/ before retrying")
    mean_c = sum(circs) / n
    var = sum((c - mean_c) ** 2 for c in circs) / (n - 1)
    val_rows = [r for r in rows if r["split"] == "validation" and not r.get("failed")]
    val_errs = [abs(mean_c * r["euclid_m"] - r["distance_m"]) / r["distance_m"]
                for r in val_rows]
    val_mean = sum(val_errs) / len(val_errs) if val_errs else None
    durs = [r["distance_m"] / r["duration_s"] for r in fit_rows]   # m/s sanity
    stats = {
        "method": "single scalar circuity = mean(road/euclid) over stratified "
                  "random OD pairs (22 short<3km, 15 mid 3-8km, 8 long>8km); "
                  "validated on 8 held-out pairs (3/3/2)",
        "pair_seed": PAIR_SEED,
        "n_fit_pairs": n, "n_failed_pairs": n_failed,
        "n_validation_pairs": len(val_rows),
        "api_calls_this_run": calls["n"],
        "api_calls_ledger_total": ledger["total_calls"],
        "circuity_mean": mean_c,
        "circuity_std": math.sqrt(var),
        "circuity_min": min(circs),
        "circuity_max": max(circs),
        "validation_mean_abs_rel_error": val_mean,
        "validation_rel_errors": val_errs,
        "amap_duration_mean_kmh": (sum(durs) / len(durs)) * 3.6,
        "model_free_flow_kmh": V_G / mean_c * 0.06,
        "rows": rows,
    }
    with open(CIRC_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return stats


# ---------------- environment construction ----------------

def build_congestion() -> list[tuple[float, float, float]]:
    """Shenzhen rush hours 7:30-9:30 / 17:30-20:00 at cong_level=0.6
    (peak factor 1 - 0.2*c, shoulder 1 - 0.4*c, 60-min shoulders; t in min
    from 07:00, same shape as generator._congestion_table but double-peaked)."""
    c_peak = 1.0 - 0.2 * CONG_LEVEL
    c_sh = 1.0 - 0.4 * CONG_LEVEL
    segs = [(0.0, 30.0, c_sh), (30.0, 150.0, c_peak), (150.0, 210.0, c_sh),
            (210.0, 570.0, 1.0), (570.0, 630.0, c_sh), (630.0, 720.0, c_peak)]
    return segs


def build_orders(customer_ids: list[int], order_seed: int) -> dict[str, Order]:
    """Hand-built order stream (seed ``order_seed``): piecewise-homogeneous
    Poisson double peak + generator-default weight/service/window draws.
    Destinations uniform over customer nodes."""
    rng = random.Random(order_seed)
    releases: list[float] = []
    for a, b, lam in RELEASE_SEGMENTS:
        n = 0
        limit = math.exp(-lam * (b - a))
        p = 1.0
        while p > limit:
            n += 1
            p *= rng.random()
        n -= 1
        releases.extend(a + (b - a) * rng.random() for _ in range(n))
    releases.sort()
    orders: dict[str, Order] = {}
    for k, t in enumerate(releases):
        node = customer_ids[rng.randrange(len(customer_ids))]
        weight = rng.uniform(0.5, 0.9 * DRONE_PAYLOAD)
        service = rng.uniform(1.0, 3.0)
        e = t + rng.uniform(0.0, 15.0)
        l = e + rng.uniform(30.0, 90.0)
        orders[f"o{k + 1}"] = Order(f"o{k + 1}", node, t, weight, service, e, l)
    return orders


def build_scenario(circuity: float, order_seed: int, wx_seed: int) -> Scenario:
    nodes: dict[int, Node] = {}
    nodes_ll: dict[int, tuple[float, float]] = {}
    names: dict[str, str] = {}
    customer_ids: list[int] = []
    for nid, (key, name, lat, lon) in enumerate(COORDS):
        x, y = to_local(lat, lon)
        kind = "depot" if nid == 0 else "vertiport" if nid <= 3 else "customer"
        if kind == "customer":
            customer_ids.append(nid)
        nodes[nid] = Node(nid, x * circuity, y * circuity, kind)
        nodes_ll[nid] = (lat, lon)
        names[str(nid)] = name

    # NFZs in the scaled metric frame
    ap_poly = [[x * circuity, y * circuity] for x, y in
               (to_local(lat, lon) for lat, lon in AIRPORT_NFZ_LATLON)]
    cx, cy = to_local(*TEMP_NFZ_CENTER)
    h = TEMP_NFZ_HALF_M
    temp_poly = [[cx * circuity - h, cy * circuity - h],
                 [cx * circuity + h, cy * circuity - h],
                 [cx * circuity + h, cy * circuity + h],
                 [cx * circuity - h, cy * circuity + h]]
    nfzs = [NFZ("z_airport_szx", ap_poly, active_from=AIRPORT_NFZ_WINDOW[0],
                active_to=AIRPORT_NFZ_WINDOW[1]),
            NFZ("z_temp_baoan_event", temp_poly, active_from=TEMP_NFZ_WINDOW[0],
                active_to=TEMP_NFZ_WINDOW[1])]

    params = InstanceParams(
        lam_peak=0.14, lam_trough_ratio=0.03 / 0.14, peak_start=570.0, peak_end=720.0,
        n_clusters=len(customer_ids), cluster_spread_m=0.0,
        n_customers_area=13700.0, cong_level=CONG_LEVEL,
        n_depots=1, n_vertiports=3, wx_chain=WX_CHAIN_RAINY,
        mean_event_min=MEAN_EVENT_MIN, nfz_density=2.0 / (SIM_END / 60.0),
        n_vehicles=N_VEHICLES, vehicle_capacity=VEH_CAPACITY, n_drones=3,
        drone_payload=DRONE_PAYLOAD, drone_battery_wh=DRONE_BATTERY,
        horizon=HORIZON, sim_duration=SIM_END)

    wx_rng = random.Random(wx_seed)
    weather = _sample_weather(wx_rng, params, SIM_END)

    vehicles = {f"gv{i}": GroundVehicle(f"gv{i}", VEH_CAPACITY, 0, 0.0, SIM_END)
                for i in range(N_VEHICLES)}
    drones = {f"da{i}": Drone(f"da{i}", DRONE_PAYLOAD, DRONE_BATTERY, DRONE_CRUISE,
                              DRONE_SORTIE_MAX, DRONE_MAX_PARCEL)
              for i in range(3)}
    orders = build_orders(customer_ids, order_seed)

    sc = Scenario(params=params, nodes=nodes, orders=orders, vehicles=vehicles,
                  drones=drones, nfzs=nfzs, weather=weather,
                  congestion=build_congestion(), sim_end=SIM_END,
                  v_g=V_G,
                  energy_alpha=0.02 / circuity, energy_beta=0.0005 / circuity)
    # record node coordinates/names for documentation and audit
    with open(NODES_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "anchor": {"lat": LAT0, "lon": LON0},
            "m_per_deg_lat": M_PER_DEG_LAT, "m_per_deg_lon": M_PER_DEG_LON,
            "coordinate_scaling": "scenario x,y = local_m * circuity (road-equivalent)",
            "nodes": [{"id": nid, "name": names[str(nid)], "kind": n.kind,
                       "lat": nodes_ll[nid][0], "lon": nodes_ll[nid][1],
                       "local_x_m": round((n.x / circuity), 3),
                       "local_y_m": round((n.y / circuity), 3),
                       "scenario_x": n.x, "scenario_y": n.y}
                      for nid, n in sorted(nodes.items())],
        }, f, indent=2, ensure_ascii=False)
    return sc


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refetch", action="store_true",
                    help="re-call the Amap API instead of using amap_raw/ cache")
    ap.add_argument("--order-seed", type=int, default=ORDER_SEED,
                    help=f"order-stream seed (default {ORDER_SEED} = archived day0)")
    ap.add_argument("--wx-seed", type=int, default=WEATHER_SEED,
                    help=f"weather-chain seed (default {WEATHER_SEED} = archived day0)")
    ap.add_argument("--out", default=SCENARIO_JSON,
                    help=f"output scenario path (default {SCENARIO_JSON})")
    args = ap.parse_args()
    nodes_ll = {nid: (lat, lon) for nid, (_k, _n, lat, lon) in enumerate(COORDS)}
    stats = fit_circuity(nodes_ll, args.refetch)
    c = stats["circuity_mean"]
    print(f"circuity fit: n={stats['n_fit_pairs']} mean={c:.4f} "
          f"std={stats['circuity_std']:.4f} [{stats['circuity_min']:.3f}, "
          f"{stats['circuity_max']:.3f}]")
    print(f"validation mean |pred-real|/real = {stats['validation_mean_abs_rel_error']:.4f} "
          f"over {stats['n_validation_pairs']} pairs")
    print(f"API calls this run: {stats['api_calls_this_run']} "
          f"(ledger total {stats['api_calls_ledger_total']}/{QUOTA_BUDGET}); "
          f"failed pairs: {stats['n_failed_pairs']}")
    print(f"amap duration mean speed {stats['amap_duration_mean_kmh']:.1f} km/h vs "
          f"model free-flow {stats['model_free_flow_kmh']:.1f} km/h (sanity only)")

    sc = build_scenario(c, args.order_seed, args.wx_seed)
    save_scenario(sc, args.out)
    n_rel = len(sc.orders)
    storms = sum(1 for w in sc.weather if w.state == "storm")
    rains = sum(1 for w in sc.weather if w.state == "rain")
    print(f"scenario saved -> {args.out}: {len(sc.nodes)} nodes "
          f"({sum(1 for n in sc.nodes.values() if n.kind == 'customer')} customers), "
          f"{n_rel} orders, {len(sc.weather)} weather episodes "
          f"(rain {rains}, storm {storms}), {len(sc.nfzs)} NFZs, v_g={sc.v_g}, "
          f"energy_alpha={sc.energy_alpha:.6f}; seeds order={args.order_seed} wx={args.wx_seed}")


if __name__ == "__main__":
    main()

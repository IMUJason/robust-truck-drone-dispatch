# Shenzhen case study

An urban low-altitude logistics scenario built on the Nanshan–Bao'an built-up
area of Shenzhen, China. All coordinates are approximate real locations
hand-picked on a public map; road distances are modeled by a fitted circuity
factor over equirectangular-projected geometry.

## Files

| File | Contents |
|---|---|
| `case_nodes.json` | the 27 service nodes (1 depot, 3 vertiports, 23 customers): WGS84 coordinates, local metric coordinates (equirectangular anchor 22.55 N, 113.93 E), and the road-equivalent scenario coordinates |
| `case_circuity.json` | the circuity fit: method, seed, per-pair records, fit and held-out validation statistics |
| `scenario.json` | operating day 0 — the canonical scenario (52 orders, 23 weather episodes, 2 scheduled NFZs, 07:00–19:00) |
| `scenario_day1.json` … `scenario_day4.json` | four further operating days (variant order-stream and weather seeds; 39–55 orders each) |
| `build_case.py` | one-click rebuild of any scenario (offline from the cache, or against the live API) |
| `amap_raw/` | cached raw Amap (Gaode) driving-directions responses used for the circuity fit |

Load a scenario with `dagcd.load_scenario`, e.g.:

```python
from dagcd import load_scenario, run_simulation, GreedyController
scenario = load_scenario("case/scenario.json")
report = run_simulation(scenario, GreedyController(), seed=0)
```

## Road distances: circuity fit

Querying the full 27×27 road-distance matrix was not feasible under the API
quota, so road distance is approximated as `circuity × euclidean`. A stratified
random sample of **45 OD pairs** (22 short < 3 km, 15 mid 3–8 km, 8 long >
8 km, fixed seed) was queried against the Amap driving-directions API; the
fitted scalar is **circuity = 1.724 ± 0.537** (mean ± std of road/euclidean).
On **8 held-out pairs** the approximation attains a mean absolute relative
error of **26.3%**. The fit is applied by scaling every node coordinate and
NFZ polygon, so all scenario geometry is in road-equivalent metres. The
API-reported driving durations are recorded in `case_circuity.json` as a
sanity statistic only and never enter the model.

## Environment

- **Congestion**: double-peaked Shenzhen rush hours (7:30–9:30, 17:30–20:00)
  at `cong_level = 0.6`, piecewise-constant on the package's standard curve
  shape.
- **Weather**: rainy-season Markov chain over `clear / windy / rain / storm`
  with storm-permitting transitions (subtropical monsoon climate), mean
  episode length 40 min.
- **No-fly zones**: a simplified 5-vertex approximation of Shenzhen Bao'an
  airport (terminal + runway strip), active 15:00–18:00, plus a small
  temporary zone near Bao'an Stadium, active 17:00–19:00.
- **Orders**: piecewise-homogeneous Poisson releases with a morning and an
  evening peak; destinations uniform over the 23 customer nodes;
  weight/service/window distributions follow the `dagcd` generator defaults.
- **Fleet**: 4 ground vehicles (40 kg, shift 07:00–19:00), 3 drones
  (5 kg payload, 500 Wh, 60 min endurance, ≤ 3 parcels per sortie).

## Rebuilding

```bash
python3 build_case.py            # offline: reuse the amap_raw/ cache (0 API calls)
python3 build_case.py --refetch  # re-query the Amap driving API (53 calls)
python3 build_case.py --order-seed 1 --wx-seed 1 --out scenario_day1.json
```

The default seeds reproduce `scenario.json` byte-identically. The published
case numbers use the frozen rolling-horizon protocol `epoch_delta=15.0`,
`n_trig=5`, `seed=0` (the package defaults are `epoch_delta=30.0`, `n_trig=5`).

`--refetch`
requires your own Amap (Gaode) Web-service key in `~/.dagcd/amap_key.env`
(one line: `AMAP_KEY=...`); the key is read at run time, never printed, and
never written into any file of this repository. A call ledger
(`amap_raw/_ledger.json`) hard-caps API usage.

## Provenance and privacy

`amap_raw/` contains the raw API responses exactly as returned, one JSON per
queried OD pair (origin/destination coordinates, status codes, distance,
duration, fetch timestamp). They contain no key material and no personal data.
The cached responses let anyone rebuild the scenarios offline without an API
key.

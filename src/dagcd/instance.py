"""Instance & solution data model for DAGCD (see docs/SPECIFICATION.md). JSON-serializable dataclasses."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Any

WEATHER_STATES = ("clear", "windy", "rain", "storm")
TOL = 1e-6


@dataclass
class Node:
    id: int
    x: float
    y: float
    kind: str = "customer"  # "customer" | "depot" | "vertiport"


@dataclass
class Order:
    id: str
    node: int
    release: float          # r_o
    weight: float           # w_o (kg)
    service: float          # s_o (min)
    e: float                # window start
    l: float                # window end (soft)
    lateness_penalty: float = 1.0
    unserved_penalty: float = 50.0


@dataclass
class GroundVehicle:
    id: str
    capacity: float         # Q_g kg
    start_depot: int
    start_time: float
    shift_end: float


@dataclass
class Drone:
    id: str
    payload: float          # Q_a kg
    battery_wh: float       # B
    cruise_speed: float     # v_a (m/min)
    max_sortie_min: float   # L_max
    max_parcel: int         # M_max parcels per sortie


@dataclass
class NFZ:
    id: str
    polygon: list[list[float]]  # [[x,y], ...]
    active_from: float | None = None   # activation schedule; None = active over whole horizon
    active_to: float | None = None     # active while active_from <= t < active_to


@dataclass
class WeatherEvent:
    t_start: float
    t_end: float
    state: str                  # one of WEATHER_STATES


@dataclass
class Instance:
    nodes: dict[int, Node]
    orders: dict[str, Order]
    vehicles: dict[str, GroundVehicle]
    drones: dict[str, Drone]
    nfzs: list[NFZ]
    weather: list[WeatherEvent]          # non-overlapping, sorted
    horizon: float                       # T
    v_g: float = 500.0                   # ground base speed (m/min)
    congestion: list[tuple[float, float, float]] = field(default_factory=list)
    # (t_start, t_end, factor c(t) in (0,1]); sorted, covers [0,T]
    speed_factor: dict[str, float] = field(default_factory=lambda:
        {"clear": 1.0, "windy": 0.9, "rain": 0.75, "storm": 0.0})
    energy_alpha: float = 0.02           # Wh per m (empty)
    energy_beta: float = 0.0005          # Wh per m per kg
    energy_mu: dict[str, float] = field(default_factory=lambda:
        {"clear": 1.0, "windy": 1.2, "rain": 1.5, "storm": 1.0})

    # ---- derived helpers ----
    def dist(self, i: int, j: int) -> float:
        """Euclidean distance [m].

        Reproducibility: uses ``sqrt(dx*dx + dy*dy)`` — IEEE-754 multiply/add
        and sqrt are correctly rounded and bit-deterministic across platforms
        and CPython minor versions, unlike ``math.hypot`` (libm-dependent;
        verified to differ between CPython 3.9 and 3.10, which silently
        changes simulation geometry and every archived J).
        """
        a, b = self.nodes[i], self.nodes[j]
        dx, dy = a.x - b.x, a.y - b.y
        return math.sqrt(dx * dx + dy * dy)

    def congestion_factor(self, t: float) -> float:
        for a, b, c in self.congestion:
            if a - TOL <= t < b:
                return c
        return self.congestion[-1][2] if self.congestion else 1.0

    def tau_g(self, i: int, j: int, t: float) -> float:
        return self.dist(i, j) / (self.v_g * self.congestion_factor(t))

    def weather_at(self, t: float) -> str:
        for w in self.weather:
            if w.t_start - TOL <= t < w.t_end:
                return w.state
        return "clear"

    def tau_a(self, i: int, j: int, t: float, cruise_speed: float) -> float:
        sf = self.speed_factor.get(self.weather_at(t), 1.0)
        if sf <= 0.0:
            return math.inf
        return self.dist(i, j) / (cruise_speed * sf)

    def active_nfzs(self, t: float) -> list[NFZ]:
        # NFZs carry an optional activation schedule (None = active over whole horizon).
        return [z for z in self.nfzs
                if (z.active_from is None or z.active_from <= t)
                and (z.active_to is None or t < z.active_to)]


@dataclass
class GroundStop:
    order: str
    node: int
    arrival: float          # claimed service start


@dataclass
class GroundRoute:
    vehicle: str
    stops: list[GroundStop]
    return_depot: int
    return_time: float


@dataclass
class DroneStop:
    order: str
    node: int
    arrival: float


@dataclass
class DroneSortie:
    drone: str
    launch_vertiport: int
    launch_time: float
    stops: list[DroneStop]
    land_vertiport: int
    land_time: float
    energy_wh: float        # claimed (checker re-derives)


@dataclass
class Solution:
    epoch: int
    ground_routes: list[GroundRoute]
    drone_sorties: list[DroneSortie]
    unserved: list[str]


# ---------------- JSON I/O ----------------

def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError


def instance_to_dict(inst: Instance) -> dict:
    d = asdict(inst)
    d["congestion"] = [list(x) for x in inst.congestion]
    return d


def instance_from_dict(d: dict) -> Instance:
    return Instance(
        nodes={int(k): Node(**v) for k, v in d["nodes"].items()},
        orders={k: Order(**v) for k, v in d["orders"].items()},
        vehicles={k: GroundVehicle(**v) for k, v in d["vehicles"].items()},
        drones={k: Drone(**v) for k, v in d["drones"].items()},
        nfzs=[NFZ(**z) for z in d.get("nfzs", [])],
        weather=[WeatherEvent(**w) for w in d.get("weather", [])],
        horizon=d["horizon"],
        v_g=d.get("v_g", 500.0),
        congestion=[tuple(x) for x in d.get("congestion", [])],
        speed_factor=d.get("speed_factor"),
        energy_alpha=d.get("energy_alpha", 0.02),
        energy_beta=d.get("energy_beta", 0.0005),
        energy_mu=d.get("energy_mu"),
    )


def solution_from_dict(d: dict) -> Solution:
    return Solution(
        epoch=d.get("epoch", 0),
        ground_routes=[GroundRoute(
            vehicle=r["vehicle"],
            stops=[GroundStop(**s) for s in r["stops"]],
            return_depot=r["return_depot"],
            return_time=r["return_time"]) for r in d.get("ground_routes", [])],
        drone_sorties=[DroneSortie(
            drone=s["drone"],
            launch_vertiport=s["launch_vertiport"],
            launch_time=s["launch_time"],
            stops=[DroneStop(**x) for x in s["stops"]],
            land_vertiport=s["land_vertiport"],
            land_time=s["land_time"],
            energy_wh=s.get("energy_wh", 0.0)) for s in d.get("drone_sorties", [])],
        unserved=list(d.get("unserved", [])),
    )


def load_instance(path: str) -> Instance:
    with open(path, "r", encoding="utf-8") as f:
        return instance_from_dict(json.load(f))


def load_solution(path: str) -> Solution:
    with open(path, "r", encoding="utf-8") as f:
        return solution_from_dict(json.load(f))

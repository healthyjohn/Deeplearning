"""Random road-cross-section scenes for synthetic GPR B-scans.

A scene is a 2D slice under a road: horizontal layers (asphalt, base,
subgrade) plus buried targets (air-filled cavities and metal pipes).
The same scene description drives both the gprMax simulation and the
YOLO label calculation, so the labels always match the image.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field

import numpy as np

C0 = 299_792_458.0  # speed of light, m/s

CLASS_NAMES = ["cavity", "pipe"]


@dataclass
class Layer:
    name: str
    top: float  # depth of layer top below the road surface (m)
    eps: float  # relative permittivity
    sigma: float  # conductivity (S/m)


@dataclass
class Target:
    cls: str  # "cavity" or "pipe"
    x: float  # horizontal centre (m, model coordinates)
    depth: float  # centre depth below the road surface (m)
    radius: float  # m

    @property
    def class_id(self) -> int:
        return CLASS_NAMES.index(self.cls)


@dataclass
class Scene:
    width: float = 1.6  # model width (m)
    depth: float = 1.0  # ground thickness modelled (m)
    air: float = 0.1  # air above the surface (m)
    dx: float = 0.006  # cell size (m)
    freq: float = 600e6  # Ricker centre frequency (Hz)
    trace_step: float = 0.025  # antenna move per trace (m)
    rx_offset: float = 0.04  # Tx-Rx separation (m)
    margin: float = 0.12  # keep antennas away from the absorbing boundary (m)
    time_window: float = 0.0  # s, filled by finalize()
    layers: list[Layer] = field(default_factory=list)
    targets: list[Target] = field(default_factory=list)

    # ---- geometry helpers -------------------------------------------------
    @property
    def surface_y(self) -> float:
        return self.depth

    @property
    def antenna_y(self) -> float:
        return self.surface_y + 2 * self.dx

    @property
    def n_traces(self) -> int:
        span = self.width - 2 * self.margin - self.rx_offset
        return int(math.floor(span / self.trace_step)) + 1

    def trace_positions(self) -> np.ndarray:
        """Midpoint between Tx and Rx for each trace (m)."""
        x0 = self.margin + self.rx_offset / 2
        return x0 + self.trace_step * np.arange(self.n_traces)

    def eps_at(self, z: float) -> float:
        eps = self.layers[0].eps
        for layer in self.layers:
            if z >= layer.top:
                eps = layer.eps
        return eps

    def two_way_time(self, z: float) -> float:
        """Vertical two-way travel time from the surface to depth z (s)."""
        tops = [l.top for l in self.layers] + [float("inf")]
        t = 0.0
        for i, layer in enumerate(self.layers):
            z0, z1 = tops[i], min(tops[i + 1], z)
            if z1 > z0:
                t += 2 * (z1 - z0) * math.sqrt(layer.eps) / C0
        return t

    def mean_velocity(self, z: float) -> float:
        z = max(z, 1e-3)
        return 2 * z / self.two_way_time(z)

    def finalize(self) -> "Scene":
        if not self.time_window:
            t = self.two_way_time(self.depth * 0.95)
            self.time_window = t * 1.05 + 3.0 / self.freq
        return self

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Scene":
        d = dict(d)
        d["layers"] = [Layer(**l) for l in d.get("layers", [])]
        d["targets"] = [Target(**t) for t in d.get("targets", [])]
        return cls(**d)


def random_scene(rng: np.random.Generator, max_targets: int = 2, empty_prob: float = 0.1, **overrides) -> Scene:
    """Draw a random road cross-section."""
    s = Scene(**overrides)
    asphalt = rng.uniform(0.05, 0.25)
    base = asphalt + rng.uniform(0.15, 0.35)
    s.layers = [
        Layer("asphalt", 0.0, rng.uniform(4.0, 7.0), rng.uniform(0.001, 0.005)),
        Layer("base", asphalt, rng.uniform(6.0, 10.0), rng.uniform(0.002, 0.01)),
        Layer("subgrade", base, rng.uniform(8.0, 16.0), rng.uniform(0.002, 0.02)),
    ]

    n = 0 if rng.random() < empty_prob else int(rng.integers(1, max_targets + 1))
    xs = s.trace_positions()
    lo, hi = xs[0] + 0.1, xs[-1] - 0.1
    for _ in range(50):
        if len(s.targets) >= n:
            break
        cls = "cavity" if rng.random() < 0.6 else "pipe"
        r = rng.uniform(0.05, 0.22) if cls == "cavity" else rng.uniform(0.02, 0.10)
        d = rng.uniform(max(asphalt + r + 0.05, 0.2), s.depth * 0.75)
        x = rng.uniform(lo, hi)
        if all(abs(x - t.x) > (r + t.radius + 0.15) for t in s.targets):
            s.targets.append(Target(cls, float(x), float(d), float(r)))
    return s.finalize()

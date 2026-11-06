"""Turn a Scene into a raw B-scan.

Two engines:
- "gprmax": full FDTD physics with gprMax (slow, realistic).
- "analytic": fast ray-based approximation (seconds per thousand scenes),
  useful for smoke tests or pre-training before gprMax data is ready.

Both return (bscan, dt): bscan has shape (n_samples, n_traces).
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
import tempfile

import numpy as np

from .scenes import C0, Scene


# ---------------------------------------------------------------- gprMax --
def gprmax_input(scene: Scene, title: str = "gpr_kiln scene") -> str:
    s = scene
    dz = s.dx  # 2D model: one cell thick
    lines = [
        f"#title: {title}",
        f"#domain: {s.width:.4f} {s.depth + s.air:.4f} {dz:.4f}",
        f"#dx_dy_dz: {s.dx:.4f} {s.dx:.4f} {dz:.4f}",
        f"#time_window: {s.time_window:.4e}",
    ]
    for layer in s.layers:
        lines.append(f"#material: {layer.eps:.3f} {layer.sigma:.4f} 1 0 {layer.name}")
    # layers top-down; each box spans from the layer top to the bottom of the model
    for layer in s.layers:
        y_top = s.surface_y - layer.top
        lines.append(f"#box: 0 0 0 {s.width:.4f} {y_top:.4f} {dz:.4f} {layer.name}")
    for t in s.targets:
        mat = "free_space" if t.cls == "cavity" else "pec"
        y = s.surface_y - t.depth
        lines.append(f"#cylinder: {t.x:.4f} {y:.4f} 0 {t.x:.4f} {y:.4f} {dz:.4f} {t.radius:.4f} {mat}")
    tx = s.margin
    lines += [
        f"#waveform: ricker 1 {s.freq:.4e} src_wave",
        f"#hertzian_dipole: z {tx:.4f} {s.antenna_y:.4f} 0 src_wave",
        f"#rx: {tx + s.rx_offset:.4f} {s.antenna_y:.4f} 0",
        f"#src_steps: {s.trace_step:.4f} 0 0",
        f"#rx_steps: {s.trace_step:.4f} 0 0",
    ]
    return "\n".join(lines) + "\n"


def simulate_gprmax(scene: Scene, workdir: str | None = None, keep_files: bool = False) -> tuple[np.ndarray, float]:
    import h5py

    tmp = None
    if workdir is None:
        tmp = tempfile.TemporaryDirectory(prefix="gprmax_")
        workdir = tmp.name
    os.makedirs(workdir, exist_ok=True)
    path = os.path.join(workdir, "scene.in")
    with open(path, "w") as f:
        f.write(gprmax_input(scene))

    n = scene.n_traces
    cmd = [sys.executable, "-m", "gprMax", path, "-n", str(n), "--geometry-fixed"]
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"gprMax failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")

    files = sorted(
        glob.glob(os.path.join(workdir, "scene*.h5")),
        key=lambda p: int(os.path.basename(p)[5:-3] or 0),
    )
    if len(files) != n:
        raise RuntimeError(f"expected {n} gprMax outputs, found {len(files)} in {workdir}")
    traces = []
    dt = None
    for fp in files:
        with h5py.File(fp, "r") as h:
            traces.append(np.asarray(h["rxs/rx1/Ez"], dtype=np.float32))
            dt = float(h.attrs["dt"])
        if not keep_files:
            os.remove(fp)
    if tmp is not None:
        tmp.cleanup()
    return np.stack(traces, axis=1), dt


# -------------------------------------------------------------- analytic --
def _ricker(t: np.ndarray, f: float) -> np.ndarray:
    a = (np.pi * f * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def simulate_analytic(scene: Scene, rng: np.random.Generator | None = None, n_samples: int = 512) -> tuple[np.ndarray, float]:
    rng = rng or np.random.default_rng()
    s = scene
    dt = s.time_window / n_samples
    t = np.arange(n_samples) * dt
    shift = np.sqrt(2) / s.freq  # gprMax Ricker delay
    xs = s.trace_positions()
    out = np.zeros((n_samples, len(xs)), dtype=np.float32)

    # direct (air/ground) wave
    direct = _ricker(t - shift - s.rx_offset / C0, s.freq)
    out += 1.0 * direct[:, None]

    # flat layer interfaces
    for upper, lower in zip(s.layers[:-1], s.layers[1:]):
        n1, n2 = np.sqrt(upper.eps), np.sqrt(lower.eps)
        r = (n1 - n2) / (n1 + n2)
        tl = s.two_way_time(lower.top)
        out += (0.8 * r * _ricker(t - shift - tl, s.freq))[:, None] * (1 + 0.05 * rng.standard_normal(len(xs)))[None, :]

    # targets: diffraction hyperbolas
    for tg in s.targets:
        v = s.mean_velocity(tg.depth)
        dist = np.sqrt((xs - tg.x) ** 2 + tg.depth**2) - tg.radius
        tt = 2 * dist / v
        n_t = np.sqrt(s.eps_at(tg.depth))
        refl = (n_t - 1) / (n_t + 1) if tg.cls == "cavity" else -1.0
        amp = refl * 0.25 / np.maximum(dist, 0.05) * np.exp(-1.2 * dist)
        amp *= np.cos(np.arctan2(np.abs(xs - tg.x), tg.depth)) ** 2
        out += amp[None, :] * _ricker(t[:, None] - shift - tt[None, :], s.freq)
        if tg.cls == "cavity":  # bottom-of-void echo
            tb = tt + 2 * 2 * tg.radius / C0
            out += 0.5 * amp[None, :] * _ricker(t[:, None] - shift - tb[None, :], s.freq)

    out += 0.004 * rng.standard_normal(out.shape).astype(np.float32)
    return out, dt


def simulate(scene: Scene, engine: str = "gprmax", rng: np.random.Generator | None = None, workdir: str | None = None):
    if engine == "gprmax":
        return simulate_gprmax(scene, workdir=workdir)
    if engine == "analytic":
        return simulate_analytic(scene, rng=rng)
    raise ValueError(f"unknown engine: {engine!r} (use 'gprmax' or 'analytic')")

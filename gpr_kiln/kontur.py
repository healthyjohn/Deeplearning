"""Use real KONTUR / Examiner data with the detector.

- load_segy / load_ascii : read "Processed Data: SEG-Y / ASCII" exports from Examiner
- detect_long            : slide a detector along a long road B-scan (tens of metres)
                           in tiles of the same physical size as the training scenes,
                           and report each detection in road distance (m) and depth (m)

Training scenes (profile "kontur-deep1808") cover ~2.6 m x 42 ns. A long B-scan is cut
into tiles of that size so hyperbolas have the same shape the model learned.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .imaging import IMG_H, IMG_W, process_bscan
from .scenes import C0, PROFILES, Scene


def profile_tile(profile: str = "kontur-deep1808") -> tuple[float, float]:
    """(tile length m, time window ns) of one training image for a profile."""
    from .scenes import random_scene

    if PROFILES[profile].get("time_window"):
        s = Scene(**PROFILES[profile]).finalize()
        tw = s.time_window
    else:  # time window depends on the random layers -> use the typical value
        rng = np.random.default_rng(0)
        scenes = [random_scene(rng, profile=profile) for _ in range(50)]
        s, tw = scenes[0], float(np.median([x.time_window for x in scenes]))
    xs = s.trace_positions()
    return float(xs[-1] - xs[0]), tw * 1e9


# ------------------------------------------------------------------ loaders --
def load_segy(path: str, channel: int | None = None, traces_per_channel: int | None = None) -> dict:
    """Read an Examiner SEG-Y export. Returns {"bscan": (samples, traces), "dt": s, "info": {...}}.

    Multi-channel files: pass traces_per_channel (and channel, 0-based) if channels are
    stored one after another. Check "info" against Examiner (time window, trace count).
    """
    import segyio

    with segyio.open(path, "r", ignore_geometry=True) as f:
        data = np.stack([np.asarray(t, dtype=np.float32) for t in f.trace], axis=1)
        interval = f.bin[segyio.BinField.Interval]  # usually microseconds; GPR exports often use ps
        n_samples = data.shape[0]
        info = {"n_traces": int(data.shape[1]), "n_samples": int(n_samples), "bin_interval_raw": int(interval)}
    # GPR time windows are tens of ns: pick the unit that gives a plausible window
    dt = None
    for unit in (1e-12, 1e-9, 1e-6):
        tw = interval * unit * n_samples
        if 5e-9 <= tw <= 500e-9:
            dt = interval * unit
            break
    info["dt_s"] = dt
    info["time_window_ns"] = dt * n_samples * 1e9 if dt else None
    if traces_per_channel:
        ch = channel or 0
        data = data[:, ch * traces_per_channel:(ch + 1) * traces_per_channel]
    return {"bscan": data, "dt": dt, "info": info}


def load_ascii(path: str, time_window_ns: float, transpose: bool | None = None) -> dict:
    """Read an Examiner ASCII export as a numeric matrix.

    The exact column layout was not verified; the matrix is taken as samples x traces
    (set transpose=True if traces are rows). Header lines that are not numbers are skipped.
    """
    rows = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.replace(",", " ").replace(";", " ").split()
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                continue
    width = max(set(len(r) for r in rows), key=[len(r) for r in rows].count)
    a = np.array([r for r in rows if len(r) == width], dtype=np.float32)
    if transpose is None:
        transpose = a.shape[0] > a.shape[1] * 4  # long road: traces >> samples
    if transpose:
        a = a.T
    return {"bscan": a, "dt": time_window_ns * 1e-9 / a.shape[0], "info": {"shape": list(a.shape)}}


def image_to_display(img: Image.Image) -> np.ndarray:
    """Examiner screenshot -> array in [-1, 1] with the same contrast mapping as training."""
    g = np.asarray(img.convert("L"), dtype=np.float32)
    g = g - np.median(g)
    scale = np.percentile(np.abs(g), 99.5) or 1.0
    return np.clip(g / scale, -1, 1)


# ---------------------------------------------------------------- detection --
def _nms(dets: list[dict], iou_thr: float = 0.3) -> list[dict]:
    dets = sorted(dets, key=lambda d: -d["confidence"])
    keep = []
    for d in dets:
        x0, x1 = d["x_from_m"], d["x_to_m"]
        t0, t1 = d["_t0"], d["_t1"]
        ok = True
        for k in keep:
            ix = max(0, min(x1, k["x_to_m"]) - max(x0, k["x_from_m"]))
            it = max(0, min(t1, k["_t1"]) - max(t0, k["_t0"]))
            inter = ix * it
            union = (x1 - x0) * (t1 - t0) + (k["x_to_m"] - k["x_from_m"]) * (k["_t1"] - k["_t0"]) - inter
            if union > 0 and inter / union > iou_thr:
                ok = False
                break
        if ok:
            keep.append(d)
    return keep


def detect_long(
    model_path: str,
    display: np.ndarray,
    distance_m: tuple[float, float],
    time_window_ns: float,
    eps: float = 7.0,
    profile: str = "kontur-deep1808",
    overlap: float = 0.5,
    conf: float = 0.25,
    annotate_path: str | None = None,
) -> dict:
    """Detect along a long B-scan.

    display: 2D array in [-1, 1] (rows = time from 0 to time_window_ns, cols = distance)
             e.g. process_bscan(segy["bscan"], dt) or image_to_display(screenshot).
    distance_m: road distance at the first and last column.
    """
    from ultralytics import YOLO

    net = YOLO(model_path)
    names = net.names
    H, W = display.shape
    length = distance_m[1] - distance_m[0]
    tile_m, tile_ns = profile_tile(profile)
    rows = min(H, int(round(H * tile_ns / time_window_ns)))  # time span of a training image
    cols = max(8, int(round(W * tile_m / length)))
    step = max(1, int(cols * (1 - overlap)))
    starts = list(range(0, max(W - cols, 0) + 1, step))
    if starts[-1] + cols < W:
        starts.append(W - cols)

    v = C0 / math.sqrt(eps)
    dets = []
    for c0 in starts:
        tile = display[:rows, c0:c0 + cols]
        img = Image.fromarray(((tile + 1) * 127.5).astype(np.uint8), mode="L").resize((IMG_W, IMG_H), Image.BILINEAR).convert("RGB")
        res = net.predict(img, conf=conf, iou=0.3, agnostic_nms=True, verbose=False)[0]
        for b in res.boxes:
            bx0, by0, bx1, by1 = [float(q) for q in b.xyxy[0]]
            gx0 = distance_m[0] + (c0 + bx0 / IMG_W * cols) / W * length
            gx1 = distance_m[0] + (c0 + bx1 / IMG_W * cols) / W * length
            t0 = by0 / IMG_H * rows / H * time_window_ns
            t1 = by1 / IMG_H * rows / H * time_window_ns
            dets.append({
                "class": names[int(b.cls[0])],
                "confidence": round(float(b.conf[0]), 3),
                "x_m": round((gx0 + gx1) / 2, 2),
                "x_from_m": round(gx0, 2),
                "x_to_m": round(gx1, 2),
                "depth_top_m": round(t0 * 1e-9 * v / 2, 2),  # apex ~ box top
                "_t0": t0, "_t1": t1,
            })
    dets = _nms(dets)
    for d in dets:
        d.pop("_t0"), d.pop("_t1")
    dets.sort(key=lambda d: d["x_m"])
    out = {"tile_m": round(tile_m, 2), "tile_ns": round(tile_ns, 1), "n_tiles": len(starts), "n_detections": len(dets), "detections": dets}
    if annotate_path:
        _annotate(display, dets, distance_m, time_window_ns, eps, annotate_path)
        out["annotated_image"] = annotate_path
    return out


def _annotate(display, dets, distance_m, time_window_ns, eps, path):
    from PIL import ImageDraw

    H, W = display.shape
    img = Image.fromarray(((display + 1) * 127.5).astype(np.uint8), mode="L").convert("RGB")
    d = ImageDraw.Draw(img)
    v = C0 / math.sqrt(eps)
    length = distance_m[1] - distance_m[0]
    for k in dets:
        x0 = (k["x_from_m"] - distance_m[0]) / length * W
        x1 = (k["x_to_m"] - distance_m[0]) / length * W
        y0 = k["depth_top_m"] * 2 / v / (time_window_ns * 1e-9) * H
        color = (255, 50, 50) if k["class"] == "cavity" else (50, 150, 255)
        d.rectangle([x0, y0, x1, min(H - 1, y0 + H * 0.15)], outline=color, width=3)
        d.text((x0 + 3, max(0, y0 - 12)), f'{k["class"]} {k["confidence"]:.2f} @{k["x_m"]}m', fill=color)
    img.save(path)

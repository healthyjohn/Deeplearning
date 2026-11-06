"""B-scan processing, image export and YOLO label geometry."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .scenes import C0, Scene

IMG_W, IMG_H = 640, 640


def process_bscan(bscan: np.ndarray, dt: float, background: bool = True, gain: float = 1.0) -> np.ndarray:
    """Standard GPR display processing; returns values in [-1, 1].

    1) background removal (mean trace subtracted -> removes direct wave and flat layers)
    2) time-power gain t**gain to boost deep echoes
    3) robust amplitude scaling
    Apply the same function to real KONTUR exports before detection.
    """
    b = np.asarray(bscan, dtype=np.float64)
    if background and b.shape[1] > 1:
        b = b - b.mean(axis=1, keepdims=True)
    t = (np.arange(b.shape[0]) + 1) * dt
    b = b * (t / t[-1])[:, None] ** gain
    scale = np.percentile(np.abs(b), 99.5) or 1.0
    return np.clip(b / scale, -1, 1)


def to_image(processed: np.ndarray, size: tuple[int, int] = (IMG_W, IMG_H)) -> Image.Image:
    """Grey-scale picture: rows = time (down), columns = traces."""
    arr = ((processed + 1) * 127.5).astype(np.uint8)
    img = Image.fromarray(arr, mode="L").resize(size, Image.BILINEAR)
    return img.convert("RGB")


def target_boxes(scene: Scene) -> list[tuple[int, float, float, float, float]]:
    """YOLO boxes (class, xc, yc, w, h normalised) for every target.

    Box covers the hyperbola apex and its arms down to an extra path length k,
    which is where the reflection is still clearly visible.
    """
    xs = scene.trace_positions()
    x_lo, x_hi = xs[0], xs[-1]
    tw = scene.time_window
    shift = math.sqrt(2) / scene.freq
    period = 1.0 / scene.freq
    boxes = []
    for tg in scene.targets:
        v = scene.mean_velocity(tg.depth)
        top = tg.depth - tg.radius
        k = min(max(0.6 * top, 0.12), 0.35)
        half = math.sqrt((top + k) ** 2 - top**2)
        t_apex = 2 * top / v + shift
        t_edge = 2 * (top + k) / v + shift
        y0 = t_apex - 0.6 * period
        y1 = t_edge + 1.0 * period
        if tg.cls == "cavity":
            y1 = max(y1, t_apex + 4 * tg.radius / C0 + 1.5 * period)
        x0, x1 = max(tg.x - half, x_lo), min(tg.x + half, x_hi)
        y0, y1 = max(y0, 0.0), min(y1, tw)
        if x1 <= x0 or y1 <= y0:
            continue
        xc = ((x0 + x1) / 2 - x_lo) / (x_hi - x_lo)
        w = (x1 - x0) / (x_hi - x_lo)
        yc = (y0 + y1) / 2 / tw
        h = (y1 - y0) / tw
        boxes.append((tg.class_id, xc, yc, w, h))
    return boxes


def yolo_label_text(boxes) -> str:
    return "".join(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n" for c, xc, yc, w, h in boxes)


def draw_boxes(img: Image.Image, boxes, names=("cavity", "pipe")) -> Image.Image:
    from PIL import ImageDraw

    img = img.copy()
    d = ImageDraw.Draw(img)
    W, H = img.size
    colors = [(255, 60, 60), (60, 160, 255)]
    for b in boxes:
        c, xc, yc, w, h = b[:5]
        x0, y0 = (xc - w / 2) * W, (yc - h / 2) * H
        x1, y1 = (xc + w / 2) * W, (yc + h / 2) * H
        d.rectangle([x0, y0, x1, y1], outline=colors[int(c) % 2], width=2)
        label = names[int(c)] + (f" {b[5]:.2f}" if len(b) > 5 else "")
        d.text((x0 + 3, max(y0 - 12, 0)), label, fill=colors[int(c) % 2])
    return img

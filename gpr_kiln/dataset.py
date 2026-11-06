"""Generate a YOLO-format dataset of synthetic GPR B-scans."""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

import numpy as np

from .imaging import process_bscan, target_boxes, to_image, yolo_label_text
from .scenes import CLASS_NAMES, Scene, random_scene
from .simulate import simulate


def render_scene(scene: Scene, engine: str, rng: np.random.Generator):
    raw, dt = simulate(scene, engine=engine, rng=rng)
    img = to_image(process_bscan(raw, dt))
    return raw, dt, img, target_boxes(scene)


def generate_dataset(
    out_dir: str,
    n_scenes: int = 20,
    engine: str = "gprmax",
    seed: int = 0,
    val_fraction: float = 0.2,
    save_raw: bool = True,
    scene_overrides: dict | None = None,
    progress: bool = True,
) -> dict:
    """Write images/, labels/, raw/ (npz) and data.yaml under out_dir.

    Re-running with the same out_dir appends new scenes after the existing ones,
    so a long gprMax job can be split into several calls.
    """
    rng = np.random.default_rng(seed)
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)
    if save_raw:
        os.makedirs(os.path.join(out_dir, "raw"), exist_ok=True)

    meta_path = os.path.join(out_dir, "scenes.jsonl")
    start = 0
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            start = sum(1 for _ in f)

    counts = {"train": 0, "val": 0, "objects": 0}
    t0 = time.time()
    for i in range(start, start + n_scenes):
        scene = random_scene(rng, **(scene_overrides or {}))
        raw, dt, img, boxes = render_scene(scene, engine, rng)
        split = "val" if rng.random() < val_fraction else "train"
        name = f"{engine}_{i:05d}"
        img.save(os.path.join(out_dir, "images", split, name + ".png"))
        with open(os.path.join(out_dir, "labels", split, name + ".txt"), "w") as f:
            f.write(yolo_label_text(boxes))
        if save_raw:
            np.savez_compressed(os.path.join(out_dir, "raw", name + ".npz"), bscan=raw, dt=dt)
        with open(meta_path, "a") as f:
            f.write(json.dumps({"name": name, "split": split, "engine": engine, "dt": dt, "scene": asdict(scene)}) + "\n")
        counts[split] += 1
        counts["objects"] += len(boxes)
        if progress:
            print(f"[{i - start + 1}/{n_scenes}] {name} {split} objects={len(boxes)} ({time.time() - t0:.0f}s)", flush=True)

    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: images/val\n")
        f.write("names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES)))
    return {"data_yaml": yaml_path, "generated": counts, "seconds": round(time.time() - t0, 1)}

"""MCP server exposing the GPR pipeline as tools for Kiln (or any MCP client).

Run:  python -m gpr_kiln serve            (stdio, for Kiln "local MCP server")
      python -m gpr_kiln serve --transport streamable-http   (remote/HTTP)

Tools
  gpr_simulate_scene    one random scene -> B-scan image + labels (preview)
  gpr_generate_dataset  many scenes -> YOLO dataset (gprMax or analytic)
  gpr_train_detector    train YOLO on a dataset
  gpr_detect            detect cavities/pipes in a B-scan image or raw file
"""
from __future__ import annotations

import os

try:  # mcp >= 2
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP as _Server

WORK_DIR = os.environ.get("GPR_KILN_WORKDIR", os.path.abspath("gpr_kiln_work"))

server = _Server(
    "gpr-kiln",
    instructions=(
        "Ground penetrating radar (GPR) tools. Use gpr_detect to find cavities and pipes "
        "in a B-scan image. Use gpr_generate_dataset + gpr_train_detector to build a "
        "detector from gprMax simulations. Paths are on the server machine; relative "
        f"paths resolve under {WORK_DIR}."
    ),
)


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(WORK_DIR, path)


@server.tool()
def gpr_simulate_scene(out_dir: str = "preview", seed: int = 0, engine: str = "analytic", freq_mhz: float = 600) -> dict:
    """Simulate one random road cross-section (asphalt/base/subgrade + cavities/pipes).

    engine: "gprmax" (realistic FDTD, ~1 min) or "analytic" (instant approximation).
    Returns the image path, labelled image path, YOLO labels and scene parameters.
    """
    import numpy as np

    from .dataset import render_scene
    from .imaging import draw_boxes
    from .scenes import CLASS_NAMES, random_scene

    out = _abs(out_dir)
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(seed)
    scene = random_scene(rng, empty_prob=0.0, freq=freq_mhz * 1e6)
    _, dt, img, boxes = render_scene(scene, engine, rng)
    stem = os.path.join(out, f"{engine}_seed{seed}")
    img.save(stem + ".png")
    draw_boxes(img, boxes).save(stem + "_labelled.png")
    return {
        "image": stem + ".png",
        "labelled_image": stem + "_labelled.png",
        "labels": [{"class": CLASS_NAMES[c], "xc": round(x, 4), "yc": round(y, 4), "w": round(w, 4), "h": round(h, 4)} for c, x, y, w, h in boxes],
        "targets": [t.__dict__ for t in scene.targets],
        "layers": [l.__dict__ for l in scene.layers],
        "scan_length_m": round(float(scene.trace_positions()[-1] - scene.trace_positions()[0]), 3),
        "time_window_ns": round(scene.time_window * 1e9, 2),
    }


@server.tool()
def gpr_generate_dataset(out_dir: str = "dataset", n_scenes: int = 20, engine: str = "gprmax", seed: int = 0, val_fraction: float = 0.2) -> dict:
    """Generate a YOLO-format training set of synthetic GPR B-scans.

    gprMax takes ~45 s per scene on 4 CPU cores; call repeatedly with the same
    out_dir (and a new seed) to add more scenes. Returns the data.yaml path.
    """
    from .dataset import generate_dataset

    return generate_dataset(_abs(out_dir), n_scenes, engine, seed, val_fraction, progress=False)


@server.tool()
def gpr_train_detector(data_yaml: str, epochs: int = 50, model: str = "yolo11n.pt", imgsz: int = 640, name: str = "train") -> dict:
    """Train a YOLO detector (classes: cavity, pipe). Returns best model path and metrics."""
    from .detector import train

    return train(_abs(data_yaml), model, epochs, imgsz, project=os.path.join(WORK_DIR, "runs"), name=name)


@server.tool()
def gpr_detect(
    input_path: str,
    model_path: str,
    conf: float = 0.25,
    scan_length_m: float | None = None,
    time_window_ns: float | None = None,
    eps: float | None = None,
    dt: float | None = None,
) -> dict:
    """Detect cavities and pipes in a GPR B-scan.

    input_path: PNG/JPG B-scan (time down, distance across) or raw .npy/.npz.
    Give scan_length_m, time_window_ns and eps (ground permittivity, ~6-12) to
    get x position and estimated depth in metres for each detection.
    """
    from .detector import detect

    src = _abs(input_path)
    annotated = os.path.splitext(src)[0] + "_detected.png"
    return detect(_abs(model_path), src, conf, scan_length_m, time_window_ns, eps, dt, annotated)


def main(transport: str = "stdio"):
    os.makedirs(WORK_DIR, exist_ok=True)
    server.run(transport=transport)


if __name__ == "__main__":
    main()

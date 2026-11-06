"""Command line: python -m gpr_kiln <command> ...

  generate  make a synthetic YOLO dataset (gprMax or analytic engine)
  train     train a YOLO detector on a dataset
  detect    run a trained detector on a B-scan image or raw .npy/.npz
  serve     start the MCP server (for Kiln and other MCP clients)
"""
from __future__ import annotations

import argparse
import json


def main(argv=None):
    p = argparse.ArgumentParser(prog="gpr_kiln")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate")
    g.add_argument("out_dir")
    g.add_argument("-n", "--n-scenes", type=int, default=20)
    g.add_argument("--engine", choices=["gprmax", "analytic"], default="gprmax")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--val-fraction", type=float, default=0.2)
    g.add_argument("--freq", type=float, default=600e6, help="antenna centre frequency, Hz")

    t = sub.add_parser("train")
    t.add_argument("data_yaml")
    t.add_argument("--model", default="yolo11n.pt")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--project", default="runs/gpr")
    t.add_argument("--name", default="train")
    t.add_argument("--device", default=None)

    d = sub.add_parser("detect")
    d.add_argument("model")
    d.add_argument("input")
    d.add_argument("--conf", type=float, default=0.25)
    d.add_argument("--scan-length-m", type=float)
    d.add_argument("--time-window-ns", type=float)
    d.add_argument("--eps", type=float, help="ground relative permittivity for depth estimate")
    d.add_argument("--dt", type=float, help="sample interval (s) for raw .npy input")
    d.add_argument("--save", help="write annotated image here")

    s = sub.add_parser("serve")
    s.add_argument("--transport", choices=["stdio", "streamable-http", "sse"], default="stdio")

    a = p.parse_args(argv)
    if a.cmd == "generate":
        from .dataset import generate_dataset

        r = generate_dataset(a.out_dir, a.n_scenes, a.engine, a.seed, a.val_fraction, scene_overrides={"freq": a.freq})
    elif a.cmd == "train":
        from .detector import train

        r = train(a.data_yaml, a.model, a.epochs, a.imgsz, a.batch, a.project, a.name, a.device)
    elif a.cmd == "detect":
        from .detector import detect

        r = detect(a.model, a.input, a.conf, a.scan_length_m, a.time_window_ns, a.eps, a.dt, a.save)
    else:
        from .mcp_server import main as serve

        serve(a.transport)
        return
    print(json.dumps(r, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

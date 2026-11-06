"""Train and run a YOLO detector on GPR B-scan images (Ultralytics)."""
from __future__ import annotations

import os

import numpy as np

from .imaging import draw_boxes, process_bscan, to_image
from .scenes import C0, CLASS_NAMES


def train(
    data_yaml: str,
    model: str = "yolo11n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch: int = 16,
    project: str = "runs/gpr",
    name: str = "train",
    device: str | None = None,
) -> dict:
    """Train a detector. Falls back to training from scratch if the
    pretrained weights cannot be downloaded (offline machines)."""
    from ultralytics import YOLO

    try:
        net = YOLO(model)
    except Exception as e:  # no internet for *.pt weights
        cfg = model.replace(".pt", ".yaml")
        print(f"could not load {model} ({e}); training from scratch with {cfg}")
        net = YOLO(cfg)
    kwargs = dict(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=batch, project=project, name=name, exist_ok=True,
                  fliplr=0.5, flipud=0.0, mosaic=0.5, hsv_h=0.0, hsv_s=0.0, plots=True)
    if device is not None:
        kwargs["device"] = device
    res = net.train(**kwargs)
    save_dir = str(getattr(res, "save_dir", None) or net.trainer.save_dir)
    best = os.path.join(save_dir, "weights", "best.pt")
    metrics = {}
    rd = getattr(res, "results_dict", None) or {}
    for k in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        if k in rd:
            metrics[k.split("/")[1].replace("(B)", "")] = round(float(rd[k]), 4)
    return {"best_model": best, "save_dir": save_dir, "metrics": metrics}


def _load_input(path: str, dt: float | None):
    """Accept a picture (png/jpg) or a raw B-scan (.npy / .npz with 'bscan' and 'dt')."""
    from PIL import Image

    if path.endswith((".npy", ".npz")):
        if path.endswith(".npz"):
            z = np.load(path)
            raw, dt = z["bscan"], float(z["dt"]) if "dt" in z else dt
        else:
            raw = np.load(path)
        if dt is None:
            raise ValueError("dt (sample interval, s) is required for .npy input")
        n_samples = raw.shape[0]
        return to_image(process_bscan(raw, dt)), n_samples * dt
    return Image.open(path).convert("RGB"), None


def detect(
    model_path: str,
    input_path: str,
    conf: float = 0.25,
    scan_length_m: float | None = None,
    time_window_ns: float | None = None,
    eps: float | None = None,
    dt: float | None = None,
    save_annotated: str | None = None,
) -> dict:
    """Detect cavities/pipes. If scan length, time window and permittivity are
    given, each detection also gets x position (m) and approximate depth (m)."""
    from ultralytics import YOLO

    img, tw = _load_input(input_path, dt)
    if tw is not None and time_window_ns is None:
        time_window_ns = tw * 1e9
    net = YOLO(model_path)
    # hyperbola boxes of one object overlap a lot -> stricter, class-agnostic NMS
    res = net.predict(img, conf=conf, iou=0.3, agnostic_nms=True, verbose=False)[0]
    names = res.names if hasattr(res, "names") else dict(enumerate(CLASS_NAMES))
    W, H = img.size
    dets = []
    boxes_for_draw = []
    for b in res.boxes:
        x0, y0, x1, y1 = [float(v) for v in b.xyxy[0]]
        c, p = int(b.cls[0]), float(b.conf[0])
        d = {"class": names[c], "confidence": round(p, 3), "bbox_px": [round(x0), round(y0), round(x1), round(y1)]}
        if scan_length_m:
            d["x_m"] = round((x0 + x1) / 2 / W * scan_length_m, 3)
            d["width_m"] = round((x1 - x0) / W * scan_length_m, 3)
        if time_window_ns:
            t_apex = (y0 / H) * time_window_ns * 1e-9  # hyperbola apex ~ box top
            d["apex_time_ns"] = round(t_apex * 1e9, 2)
            if eps:
                d["depth_m"] = round(t_apex * C0 / np.sqrt(eps) / 2, 3)
        dets.append(d)
        boxes_for_draw.append((c, (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H, p))
    out = {"input": input_path, "n_detections": len(dets), "detections": dets}
    if save_annotated:
        draw_boxes(img, boxes_for_draw, names=[names[i] for i in sorted(names)]).save(save_annotated)
        out["annotated_image"] = save_annotated
    return out

import asyncio
import os

import numpy as np
import pytest

from gpr_kiln.dataset import generate_dataset
from gpr_kiln.imaging import target_boxes
from gpr_kiln.scenes import Scene, random_scene
from gpr_kiln.simulate import gprmax_input


def test_random_scene_targets_inside_scan():
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = random_scene(rng)
        xs = s.trace_positions()
        for t in s.targets:
            assert xs[0] < t.x < xs[-1]
            assert t.depth + t.radius < s.depth
        assert s.time_window > s.two_way_time(max([t.depth for t in s.targets], default=0.1))


def test_boxes_are_normalised():
    rng = np.random.default_rng(1)
    for _ in range(50):
        s = random_scene(rng, empty_prob=0)
        for c, xc, yc, w, h in target_boxes(s):
            assert c in (0, 1)
            for v in (xc, yc, w, h):
                assert 0 <= v <= 1
            assert -1e-9 <= xc - w / 2 and xc + w / 2 <= 1 + 1e-9


def test_gprmax_input_has_all_objects():
    s = random_scene(np.random.default_rng(2), empty_prob=0)
    text = gprmax_input(s)
    assert text.count("#cylinder:") == len(s.targets)
    assert text.count("#box:") == len(s.layers)
    assert "#src_steps:" in text and "#rx_steps:" in text


def test_scene_roundtrip():
    s = random_scene(np.random.default_rng(3))
    import json

    assert Scene.from_dict(json.loads(s.to_json())) == s


def test_generate_analytic_dataset(tmp_path):
    r = generate_dataset(str(tmp_path), n_scenes=6, engine="analytic", seed=0, progress=False)
    assert os.path.exists(r["data_yaml"])
    imgs = list((tmp_path / "images").rglob("*.png"))
    labels = list((tmp_path / "labels").rglob("*.txt"))
    assert len(imgs) == len(labels) == 6
    # appending keeps names unique
    generate_dataset(str(tmp_path), n_scenes=2, engine="analytic", seed=9, progress=False)
    assert len(list((tmp_path / "images").rglob("*.png"))) == 8


def test_mcp_server_lists_tools():
    from gpr_kiln.mcp_server import server

    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert {"gpr_simulate_scene", "gpr_generate_dataset", "gpr_train_detector", "gpr_detect"} <= names


@pytest.mark.skipif(os.environ.get("GPR_SLOW") != "1", reason="set GPR_SLOW=1 to run gprMax")
def test_gprmax_scene_runs():
    from gpr_kiln.simulate import simulate_gprmax

    s = random_scene(np.random.default_rng(4), empty_prob=0, width=0.8, depth=0.5)
    raw, dt = simulate_gprmax(s)
    assert raw.shape[1] == s.n_traces and dt > 0

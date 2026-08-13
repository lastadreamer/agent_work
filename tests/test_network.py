import pytest

torch = pytest.importorskip("torch")

from xiangqi_engine import Board, Encoder, load_config
from xiangqi_engine.config import deepcopy_config, resolve_device
from xiangqi_engine.network import PolicyValueNet, build_network, infer


def _tiny_cfg():
    cfg = deepcopy_config(load_config())
    cfg["network"]["blocks"] = 1
    cfg["network"]["channels"] = 8
    cfg["network"]["policy_head_channels"] = 4
    cfg["network"]["value_head_channels"] = 4
    cfg["network"]["value_hidden"] = 16
    return cfg


def test_forward_shapes_and_value_range():
    cfg = _tiny_cfg()
    net = PolicyValueNet(cfg)
    enc = Encoder(cfg)
    x = torch.from_numpy(enc.tensor(Board())).unsqueeze(0)
    logits, value = net(x)
    assert logits.shape == (1, cfg["action"]["size"])
    assert value.shape == (1,)
    assert -1.0 <= float(value.detach()[0]) <= 1.0


def test_batch_and_masked_policy_sums_to_one():
    cfg = _tiny_cfg()
    net = PolicyValueNet(cfg).eval()
    enc = Encoder(cfg)
    b = Board()
    x = torch.from_numpy(enc.tensor(b)).unsqueeze(0)
    with torch.no_grad():
        logits, _ = net(x)
        legal = enc.legal_action_indices(b)
        probs = net.masked_policy(logits[0], legal)
    assert abs(float(probs.sum()) - 1.0) < 1e-5
    illegal = [i for i in range(cfg["action"]["size"]) if i not in set(legal)]
    assert float(probs[illegal].sum()) == 0.0


def test_infer_helper_and_build_network():
    cfg = _tiny_cfg()
    net = build_network(cfg, device="cpu")
    enc = Encoder(cfg)
    legal, probs, value = infer(net, enc, Board(), device="cpu")
    assert len(legal) == 44
    assert len(probs) == 44
    assert abs(sum(probs) - 1.0) < 1e-5
    assert -1.0 <= value <= 1.0
    assert resolve_device(cfg, "cpu") == "cpu"

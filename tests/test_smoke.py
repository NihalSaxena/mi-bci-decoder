from bci.config import load_config


def test_config_loads():
    cfg = load_config()
    assert cfg["dataset"]["n_channels"] == 22
    assert cfg["epoching"]["tmax_s"] > cfg["epoching"]["tmin_s"]

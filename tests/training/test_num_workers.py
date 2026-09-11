import pytest

from scripts.train import resolve_num_workers


def test_num_workers_precedence(monkeypatch):
    monkeypatch.delenv("XRAY_NUM_WORKERS", raising=False)
    assert resolve_num_workers({}) == 4
    monkeypatch.setenv("XRAY_NUM_WORKERS", "0")
    assert resolve_num_workers({}) == 0
    assert resolve_num_workers({"num_workers": 2}) == 2  # explicit config wins


@pytest.mark.parametrize("value", ["-1", "many"])
def test_invalid_num_workers_fail_fast(monkeypatch, value):
    monkeypatch.setenv("XRAY_NUM_WORKERS", value)
    with pytest.raises(ValueError, match="num_workers"):
        resolve_num_workers({})

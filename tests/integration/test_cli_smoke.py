"""End-to-end CLI checks on synthetic data (not project results)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import run_all
from src.inference.smoke import build_synthetic_workspace
from src.training.config import load_config, save_resolved_config

ROOT = Path(__file__).resolve().parents[2]


def run_cli(*args, env=None):
    clean = {k: v for k, v in os.environ.items() if not k.startswith("XRAY_")}
    clean["XRAY_SKIP_DOTENV"] = "1"
    clean.update(env or {})
    return subprocess.run([sys.executable, "run_all.py", *args], cwd=ROOT, env=clean,
                          capture_output=True, text=True)


def test_help_lists_every_subcommand():
    result = run_cli("--help")
    assert result.returncode == 0, result.stderr
    for command in ("check", "train", "develop", "evaluate", "infer", "benchmark", "verify-release"):
        assert command in result.stdout


def test_check_and_legacy_alias_pass_without_data():
    assert run_cli("check").returncode == 0
    legacy = run_cli("--check")
    assert legacy.returncode == 0 and "check passed" in legacy.stdout


def test_dotenv_is_loaded_only_when_not_skipped(tmp_path, monkeypatch):
    from src import pipeline
    env_file = tmp_path / ".env"
    env_file.write_text("XRAY_OUTPUT_ROOT=/should/not/leak\n")
    monkeypatch.delenv("XRAY_OUTPUT_ROOT", raising=False)
    assert pipeline.load_env_file(env_file) is False
    assert "XRAY_OUTPUT_ROOT" not in os.environ
    monkeypatch.setenv("XRAY_SKIP_DOTENV", "0")
    assert pipeline.load_env_file(env_file) is True
    assert os.environ["XRAY_OUTPUT_ROOT"] == "/should/not/leak"
    monkeypatch.delenv("XRAY_OUTPUT_ROOT")


def test_invalid_inputs_fail_fast_with_exit_codes(tmp_path):
    assert run_cli("check", "--config", "configs/missing.yaml").returncode == 1
    assert run_cli("--check", "--config", "configs/missing.yaml").returncode == 2
    missing_env = run_cli("train", "--config", "configs/baseline.yaml")
    assert missing_env.returncode == 2 and "XRAY_OUTPUT_ROOT" in missing_env.stderr
    no_freeze = run_cli("infer", "--checkpoint", str(tmp_path / "x.pt"), "--split", "test",
                        env={"XRAY_DATA_ROOT": str(tmp_path)})
    assert no_freeze.returncode == 2 and "threshold" in no_freeze.stderr.lower()
    assert run_cli("evaluate", "--help").returncode == 0


def write_config(source, name, workspace, path, **model):
    cfg = load_config(ROOT / source)
    cfg["data"]["manifests"] = {k: str(workspace[k]) for k in ("train", "validation", "test")}
    cfg["data"]["loader"] = {"num_workers": 0}
    cfg["training"].update(epochs=1, batch_size=2, amp=False)
    cfg["augmentation"] = {}
    cfg["model"].update(name=name, **model)
    save_resolved_config(cfg, path)
    return path


@pytest.fixture
def synthetic_env(tmp_path, monkeypatch):
    workspace = build_synthetic_workspace(tmp_path / "data", n_per_class=3)
    monkeypatch.setenv("XRAY_DATA_ROOT", str(workspace["data_root"]))
    monkeypatch.setenv("XRAY_OUTPUT_ROOT", str(tmp_path / "outputs"))
    monkeypatch.setenv("XRAY_NUM_WORKERS", "0")
    configs = [
        write_config("configs/baseline.yaml", "small_cnn", workspace, tmp_path / "baseline.yaml"),
        write_config("configs/densenet121.yaml", "densenet121", workspace, tmp_path / "densenet.yaml",
                     pretrained=False, freeze_strategy="head_only"),
    ]
    return workspace, configs, tmp_path


def test_develop_then_single_locked_test_run(synthetic_env, capsys):
    workspace, configs, tmp = synthetic_env
    outputs, report = tmp / "outputs", tmp / "report"
    develop = ["develop", "--config", str(configs[0]), "--config", str(configs[1]),
               "--run-ids", "cnn_s42", "dn_s42", "--device", "cpu", "--out-dir", str(report)]
    assert run_all.main(develop) == 0
    for run in ("cnn_s42", "dn_s42"):
        assert (outputs / run / "checkpoints/best.pt").is_file()
        assert (outputs / run / "predictions_val.csv").is_file()
    frozen_path = report / "tables/frozen_threshold.json"
    frozen = json.loads(frozen_path.read_text())
    assert frozen["selected_on"] == "validation" and frozen["run_id"] in {"cnn_s42", "dn_s42"}
    run_dir = outputs / frozen["run_id"]
    checkpoint = run_dir / "checkpoints/best.pt"

    # Re-running develop reuses finished runs instead of retraining them.
    stamp = (run_dir / "metrics.json").stat().st_mtime_ns
    assert run_all.main(develop) == 0
    assert (run_dir / "metrics.json").stat().st_mtime_ns == stamp

    infer = ["infer", "--checkpoint", str(checkpoint), "--split", "test", "--device", "cpu",
             "--threshold-file", str(frozen_path), "--manifest", str(workspace["test"])]
    assert run_all.main(infer) == 0
    predictions = run_dir / "predictions_test.csv"
    frame = pd.read_csv(predictions)
    assert len(frame) == len(pd.read_csv(workspace["test"]))
    assert set(frame.split) == {"test"} and set(frame.threshold) == {frozen["threshold"]}
    assert run_all.main(infer) == 2  # the locked test is predicted once

    assert run_all.main(["evaluate", "test", "--predictions", f"{frozen['recommended_model']}={predictions}",
                         "--manifest", str(workspace["test"]), "--threshold-file", str(frozen_path),
                         "--out-dir", str(report)]) == 0
    metrics = pd.read_csv(report / "tables/test_metrics.csv")
    assert metrics.loc[0, "run_id"] == frozen["run_id"] and metrics.loc[0, "split"] == "test"

    capsys.readouterr()
    image = next(workspace["data_root"].rglob("*.png"))
    assert run_all.main(["infer", "--checkpoint", str(checkpoint), "--image", str(image),
                         "--threshold-file", str(frozen_path), "--device", "cpu"]) == 0
    record = json.loads(capsys.readouterr().out)[0]
    assert record["predicted_class"] in {"NORMAL", "PNEUMONIA"}
    assert record["predicted_label"] == int(record["probability"] >= frozen["threshold"])

    bench = tmp / "bench.json"
    assert run_all.main(["benchmark", "--checkpoint", str(checkpoint), "--device", "cpu",
                         "--repeats", "2", "--output", str(bench)]) == 0
    assert json.loads(bench.read_text())["run_id"] == frozen["run_id"]


def test_train_resumes_interrupted_run_under_develop(synthetic_env):
    from src import pipeline
    workspace, configs, tmp = synthetic_env
    run_dir = pipeline.train(configs[0], run_id="resume_me", device="cpu")
    (run_dir / "metrics.json").unlink()  # simulate a run killed before it finished
    assert pipeline.run_state(run_dir) == "resumable"
    assert pipeline.train(configs[0], run_id="resume_me", device="cpu", reuse_existing=True) == run_dir
    assert pipeline.run_state(run_dir) == "complete"
    (tmp / "outputs" / "broken").mkdir()
    with pytest.raises(pipeline.PipelineError, match="without a checkpoint"):
        pipeline.train(configs[0], run_id="broken", device="cpu", reuse_existing=True)


def test_develop_rejects_duplicate_model_names(synthetic_env):
    from src import pipeline
    _, configs, _ = synthetic_env
    with pytest.raises(pipeline.PipelineError, match="distinct model names"):
        pipeline.develop([configs[0], configs[0]], device="cpu")

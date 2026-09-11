import json

import numpy as np
import pandas as pd
import pytest

from scripts import release_experiments
from src.inference import predict
from src.inference.smoke import build_smoke_checkpoint, build_synthetic_workspace


@pytest.fixture
def frozen_report(tmp_path, monkeypatch):
    workspace = build_synthetic_workspace(tmp_path / "data", n_per_class=4)
    monkeypatch.setenv("XRAY_DATA_ROOT", str(workspace["data_root"]))
    output = tmp_path / "outputs"
    runs = {"small_cnn": "cnn_run", "densenet121": "dn_run"}
    for model, run in runs.items():
        loaded = predict.load_model(build_smoke_checkpoint(workspace, output, model_name=model, run_id=run),
                                    device="cpu")
        frame = predict.predict_manifest(loaded, workspace["validation"], "validation")
        predict.write_predictions(frame, output / run / "predictions_val.csv")
    tables = tmp_path / "report" / "tables"
    tables.mkdir(parents=True)
    pd.DataFrame({"model": list(runs), "run_id": list(runs.values())}).to_csv(tables / "validation_metrics.csv", index=False)
    frozen = {"recommended_model": "densenet121", "run_id": "dn_run", "threshold": 0.5, "selected_on": "validation",
              "per_model_selection": {"small_cnn": {"threshold": 0.5}, "densenet121": {"threshold": 0.5}}}
    (tables / "frozen_threshold.json").write_text(json.dumps(frozen))
    return tmp_path / "report", output


def test_release_experiments_write_aggregate_tables(frozen_report):
    report, output = frozen_report
    assert release_experiments.main(["--report-dir", str(report), "--output-root", str(output),
                                     "--device", "cpu", "--bootstrap", "200", "--repeats", "2"]) == 0
    tables = report / "tables"

    efficiency = pd.read_csv(tables / "efficiency.csv")
    assert set(efficiency.model) == {"small_cnn", "densenet121"} and set(efficiency.device) == {"cpu"}
    assert (efficiency.latency_ms_median > 0).all()
    assert efficiency.set_index("model").parameters["densenet121"] > efficiency.set_index("model").parameters["small_cnn"]

    ci = pd.read_csv(tables / "validation_bootstrap_ci.csv")
    assert set(ci.comparison) == {"small_cnn", "densenet121", "densenet121 - small_cnn"}
    assert ((ci.ci95_low <= ci.value + 1e-9) & (ci.value <= ci.ci95_high + 1e-9)).all()
    assert (ci.n_images == 8).all()

    same = json.loads((tables / "inference_consistency.json").read_text())
    assert same["model"] == "densenet121" and same["n_images"] == 8
    assert same["max_abs_probability_difference"] < 1e-5 and same["decision_agreement"] == 1.0


def test_bootstrap_matches_exact_metrics_and_is_paired(tmp_path):
    output = tmp_path / "outputs"
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 200)
    for run, noise in (("a", 0.2), ("b", 0.45)):
        probability = np.clip(labels + rng.normal(0, noise, 200), 0, 1)
        (output / run).mkdir(parents=True)
        pd.DataFrame({"image_path": [f"x/{i}.png" for i in range(200)], "label": labels,
                      "probability": probability}).to_csv(output / run / "predictions_val.csv", index=False)
    table = release_experiments.bootstrap({"densenet121": "a", "small_cnn": "b"},
                                          {"densenet121": 0.5, "small_cnn": 0.5}, output, n_boot=300)
    from sklearn.metrics import f1_score
    frame = pd.read_csv(output / "a" / "predictions_val.csv")
    exact = f1_score(frame.label, (frame.probability >= 0.5).astype(int), average="macro")
    row = table[(table.comparison == "densenet121") & (table.metric == "macro_f1")].iloc[0]
    assert row.value == pytest.approx(exact)
    diff = table[(table.comparison == "densenet121 - small_cnn") & (table.metric == "macro_f1")].iloc[0]
    assert diff.value > 0 and diff.share_of_resamples_above_zero > 0.9


def test_selection_must_come_from_validation(frozen_report):
    report, _ = frozen_report
    path = report / "tables" / "frozen_threshold.json"
    frozen = json.loads(path.read_text())
    path.write_text(json.dumps(dict(frozen, selected_on="test")))
    with pytest.raises(ValueError, match="validation"):
        release_experiments.load_selection(report)

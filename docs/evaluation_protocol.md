# Evaluation protocol

Owner: Member 4. This document fixes how every reported number in the paper is
produced. Changing anything here after the split freeze requires a decision-log
entry and a pull request, not a silent edit.

## Label and probability contract

| Item | Value |
|---|---|
| Negative class | `NORMAL` = 0 |
| Positive class | `PNEUMONIA` = 1 |
| Prediction level | Image |
| Model output | One pneumonia logit; sigmoid applied before export |
| Decision rule | `predicted_pneumonia = probability >= threshold` |

The `>=` convention is fixed project-wide so a threshold of 0.0 predicts
pneumonia everywhere and 1.0 predicts it almost nowhere. Anything that reports
a metric must go through `src/evaluation/`, so the baseline and DenseNet121 are
never scored by two slightly different code paths.

## Prediction file contract

Each run exports one CSV per evaluated split (`predictions_val.csv`,
`predictions_test.csv`) with at least:

| Column | Meaning |
|---|---|
| `image_path` | Path relative to `XRAY_DATA_ROOT`; joins to the split manifest |
| `label` | Ground truth, `0` or `1` — numeric, not `NORMAL`/`PNEUMONIA` |
| `probability` | Pneumonia probability in `[0, 1]`, **after** sigmoid |
| `model` | Factory model name: `small_cnn` or `densenet121` |
| `run_id` | Identity of the run that produced the checkpoint |
| `split` | `validation` or `test`; required from the shared trainer |

Both modes require a canonical `--manifest`. Every prediction must match its
complete image set and numeric labels, joined by full relative `image_path`.
Duplicate, missing and extra images are rejected. Manifest split values must
match the mode; prediction split values are checked whenever present. Folder
names in image paths describe the original archive, not the rebuilt split.
Each CSV must identify exactly one model and run. Comparison uses one shared
validation manifest; run IDs can differ between models and are preserved.
Subtype and other audit metadata are read from the canonical manifest.

`load_predictions` rejects, with a named error: missing columns, empty files,
duplicated `image_path`, missing values, text labels, probabilities outside
`[0, 1]` (the symptom of exporting raw logits), and columns containing only
0.0/1.0 (the symptom of exporting hard predictions).

## Metrics

| Metric | Definition | Role |
|---|---|---|
| Macro F1 | Unweighted mean of per-class F1 | **Primary** — model selection |
| Pneumonia sensitivity | `TP / (TP + FN)` | **Clinical** — missed pneumonia |
| Specificity | `TN / (TN + FP)` | False-alarm rate |
| ROC-AUC | Threshold-free ranking quality | Secondary |
| PR-AUC | Threshold-free, imbalance-aware | Secondary |
| Confusion matrix | `TN, FP, FN, TP` | Reported for every headline result |

Macro rather than weighted averaging is deliberate: pneumonia is ~72% of this
dataset, so a weighted average would flatter a model that rarely predicts
NORMAL. Accuracy is recorded but is never used to choose anything.

**Zero-division convention.** When a denominator is zero the metric is reported
as `0.0`, matching `sklearn`'s `zero_division=0`. Class supports (`n_normal`,
`n_pneumonia`) always accompany the metrics so a degenerate value is visible as
degenerate rather than read as a bad score.

## Threshold selection

The operating threshold is selected **on validation predictions only**.

- **Candidates:** every distinct predicted probability, plus 0.0 and 1.0. Under
  the `>=` rule each is the exact point where one more image flips to
  pneumonia, which makes the set both minimal and exhaustive — a fixed
  0.01 grid can step straight over the optimum when probabilities cluster.
- **Objective:** macro F1, the same quantity the models are compared on.
- **Tie-break, applied in order:** higher pneumonia sensitivity (a missed
  pneumonia is the costlier error), then the lower threshold.

Selection is deterministic and invariant to row order, both of which are
covered by tests — the frozen threshold must not depend on DataLoader ordering.

## Test-set discipline

The locked test set is evaluated **once**, after the team freezes the model and
the threshold.

`scripts/evaluate_model.py` has two asymmetric modes:

- `validation` searches for a threshold and records the selected model and run
  in `frozen_threshold.json`, alongside the validation manifest SHA-256.
- `test` requires `--threshold-file` and the test manifest. It checks
  `selected_on=validation`, the model and run identity, and a finite threshold
  in `[0, 1]`. Only the frozen model/run can be scored. A bare numeric
  `--threshold` is not accepted.

These checks prevent accidental split or run substitution; they do not
cryptographically authenticate user-edited CSVs or enforce a one-time lock.
The team remains responsible for performing the final test only once and
recording that event. The shared training command exports validation only;
test prediction/inference integration remains release work.

## Commands

```bash
# 1. Validation: compare models, select and freeze the threshold
python scripts/evaluate_model.py validation \
    --predictions small_cnn=$XRAY_OUTPUT_ROOT/<run>/predictions_val.csv \
    --predictions densenet121=$XRAY_OUTPUT_ROOT/<run>/predictions_val.csv \
    --manifest data/manifests/validation.csv \
    --out-dir report

# 2. Locked test: exactly once, at the frozen threshold
python scripts/evaluate_model.py test \
    --predictions densenet121=$XRAY_OUTPUT_ROOT/<run>/predictions_test.csv \
    --manifest data/manifests/test.csv \
    --threshold-file report/tables/frozen_threshold.json \
    --out-dir report
```

## Outputs

| Path | Contents |
|---|---|
| `report/tables/validation_metrics.csv` | Model comparison, sorted by macro F1 |
| `report/tables/frozen_threshold.json` | Recommended model + frozen threshold — the hand-off to Member 5 |
| `report/tables/error_by_subtype.csv` | FN/FP rates for normal / bacterial / viral |
| `report/tables/error_by_confidence.csv` | Outcome counts per probability bin |
| `report/tables/test_metrics.csv` | Locked-test result (written once) |
| `report/figures/roc_validation.png` | ROC, both models on shared axes |
| `report/figures/pr_validation.png` | PR curves with the no-skill prevalence line |
| `report/figures/confusion_matrix_<model>_<split>.png` | Counts and row percentages |
| `report/figures/threshold_sweep_<model>.png` | Macro F1 / sensitivity / specificity vs threshold, with the selected point marked |

## Grad-CAM

`src/evaluation/gradcam.py` weights the last convolutional feature maps by the
gradient of a target score. The actual DenseNet wrapper resolves to
`backbone.features`. Both `head_only` and `last_block` are supported: input
gradients enable attribution without unfreezing or updating backbone weights.
Call `model.eval()` before using Grad-CAM. Two project-specific points:

- **Single-logit sign convention.** Both models emit one pneumonia logit, not a
  two-class softmax, so explaining a NORMAL prediction requires
  backpropagating the *negated* logit. `target="predicted"` handles this;
  getting it wrong yields a heatmap that explains the opposite decision.
- **Runs on CPU.** One forward plus one backward pass per image — measured at
  ~33 ms per 112x112 image on CPU. A dozen report examples cost seconds, so
  **no GPU slot is needed**; only the trained checkpoint file is required.

Correctness is verified functionally, not just structurally: a small CNN is
trained on a synthetic task whose discriminative region is known, and the test
asserts Grad-CAM concentrates its mass there (measured ~63% versus ~25% for
random attention), and that the `normal` target does not highlight the same
region. See `tests/evaluation/test_gradcam.py`.

**Privacy exception.** Unlike every other output in `src/evaluation/`, Grad-CAM
figures contain the X-ray itself. `data/README.md` and `CONTRIBUTING.md` forbid
committing raw or processed images, so the default output location is
`XRAY_OUTPUT_ROOT`, outside Git. Publishing a small number of examples in the
report PDF is a separate, deliberate decision that belongs in
`docs/decisions.md`.

## Error analysis and privacy

`error_analysis.py` returns **aggregates only**. It emits no filename, no
patient id, and no image. Subgroups smaller than `MIN_GROUP_SIZE = 10` are
collapsed into an `other` bucket so no row can be traced to an individual.
"Most confident false negative" is reported as a probability, never as a path.

Pneumonia subtype (bacterial / viral) is read from the manifest, with a fallback to the filename pattern
`personNNN_{bacteria,virus}_MMM`. It is **audit metadata for subgroup error
analysis only** — never a training target, never an input to model selection.
This provides the per-group performance breakdown the Track 2 brief requires.

## Known limitations to state in the paper

1. **Patient identities are inferred from filenames.** The audit extracts
   keys for both NORMAL and PNEUMONIA filenames and combines them with exact
   hashes and reviewed near-duplicate links. These are grouping heuristics,
   not independently verified clinical patient identifiers; undetected
   identity overlap remains a limitation.
2. **Grad-CAM is not clinical evidence.** It shows where the network's
   activations are, not where the pathology is, and must be presented as a
   qualitative sanity check only.
3. **Single validation split, single seed.** Reported differences carry no
   confidence interval; a difference of a few tenths of a macro-F1 point should
   not be read as a real ranking.
4. **Pediatric, single-centre data** (Guangzhou, ages 1–5). Nothing here
   transfers to adults or other sites without external validation.

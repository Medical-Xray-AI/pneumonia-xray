# Model card: pediatric pneumonia classifier

Owner: Member 5. Headline numbers are filled from `report/tables/` only after
the validation freeze and the single locked-test run; until then they are
`pending`. Never copy numbers from synthetic smoke tests into this card.

## Intended use

- **Task:** image-level binary classification of a pediatric chest X-ray as
  `NORMAL` (0) or `PNEUMONIA` (1).
- **Users:** course project, research and teaching.
- **Not intended for:** clinical diagnosis, triage, treatment decisions, adults,
  lateral views, or any site other than the source hospital without external
  validation.

## Data

| Item | Value |
|---|---|
| Source | Kermany et al. (2018), Kaggle `paultimothymooney/chest-xray-pneumonia` v2 |
| Population | Guangzhou Women and Children's Medical Center, ages ~1-5 |
| Split | `split_v1`, seed 42: 4,102 train / 877 validation / 877 test images |
| Split unit | Filename patient key, exact hash and near-duplicate groups kept within one split |
| Class balance | Pneumonia is ~72% of images; class weight fitted on train only |

## Models

| Model | Architecture | Input | Normalization |
|---|---|---|---|
| Baseline | `small_cnn` (custom CNN, ~0.4M parameters) | 224x224, aspect-preserving pad, gray -> 3 channels | Train-split statistics |
| Main | `densenet121`, ImageNet-pretrained, one-logit head | same | ImageNet |

Both emit one pneumonia logit; probability = sigmoid(logit). The decision rule
is `probability >= threshold`, where the threshold is selected on validation
only (macro F1, ties -> higher sensitivity, then lower threshold).

## Metrics

| Split | Model | Threshold | Macro F1 | Sensitivity | Specificity | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| Validation | small_cnn | pending | pending | pending | pending | pending | pending |
| Validation | densenet121 | pending | pending | pending | pending | pending | pending |
| Locked test (once) | frozen model | frozen | pending | pending | pending | pending | pending |

Sources: `report/tables/validation_metrics.csv`, `report/tables/test_metrics.csv`,
subgroup errors in `report/tables/error_by_subtype.csv`.

Efficiency (from `python run_all.py benchmark --checkpoint <best.pt>`):
parameters, weights size and median latency per image are recorded here for the
frozen model on the GPU server and on CPU: pending.

## Limitations and risks

1. Patient identity is inferred from filenames; undetected identity overlap
   between splits cannot be ruled out.
2. Single centre, single seed, single validation split: small metric
   differences are not a reliable ranking and carry no confidence interval.
3. Pediatric-only data; performance on adults or other scanners is unknown.
4. Label noise in the public dataset has been reported by others; labels were
   not re-read by radiologists in this project.
5. Bacterial/viral subtype is used only for error analysis. Subgroups smaller
   than 10 images are pooled, so rare-group behaviour is not characterised.
6. Grad-CAM shows where activations concentrate, not where pathology is; it is a
   qualitative sanity check only.

## Bias considerations

The training population is one age band from one hospital. Differences in
equipment, positioning, age or disease prevalence can shift both the
probability calibration and the validation-selected threshold. Any reuse must
re-select the threshold on local validation data and re-evaluate sensitivity.

## Reproduction

```bash
python run_all.py check --require-data
python run_all.py develop --run-ids baseline_s42 densenet121_s42 --update-registry
```

The locked test is run once, after sign-off, with the commands in
[`release_checklist.md`](release_checklist.md).

## Disclaimer

Educational research output. Not a medical device and not validated for clinical use.

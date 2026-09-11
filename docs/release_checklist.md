# Release checklist

Owner: Member 5, final sign-off: team lead. Tick each item in the release pull
request with a link to its evidence (command output, commit or file).

## 1. Environment (GPU server)

- [ ] `python scripts/audit_environment.py` output recorded (Python, driver, CUDA, torch, torchvision).
- [ ] `python -m pytest -q` passes on the server.
- [ ] `python scripts/audit_environment.py --write-lock requirements-lock.txt` committed.
- [ ] DenseNet121 ImageNet weights cached on the server before the booked run.

## 2. Split freeze

- [ ] `python scripts/verify_manifests.py --check-files --check-dataset` prints READY.
- [ ] Manifests are unchanged `split_v1` (seed 42); no personal re-splits.

## 3. Development runs

- [ ] `python run_all.py check --require-data` passes.
- [ ] `python run_all.py develop --run-ids baseline_s42 densenet121_s42 --update-registry`
      completed; interrupted runs were resumed with the same command.
- [ ] Every run is in `docs/experiment_registry.csv` with commit SHA, config, device,
      peak VRAM, best epoch and a relative checkpoint path.
- [ ] Checkpoints and logs stay under `XRAY_OUTPUT_ROOT`, outside Git.

## 4. Model and threshold freeze

- [ ] `report/tables/validation_metrics.csv` and `frozen_threshold.json` reviewed by the team.
- [ ] Team lead approves the frozen model, run ID and threshold (record in `docs/decisions.md`).

## 5. Locked test - exactly once

```bash
python run_all.py infer --checkpoint "$XRAY_OUTPUT_ROOT/<run_id>/checkpoints/best.pt" \
    --split test --threshold-file report/tables/frozen_threshold.json
python run_all.py evaluate test \
    --predictions <model>="$XRAY_OUTPUT_ROOT/<run_id>/predictions_test.csv" \
    --manifest data/manifests/test.csv \
    --threshold-file report/tables/frozen_threshold.json --out-dir report
```

- [ ] Run on the release branch; date, operator and commit SHA recorded.
- [ ] `test_evaluated` set to `True` for that run in the registry.
- [ ] No model, threshold or preprocessing change after this point.

## 6. Efficiency and demo

- [ ] `python run_all.py benchmark --checkpoint <best.pt>` on GPU and CPU; numbers in `docs/model_card.md`.
- [ ] Single-image demo: `python run_all.py infer --checkpoint <best.pt> --threshold-file report/tables/frozen_threshold.json --image <file>`.

## 7. Repository QA

- [ ] `python scripts/verify_release.py --with-data --clean-clone` prints RELEASE READY.
- [ ] No `.env`, credentials, VPN files, tokens, raw X-rays, checkpoints or absolute paths tracked.
- [ ] Grad-CAM figures published only after the privacy decision in `docs/decisions.md`.
- [ ] `docs/model_card.md` metrics filled from `report/tables/`.
- [ ] README commands match the working CLI.
- [ ] Release tagged (for example `v1.0`) after review.

# Data module boundary

The leader-owned infrastructure currently provides:

- `download.py` - version-pinned Kaggle download;
- exact source-folder and class-count validation;
- canonical dataset-root discovery;
- safe local `.env` configuration without committing credentials or images.

Member 1 will add the reviewed implementations for:

- integrity and duplicate auditing;
- patient/group-aware splitting;
- portable manifests;
- PyTorch Dataset/DataLoader helpers;
- audit visualizations and data sanity tests.

Those modules must arrive through Member 1's feature branch and pull request.
The shared contract remains documented in `configs/data.yaml` and
`data/manifests/README.md`.

# Data audit summary

- Dataset root: supplied at runtime through `XRAY_DATA_ROOT` (not stored in reports)
- Images scanned: **5856**
- Corrupt or unreadable images: **0**
- Images with a parsed patient identifier: **5856**
- Exact duplicate groups: **30**
- Conservative near-duplicate pairs: **44**
- Leakage groups touching both development data and provided test: **5**
- Provided test is safe to lock: **false**

## Class distribution

| Source split | Label | Count |
|---|---:|---:|
| provided_test | 0 | 234 |
| provided_test | 1 | 390 |
| provided_train | 0 | 1341 |
| provided_train | 1 | 3875 |
| provided_val | 0 | 8 |
| provided_val | 1 | 8 |

## Identity and duplicate policy

Pneumonia identifiers include the bacterial/viral namespace, for example `pneumonia:bacterial:person1` and `pneumonia:viral:person1`. Normal identifiers retain the `IM` versus `NORMAL2-IM` namespace. Unparseable names keep an empty patient ID and are controlled by exact/perceptual duplicate groups.

Near duplicates are conservative candidates that pass both pHash and dHash Hamming-distance thresholds (<= 4).

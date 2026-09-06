# Dataset source and scope

## Approved data

- Title: **Large Dataset of Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images**
- Chest X-ray subset: pediatric anterior-posterior radiographs, approximately ages 1-5
- Current canonical release: **Mendeley Data Version 3**
- DOI: <https://doi.org/10.17632/rscbjbr9sj.3>
- Team download mirror: <https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>
- License listed by Mendeley Data: **CC BY 4.0**

The project uses binary image-level labels:

```text
0 = NORMAL
1 = PNEUMONIA
```

The bacterial/viral filename annotation is retained as audit metadata and is not a separate binary training target.

## Expected Kaggle layout

```text
chest_xray/
|-- train/
|   |-- NORMAL/
|   `-- PNEUMONIA/
|-- val/
|   |-- NORMAL/
|   `-- PNEUMONIA/
`-- test/
    |-- NORMAL/
    `-- PNEUMONIA/
```

The commonly distributed copy contains 5,856 images. Counts and integrity must still be verified from the actual downloaded archive; they are not hard-coded as proof of a successful audit.

## Split and identity cautions

The Mendeley page describes the training and testing sets as independent patients. The audit nevertheless verifies recoverable identifiers, exact hashes, and conservative perceptual duplicate candidates before locking the provided test.

Pneumonia filename counters are interpreted inside bacterial/viral namespaces. NORMAL identifiers are parsed only from recognized `IM-...` and `NORMAL2-IM-...` filename structures. Unknown formats retain an empty patient ID and use duplicate-cluster fallback grouping.

Raw images and machine-specific absolute paths must never be committed to Git. Set the location with `XRAY_DATA_ROOT`.

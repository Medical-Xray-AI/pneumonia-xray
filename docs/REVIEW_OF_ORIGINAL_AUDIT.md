# Read-only correction of the submitted audit evidence

The following checks were recomputed from the submitted `file_manifest.csv` without modifying it and without claiming that the real images were rescanned:

| Check | Corrected result |
|---|---:|
| Manifest rows | 5,856 |
| NORMAL images with parsed namespace-aware ID | 1,583 / 1,583 |
| PNEUMONIA images with parsed subtype-aware ID | 4,273 / 4,273 |
| Unique NORMAL IDs | 1,444 |
| Unique PNEUMONIA IDs | 2,653 |
| Parsed patient IDs crossing provided source splits | 0 |
| Submitted SHA-256 exact-duplicate groups | 30 |
| Submitted exact hashes crossing provided source splits | 0 |

The rejected report listed 1,674 unique pneumonia patient IDs and 170 train/test overlaps because it reduced both `personN_bacteria_...` and `personN_virus_...` to the same numeric key `N`. There are 979 numeric values reused by the two subtype namespaces, so that key is not reliable.

The submitted manifest contains hashes but not perceptual hashes. Consequently, the provided test is only **provisionally clean** from the corrected patient-ID and exact-hash evidence. Run the corrected full audit against the real 5,856 image files before freezing the split. If its `test_overlap_report.csv` is empty, retain the provided test; otherwise use the configured complete group-level rebuild.

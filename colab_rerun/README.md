# Colab rerun with extended metrics

This folder holds a full rerun of the Group 6 PatchCore study on Google Colab, plus an extra analysis with stricter and more detailed localization metrics. The original code, data split and settings are unchanged (repo commit `e68b117`). The new code is `extended_metrics.py`, which reads the finished runs and computes the extra metrics.

## What is in this folder

| Path | Contents |
| --- | --- |
| `group6_patchcore_colab.ipynb` | Colab notebook (outputs cleared, notes in Chinese): mount Drive, clone this repo, install packages, download data and weights, run the audit and all 81 runs, rebuild the original summaries, run the extended analysis, compare with the Windows results |
| `extended_metrics.py` | New analysis script (extra metrics, size bins, bootstrap, claim table). It only reads each run's saved `predictions.npz` and `samples.json`, so no model is retrained |
| `tests/test_extended_metrics.py` | 13 unit tests for `extended_metrics.py` |
| `results/summary/mean_std.csv` | Same table format as `results/summary_windows/mean_std.csv`, from the Colab runs |
| `results/extended/mean_std_extended.csv` | All new metrics, mean ± std over seeds 0, 1, 2 |
| `results/extended/claims.csv` | For each metric and subgroup, how many of 9 cases support each original claim |
| `results/extended/robustness_layers.csv`, `robustness_size.csv` | Per-case layer and input-size comparisons behind `claims.csv` |
| `results/extended/bootstrap_layer2_vs_fusion.csv` | Paired bootstrap (2,000 repeats, every seed) for pointing@8px and region recall |
| `results/extended/region_recall_by_size.csv` | Region recall at 1% normal-pixel FPR, by defect size (<64, 64–255, 256–1023, ≥1024 px) |
| `results/extended/extended_summary.json` | Settings used by `extended_metrics.py` |
| `results/comparisons/` | Seed-0 paired AUPRO@0.3 bootstrap, layer2 vs fusion |
| `results/figures/` | Region recall by defect size (224 px), and heatmap comparisons |

Raw run folders (81 runs, about 7.8 GB), logs, data and weights are not included. They are on Google Drive.

## How to reproduce

**Full rerun on Colab.** Open `group6_patchcore_colab.ipynb` in Colab, choose a GPU runtime, and run the cells in order. Results are saved to `MyDrive/group6_patchcore/`, so the matrix can resume after a disconnect. The notebook writes its own copy of `extended_metrics.py` into the cloned repo, so it does not depend on this folder.

**Extended metrics only.** If you already have finished run folders, run these from the repo root. `extended_metrics.py` imports `study/`, so the repo root must be on `PYTHONPATH`:

```bash
PYTHONPATH=. python colab_rerun/extended_metrics.py --root <runs_dir> --output <out_dir>
PYTHONPATH=.:colab_rerun python -m unittest discover -s colab_rerun/tests -v
```

`extended_metrics.py` sits outside `study/`, so it does not change `study_source_sha256`.

## Reproduction check

The Colab results match the Windows results in `results/summary_windows/`. Over all 81 rows, the largest difference is 2.2e-5 for AUPRO@0.3 and 2.1e-4 for image AUROC. The dataset and study-source hashes are identical. Latency is not comparable: it was measured on a different GPU, and Colab timings varied a lot between runs.

## New metrics

- AUPRO with stricter FPR limits (0.05 and 0.01), next to the original AUPRO@0.3
- Pixel AP, image AP, and image TPR at 5% FPR
- Pointing accuracy: whether the highest-scoring pixel falls on the defect (0 px and 8 px tolerance)
- Region recall at a threshold set on normal test images only (1% normal-pixel FPR), also split by defect size
- Oracle pixel F1 (its threshold is picked on test masks, so it is a diagnostic only)

## Main findings

1. **layer2 > layer3 holds clearly.** It holds in 9/9 cases for AUPRO at all three FPR limits.
2. **320 px > 224 px, and the gain grows at strict FPR.** For layer2, AUPRO@0.3 improves by 0.010–0.037. AUPRO@0.01 improves by 0.05 (metal_nut), 0.11 (screw) and 0.20 (capsule).
3. **layer2 ≥ fusion on small defects holds, but gaps are small.** On small defects it holds in 8–9/9 cases for AUPRO and pixel AP. On large defects it holds in only 1/9 cases (AUPRO@0.3). Fusion does better on metal_nut, which is mostly large defects. 43 of 45 bootstrap intervals for pointing and region recall include zero.
4. **Very small defects are the weak spot.** For regions under 64 px, region recall averages 0.78, compared with ≥ 0.95 for larger regions. In this size bin fusion ≥ layer2 in all 9 settings, but each category has only 11–15 such regions.

## Caveats

- Screw has 1 large-defect test image and metal_nut has 7 small-defect images, so those subgroups are only descriptive.
- Image TPR at 5% FPR varies a lot across seeds (std up to 0.09), because each category has only 22–41 normal test images.
- No threshold or hyperparameter was tuned on test masks, except the oracle F1 diagnostic.

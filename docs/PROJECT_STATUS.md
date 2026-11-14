# Final project status on Windows

Paths in this status report are relative to the project root.

The experiments and final report are complete. An editable English Word counterpart and 16-slide English and Traditional Chinese presentations are also available. The slide-only presentation video is pending. All formal runs used Python 3.12, CUDA PyTorch 2.14.1+cu130, and an NVIDIA GeForce RTX 5060 Laptop GPU. The packages installed for those runs are recorded in `requirements-windows-lock.txt`; report generation additionally uses `tools/requirements-report.txt`.

## Completed experiments

The main study has 54 runs: three MVTec AD categories, three feature settings, three coreset seeds, and two input sizes, with a 10% approximate greedy coreset. Another 27 runs compare the same categories, features, and seeds at 224 pixels with 4,096 vectors per bank (16 MiB of float32 vector storage).

Each of the 81 run folders contains `complete.json`, `config.json`, `metrics.json`, `runtime.json`, saved predictions, a FAISS model, and representative images. The records identify CUDA and the RTX 5060 for every formal run. Within each category, the main runs have matching data and study-source hashes. Nine scientific unit tests and small CUDA pipeline checks at both input sizes passed.

The data audit, three-seed aggregate tables, six paired image-bootstrap comparisons of layer2 and fusion, and qualitative examples are complete. The final report is seven pages and includes nine cited research papers.

## Teammate Colab rerun and extended analysis

The merged contribution in `colab_rerun/` is supplementary to the Windows-based final PDF and slides. Its notebook describes a full 81-run Colab rerun; the repository contains cleared notebook cells, aggregate tables, extended metrics, comparisons, figures, code, and tests. Full Colab run folders are not in GitHub. See `docs/組員新增資料索引.md` before using its results in a report.

The Colab and Windows summary CSVs contain the same 81 condition rows and matching dataset and study-source hashes. The largest AUPRO@0.3 mean difference is about 2.24e-5. The original 9 scientific tests and the new 13 extended-metric tests pass locally. Colab and Windows latency values describe different hardware and must remain separate. The existing final PDF and presentations have not been rewritten to include the extended metrics.

## Findings

Layer2 has higher small-defect AUPRO@0.3 than layer3 in all three categories at both input sizes. Fusion scores are close to layer2 but do not consistently improve on it. With equal 16 MiB vector banks, layer2 scores 0.9593 versus 0.9505 for fusion on screw and 0.9733 versus 0.9719 on capsule. Fusion scores 0.9953 versus 0.9951 on metal_nut, a difference of 0.0002 in a subgroup with seven anomaly images. Layer2 is faster than fusion in all three matched-bank categories.

Moving from 224 to 320 pixels improves small-defect AUPRO in all nine category-feature comparisons and increases bank storage and inference time. Layer2 gains 0.0139 on screw, 0.0130 on capsule, and 0.0011 on metal_nut; its mean latency rises by about three times. Resize and smoothing scales also change, so they contribute to the input-size comparison.

## Files and interpretation

- Final report: `output/pdf/group6_patchcore_final_report.pdf`; editable Word version: `output/word/group6_patchcore_final_report.docx`.
- Slides: `output/presentation/group6_patchcore_final_presentation.pptx` and `output/presentation/group6_patchcore_final_presentation_zh.pptx`.
- Full Windows results: `results/runs_windows/formal/` on the original workstation only; the public repository contains compact summaries and figures.
- Aggregate tables and plots: `results/summary_windows/`.
- Paired comparisons: `results/comparisons/`.
- Data audit: `results/audit/`.
- Reproduction instructions: `README.md` and `docs/實作與執行指南.md`.
- Recording and submission guide: `docs/錄影報告指南.md`. The presentation video is pending.
- Historical Mac baseline: `results/legacy_mac/` on the original workstation only. Its timings are kept separate from the Windows results.

The screw larger-defect subgroup has one anomaly image, and the metal_nut small-defect subgroup has seven. Center cropping removes some annotated pixels, though every defect mask retains pixels. Bootstrap intervals describe test-image sampling for one fitted model pair; three-seed standard deviations describe coreset variation. No threshold or hyperparameter was tuned on test masks.

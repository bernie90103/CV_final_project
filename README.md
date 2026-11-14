# Group 6 — PatchCore small defect localization

組員快速入口：[英文期末報告 PDF](output/pdf/group6_patchcore_final_report.pdf) · [可編輯英文 Word](output/word/group6_patchcore_final_report.docx) · [Windows 與 Colab 中文比較報告](output/pdf/windows_colab_comparison_zh.pdf) · [比較報告 Word](output/word/windows_colab_comparison_zh.docx) · [英文簡報](output/presentation/group6_patchcore_final_presentation.pptx) · [中文對照簡報](output/presentation/group6_patchcore_final_presentation_zh.pptx) · [新增資料與報告索引](docs/組員新增資料索引.md) · [中文專案與重現指南](docs/實作與執行指南.md) · [中文錄影報告指南](docs/錄影報告指南.md) · [結果摘要](results/summary_windows/mean_std.csv)。正式報告維持全英文；其餘操作說明使用中文並保留必要的英文術語。

這個公開儲存庫收錄程式、七頁英文報告、精簡實驗結果與圖表。原始 MVTec AD 圖片、預訓練權重及約 7.8 GB 的逐次執行輸出留在本機，未上傳 GitHub。想從頭執行的組員需下載資料與權重；只想看成果可直接開啟報告與 `results/summary_windows/`。

2026 年 11 月隊員另外加入了 [Colab 重跑與延伸指標](colab_rerun/README.md)。它保留在獨立的 `colab_rerun/`，作為原 Windows 實驗的補充驗證；新增的嚴格誤報率指標、區域召回率與 bootstrap 表不在目前的正式 PDF 和簡報中。兩邊的準確度可以對照，延遲不能跨硬體直接比較。具體數字與限制整理在[中文比較報告](docs/Windows與Colab實驗比較報告.md)，各檔案用途見[新增資料與報告索引](docs/組員新增資料索引.md)。

This Windows project compares frozen ImageNet WideResNet50 layer2, layer3, and fusion features on MVTec AD screw, capsule, and metal_nut. The protocol follows `docs/group6_proposal.pdf` and includes 224 and 320 pixel inputs. Start with `docs/實作與執行指南.md` for the Traditional Chinese project explanation and reproduction steps. `docs/錄影報告指南.md` covers the presentation recording and submission.

The root contains the runnable Python and Windows command files. `study/` holds the study implementation, `tests/` holds scientific checks, and `vendor/` points to the pinned PatchCore source as a Git submodule. Local copies of images and weights live in `data/` and `.cache/torch/`; they are excluded from GitHub. The repository includes compact results in `results/summary_windows/`, `results/audit/`, `results/comparisons/`, and `results/figures/`. The English report is in `output/pdf/`, with an editable Word counterpart in `output/word/`. English and Traditional Chinese 16-slide presentations are in `output/presentation/`. Project documents are in `docs/`, and the report builders recreate `tmp/` as scratch space.

To obtain the pinned PatchCore source when cloning, use `git clone --recurse-submodules https://github.com/bernie90103/CV_final_project.git`. For an existing clone, run `git submodule update --init --recursive`.

## Windows setup (RTX 5060)

From PowerShell in this directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_data.py
.\.venv\Scripts\python.exe prepare_weights.py
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The CUDA check must report `True` and identify the RTX 5060. Use the explicit `.venv\Scripts\python.exe` path below so the commands work without changing PowerShell's execution policy. The pinned official PatchCore source is in `vendor/patchcore-inspection`; keep its LICENSE and NOTICE.

`requirements-windows-lock.txt` records the exact packages installed for the first RTX 5060 run. The CUDA wheels come from the PyTorch `cu130` index above.

## Data, protocol, and experiments

On the original workstation, extracted images are in `data/mvtec/` and cached ImageNet V1 WideResNet50-2 weights are in `.cache/torch/`. A fresh clone must run `download_data.py` and `prepare_weights.py` to obtain them. Download manifests record locally calculated archive hashes, not an official checksum comparison.

Small defects occupy at most 1% of the **original** image area according to the original mask; normal test images are included in each subgroup comparison. Images use bilinear resize and center crop, masks use nearest-neighbor resize. The 224 configuration uses resize 256; the 320 configuration uses resize 366. Both resolutions retain roughly the same field of view, but Gaussian smoothing has a different effective scale relative to the original image.

```powershell
.\.venv\Scripts\python.exe inspect_dataset.py --size 224
.\.venv\Scripts\python.exe inspect_dataset.py --size 320
.\.venv\Scripts\python.exe run_experiment.py --category screw --layers fusion --seed 0 --device cuda --output results/runs_windows --resume
.\.venv\Scripts\python.exe run_matrix.py --sizes 224 --device cuda --output results/runs_windows
.\.venv\Scripts\python.exe run_matrix.py --sizes 320 --device cuda --output results/runs_windows
.\.venv\Scripts\python.exe summarize_results.py --root results/runs_windows --output results/summary_windows
```

The main matrix contains 3 categories × 3 feature settings × 3 seeds × 2 sizes = 54 full-data runs. Runs execute sequentially and completed matching configurations are skipped. If an incomplete run exists, move it aside before retrying. The optional fixed-bank experiment uses `run_matrix.py --sizes 224 --bank-vectors 4096 --device cuda --output results/runs_windows`; it is a separate condition.

`run_windows_matrix.cmd` runs both sizes and writes a progress log at `results/windows_matrix.log`; after successful completion it refreshes `results/summary_windows/`.

The runner uses only normal training images. It stores the exact CPU FAISS index, continuous scores and anomaly maps, metrics, representative heatmaps, timings, configuration, package versions, hashes, and a `complete.json` marker. Feature extraction and coreset sampling use CUDA; FAISS search remains on CPU. AUPRO@0.3 is the primary metric; pixel and image AUROC are secondary. Latency is a synchronized batch-1 median over 30 timed iterations after 10 warmups, including device transfer, feature extraction, CPU search, upsampling, and smoothing, but excluding image loading and preprocessing.

The prior Mac baseline is retained in `results/legacy_mac/`. Its timing is specific to that Mac and must not be pooled with Windows timings. Debug and validation runs remain in `results/verification/` and are excluded from formal summaries.

After a complete Windows run, inspect `results/runs_windows/formal/.../metrics.json` and `runtime.json`. `evaluate.py`, `compare_runs.py`, `plot_comparison.py`, and `predict.py` accept completed run paths. Example prediction:

```powershell
.\.venv\Scripts\python.exe predict.py --run results/runs_windows/formal/screw/size224/fusion/ratio0.1/seed0 --image data/mvtec/screw/test/thread_side/000.png --device cuda
```

The output score is not a calibrated defect probability; no decision threshold is learned from test masks. Screw has only one large-defect test image, and metal_nut has seven small-defect images, so those subgroup results are descriptive. See `THIRD_PARTY.md` for source and license attribution and `docs/PROJECT_STATUS.md` for current progress.

The completed seven-page English PDF is at `output/pdf/group6_patchcore_final_report.pdf`; an editable English Word version with the same content is at `output/word/group6_patchcore_final_report.docx`. Both are generated from the Windows result tables and include the data audit, performance and efficiency comparisons, matched-bank experiment, bootstrap analysis, and qualitative examples. To rebuild them, install `tools/requirements-report.txt`, then run `tools/build_final_report.py` and `tools/build_final_report_docx.py`. When `docs/presentation_video_url.txt` contains a valid HTTPS sharing link, both builders add that link.

The slide-only presentation is available in English and Traditional Chinese in `output/presentation/`. Both decks have 16 corresponding slides, including data tables, figures, qualitative examples, and short source-code excerpts. To rebuild them, install `tools/requirements-presentation.txt`, run `tools/build_final_report.py` to refresh the chart images, then run `tools/build_presentation.py` and `tools/build_presentation_zh.py`. The Chinese slides are a study and recording aid; the formal written report remains English.

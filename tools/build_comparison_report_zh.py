"""Build the supplementary Traditional Chinese Windows/Colab comparison report."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
WIN = ROOT / "results/summary_windows/mean_std.csv"
COLAB = ROOT / "colab_rerun/results/summary/mean_std.csv"
CLAIMS = ROOT / "colab_rerun/results/extended/claims.csv"
REGIONS = ROOT / "colab_rerun/results/extended/region_recall_by_size.csv"
BOOT = ROOT / "colab_rerun/results/extended/bootstrap_layer2_vs_fusion.csv"
MD = ROOT / "docs/Windows與Colab實驗比較報告.md"
DOCX = ROOT / "output/word/windows_colab_comparison_zh.docx"
CHART = ROOT / "output/figures/windows_colab_aupro_parity_zh.png"
KEYS = ("category", "image_size", "layers", "budget", "group")


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def key(row):
    return tuple(row[k] for k in KEYS)


def cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def cell_borders(cell):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "single")
        el.set(qn("w:color"), "D9D9D9")
        el.set(qn("w:sz"), "4")
        borders.append(el)
    tc_pr.append(borders)


def set_east_asia(run, font="Microsoft JhengHei"):
    run.font.name = font
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.rFonts
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.insert(0, rfonts)
    rfonts.set(qn("w:eastAsia"), font)


def add_table(doc, headers, data, widths=None):
    tab = doc.add_table(rows=1, cols=len(headers))
    tab.autofit = False
    for j, h in enumerate(headers):
        tab.rows[0].cells[j].text = str(h)
    for values in data:
        row = tab.add_row()
        for j, value in enumerate(values):
            row.cells[j].text = str(value)
    for i, row in enumerate(tab.rows):
        for j, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if widths:
                cell.width = Cm(widths[j])
            cell_borders(cell)
            if i == 0:
                cell_shading(cell, "E9EEF3")
            elif i % 2 == 0:
                cell_shading(cell, "F8FAFC")
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                for run in p.runs:
                    set_east_asia(run)
                    run.font.size = Pt(8.3)
                    if i == 0:
                        run.bold = True
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return tab


def write_paragraph(doc, text):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(8)
    for run in p.runs:
        set_east_asia(run)
    return p


def section(doc, title, paragraphs=(), table=None):
    doc.add_heading(title, level=1)
    for para in paragraphs:
        write_paragraph(doc, para)
    if table:
        add_table(doc, *table)


def md_table(headers, data):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in data)
    return "\n".join(lines)


def main():
    windows = rows(WIN)
    colab = rows(COLAB)
    cidx = {key(r): r for r in colab}
    assert len(windows) == len(colab) == len(cidx) == 81
    assert {key(r) for r in windows} == set(cidx)
    assert all(r["dataset_sha256"] == cidx[key(r)]["dataset_sha256"] for r in windows)
    assert all(r["study_source_sha256"] == cidx[key(r)]["study_source_sha256"] for r in windows)
    assert {r["seeds"] for r in windows + colab} == {"0,1,2"}
    differences = []
    for metric, label in (("aupro_mean", "AUPRO@0.3"), ("pixel_auroc_mean", "Pixel AUROC"), ("image_auroc_mean", "Image AUROC")):
        diffs = [abs(float(r[metric]) - float(cidx[key(r)][metric])) for r in windows]
        differences.append((label, f"{max(diffs):.8f}", f"{sum(diffs)/len(diffs):.8f}"))

    small = []
    for cat in ("screw", "capsule", "metal_nut"):
        for size in ("224", "320"):
            w2 = next(r for r in windows if r["category"] == cat and r["image_size"] == size and r["layers"] == "layer2" and r["budget"] == "ratio0.1" and r["group"] == "small")
            wf = next(r for r in windows if r["category"] == cat and r["image_size"] == size and r["layers"] == "fusion" and r["budget"] == "ratio0.1" and r["group"] == "small")
            small.append((cat, size, f"{float(w2['aupro_mean']):.4f}", f"{float(cidx[key(w2)]['aupro_mean']):.4f}", f"{float(wf['aupro_mean']):.4f}", f"{float(cidx[key(wf)]['aupro_mean']):.4f}"))

    claims = {(r["metric"], r["group"], r["claim"]): f"{r['n_holds']}/{r['n_cases']}" for r in rows(CLAIMS)}
    claim_table = []
    for metric, label in (("aupro@0.3", "AUPRO@0.3"), ("aupro@0.01", "AUPRO@0.01"), ("pixel_ap", "Pixel AP"), ("region_recall@nfpr0.01", "區域召回率@1% FPR")):
        claim_table.append((label, claims[metric, "small", "layer2 >= fusion"], claims[metric, "large", "layer2 >= fusion"], claims[metric, "small", "layer2 > layer3"], claims[metric, "small", "320 > 224"]))

    regions = rows(REGIONS)
    region_table = []
    for bin_name, label in (("<64", "小於 64"), ("64-255", "64–255"), ("256-1023", "256–1023"), (">=1024", "至少 1024")):
        matched = [r for r in regions if r["size_bin"] == bin_name]
        avg = sum(float(r["recall_mean"]) for r in matched) / len(matched)
        region_table.append((label, str(len(matched)), f"{avg:.3f}"))

    import ast
    bootstrap = [r for r in rows(BOOT) if r["status"] == "ok"]
    point_overlap = sum((lambda x: x[0] <= 0 <= x[1])(ast.literal_eval(r["pointing@8px_ci95"])) for r in bootstrap)
    region_overlap = sum((lambda x: x[0] <= 0 <= x[1])(ast.literal_eval(r["region_recall_ci95"])) for r in bootstrap)
    assert len(bootstrap) == 45 and point_overlap == 43 and region_overlap == 45

    CHART.parent.mkdir(parents=True, exist_ok=True)
    x = [float(r["aupro_mean"]) for r in windows]
    y = [float(cidx[key(r)]["aupro_mean"]) for r in windows]
    fig, ax = plt.subplots(figsize=(6.7, 4.0), dpi=180)
    ax.scatter(x, y, s=21, alpha=0.78, color="#1A6B7A", edgecolors="white", linewidths=0.25)
    low = min(x + y) - .015
    ax.plot([low, 1], [low, 1], color="#A45446", lw=1.4, ls="--")
    ax.set_xlim(low, 1.005)
    ax.set_ylim(low, 1.005)
    ax.set_xlabel("Windows AUPRO@0.3")
    ax.set_ylabel("Colab AUPRO@0.3")
    ax.grid(alpha=.18)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(CHART, bbox_inches="tight")
    plt.close(fig)

    title = "Windows 與 Colab 實驗比較報告"
    intro = "這份補充報告比較原本在 Windows 與 RTX 5060 上完成的 PatchCore 實驗，以及組員在 Google Colab 的重跑和延伸分析。兩份摘要在 81 個相同條件下幾乎重合；Colab 的主要貢獻，是讓我們看見低誤報要求與不同瑕疵大小下，原本結論的適用範圍。正式交件的英文報告另存，這份中文文件供組內討論與修改。"
    overview = [
        ("研究問題", "以正常影像建立 PatchCore 記憶庫，找出 screw、capsule、metal_nut 影像中的異常位置。"),
        ("原實驗", "Windows 11、RTX 5060；完成 81 個彙整條件，使用 seeds 0、1、2。"),
        ("組員實驗", "Google Colab 的 CUDA 環境；依相同資料、程式與設定重跑 81 個條件，再從已存預測結果計算新指標。"),
        ("共同條件", "224／320 像素，layer2／layer3／fusion，10% 記憶庫與固定 4096 patch 記憶庫；依 all、small、large 分組。"),
        ("可比較的結果", "AUPRO@0.3、Pixel AUROC、Image AUROC 與記憶庫大小。兩邊每列的資料雜湊與 study 程式雜湊相同。"),
        ("不可直接比較", "推論延遲。Colab 使用另一套硬體與執行環境，計時不能拿來判定 RTX 5060 與 Colab GPU 的速度高低。"),
    ]
    sections = [
        ("先看結論", ["Windows 與 Colab 的 AUPRO@0.3 平均值，在 81 個逐列配對的條件中，最大絕對差為 0.00002237。這支持 Colab 重跑與原實驗結果一致；它不能單獨證明整個訓練與執行過程完全相同，因為 GitHub 沒有收錄 Colab 的 81 份原始執行資料。", "在小瑕疵的 AUPRO 評估裡，layer2 優於 layer3 的九個條件都成立。layer2 與 fusion 差距較小；換成較大瑕疵時，優勢常轉向 fusion。小於 64 像素的瑕疵區域，召回率也明顯低於較大區域。"], None),
        ("實驗設計與可比性", ["兩份摘要都以同一組條件鍵（類別、輸入尺寸、特徵層、記憶庫預算、瑕疵分組）記錄結果；三個隨機種子也相同。這使準確度能逐列核對。資料集雜湊是按類別不同而有三個值，但每個配對條件的雜湊一致；study 程式雜湊在全部條件一致。"], (["項目", "比較內容"], overview, [3.2, 13.4])),
        ("原指標重跑結果", ["下表的「最大差」與「平均差」都是 81 列配對後的絕對差。這些差距遠小於通常用來討論模型差異的幅度，對原報告的主要準確度趨勢沒有實質影響。"], (["指標", "最大絕對差", "平均絕對差"], differences, [5.6, 5.4, 5.6])),
        ("小瑕疵的直接對照", ["下表固定使用 10% 記憶庫，只列原報告最關心的 small 組。數字是三個種子的 AUPRO@0.3 平均值；四捨五入到小數四位後，Windows 與 Colab 完全一致。layer2 與 fusion 的差別要看類別和尺寸，不能只用一個總平均概括。"], (["類別", "尺寸", "Windows L2", "Colab L2", "Windows 融合", "Colab 融合"], small, [3.0, 1.6, 3.1, 3.1, 3.0, 2.8])),
        ("新增指標改變了哪些解讀", ["組員從已保存的預測分數計算更嚴格的 AUPRO、Pixel AP、pointing accuracy 與區域召回率；沒有重新訓練模型。比較層數時，分母 9 是三種類別乘上三種尺寸／記憶庫設定（224／10%、320／10%、224／4096）；比較解析度時，分母 9 是三種類別乘上三種特徵層。「成立」只是平均值的方向，並不等於每個比較都達統計顯著。"], (["指標", "小瑕疵 L2≥融合", "大瑕疵 L2≥融合", "小瑕疵 L2>L3", "小瑕疵 320>224"], claim_table, [4.4, 3.0, 3.0, 3.0, 3.0])),
        ("很小的瑕疵仍難定位", ["在正常像素誤報率設為 1% 的門檻下，將區域按面積分組後，小於 64 像素的設定平均召回率是 0.782；64 像素以上各組約為 0.950 或更高。表內平均值是各設定的簡單平均，不是把所有區域合併後計算；同一批影像會在多個設定中重複出現，因此設定數也不是獨立樣本數。"], (["瑕疵區域面積（像素）", "設定數", "平均區域召回率"], region_table, [7.2, 3.0, 6.4])),
        ("不確定性與限制", [f"組員以每個 seed 的影像做 2,000 次成對 bootstrap。45 個可計算的小組中，pointing@8px 的 95% 信賴區間有 {point_overlap} 個包含 0；區域召回率則有 {region_overlap} 個包含 0。這表示 layer2 與 fusion 在這兩項指標上的差距，多數情況仍不足以作穩健的勝負判斷。", "metal_nut 的小瑕疵只有 7 張異常圖，screw 的大瑕疵只有 1 張。這些子組的方向值得觀察，但不宜延伸成普遍結論。Colab 的 oracle pixel F1 使用測試 mask 選門檻，只能當診斷值；其餘新門檻沒有用測試 mask 調整。", "原始 Colab 執行資料、log、資料集與模型權重存放於組員的 Google Drive，未進入 GitHub；本報告核對的是儲存庫中的摘要表、分析表與程式。Colab notebook 也清除了執行輸出。"], None),
        ("組內可以怎麼使用", ["保留目前英文正式報告作為原實驗版本。若要把 Colab 分析納入交件版，建議加入一段重現檢查，並用一張表呈現嚴格 FPR 和瑕疵大小結果；同時同步更新英文 Word、PDF 與錄影簡報。報告中要把 AUPRO@0.3 的原始主結果和 Colab 新算的指標分開標示。"], None),
    ]

    md = [f"# {title}", "", f"更新日期：{date.today().isoformat()}　｜　用途：組內討論，非正式英文交件檔", "", intro, ""]
    for heading, paras, table in sections:
        md.extend([f"## {heading}", ""])
        for p in paras:
            md.extend([p, ""])
        if table:
            md.extend([md_table(table[0], table[1]), ""])
        if heading == "原指標重跑結果":
            md.extend(["![81 組 AUPRO 逐列對照](../output/figures/windows_colab_aupro_parity_zh.png)", "", "圖 1　每一點代表一個相同條件；虛線是兩邊完全相等的位置。", ""])
    md.extend(["## 資料來源", "", "- Windows 原結果：`results/summary_windows/mean_std.csv`", "- Colab 重跑摘要：`colab_rerun/results/summary/mean_std.csv`", "- 新指標、比較與 bootstrap：`colab_rerun/results/extended/`", "- Colab 說明：`colab_rerun/README.md`；原實驗方法：`output/pdf/group6_patchcore_final_report.pdf`", ""])
    MD.write_text("\n".join(md), encoding="utf-8")

    DOCX.parent.mkdir(parents=True, exist_ok=True)
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Cm(2.1)
    sec.bottom_margin = Cm(2.0)
    sec.left_margin = Cm(2.2)
    sec.right_margin = Cm(2.2)
    normal = doc.styles["Normal"]
    normal.font.name = "Microsoft JhengHei"
    normal.font.size = Pt(10)
    normal.font.color.rgb = RGBColor(28, 36, 44)
    normal.paragraph_format.line_spacing = 1.22
    for style_name, size in (("Title", 20), ("Heading 1", 13)):
        style = doc.styles[style_name]
        style.font.name = "Microsoft JhengHei"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.paragraph_format.space_before = Pt(13 if style_name == "Heading 1" else 0)
        style.paragraph_format.space_after = Pt(6)
    title_p = doc.add_paragraph(style="Title")
    title_run = title_p.add_run(title)
    set_east_asia(title_run)
    title_ppr = title_p._p.get_or_add_pPr()
    title_border = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "nil")
    title_border.append(bottom)
    title_ppr.append(title_border)
    meta = doc.add_paragraph(f"更新日期：{date.today().isoformat()}    用途：組內討論，非正式英文交件檔")
    meta.paragraph_format.space_after = Pt(12)
    for r in meta.runs:
        set_east_asia(r)
        r.font.size = Pt(8.8)
        r.font.color.rgb = RGBColor(85, 95, 105)
    write_paragraph(doc, intro)
    for heading, paras, table in sections:
        if heading in ("小瑕疵的直接對照", "很小的瑕疵仍難定位"):
            doc.add_page_break()
        section(doc, heading, paras, table)
        if heading == "原指標重跑結果":
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.add_run().add_picture(str(CHART), width=Cm(12.5))
            caption = doc.add_paragraph("圖 1　81 組 AUPRO 逐列對照。虛線代表兩邊完全相等。")
            caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for r in caption.runs:
                set_east_asia(r)
                r.font.size = Pt(8.5)
    section(doc, "資料來源", ["Windows 原結果：results/summary_windows/mean_std.csv；Colab 重跑摘要：colab_rerun/results/summary/mean_std.csv。新指標、比較與 bootstrap：colab_rerun/results/extended/。方法說明：colab_rerun/README.md 與英文正式報告。"])
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.text = "Group 6 PatchCore｜Windows 與 Colab 比較｜補充文件"
    for r in footer.runs:
        set_east_asia(r)
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(100, 110, 120)
    doc.save(DOCX)
    print(f"Wrote {MD}\nWrote {DOCX}\nWrote {CHART}")


if __name__ == "__main__":
    main()

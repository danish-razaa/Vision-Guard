"""Generate comprehensive, publication-grade confusion matrix images for all models in VisionGuard."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "confusion_matrices"

def plot_confusion_matrix(
    cm,
    model_name: str,
    subtitle: str,
    metrics: dict,
    out_path: Path,
    cmap_name: str = "Blues",
    threshold: float | None = None,
):
    """Plot an individual high-resolution confusion matrix with full diagnostic annotations."""
    cm = np.array(cm, dtype=int)
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = cm[1, 0], cm[1, 1]
    total = np.sum(cm)

    # Derived metrics if not provided
    accuracy = metrics.get("accuracy", (tp + tn) / total if total > 0 else 0)
    precision = metrics.get("precision", tp / (tp + fp) if (tp + fp) > 0 else 0)
    recall = metrics.get("recall", tp / (tp + fn) if (tp + fn) > 0 else 0)
    f1 = metrics.get("f1", 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    fig, ax = plt.subplots(figsize=(6.5, 6.2), dpi=300)
    cmap = plt.get_cmap(cmap_name)

    # Normalized matrix for background coloring
    norm_cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    cax = ax.imshow(norm_cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=1)

    # Colorbar
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Class Normalized Ratio", rotation=-90, va="bottom", fontsize=9, fontweight="medium")
    cbar.ax.tick_params(labelsize=8)

    # Cell coordinates and contents
    cell_info = [
        (0, 0, tn, "True Negative (TN)", f"{tn}/{tn+fp} ({tn/(tn+fp)*100:.1f}%)" if (tn+fp)>0 else "0%"),
        (0, 1, fp, "False Positive (FP)", f"{fp}/{tn+fp} ({fp/(tn+fp)*100:.1f}%)" if (tn+fp)>0 else "0%"),
        (1, 0, fn, "False Negative (FN)", f"{fn}/{fn+tp} ({fn/(fn+tp)*100:.1f}%)" if (fn+tp)>0 else "0%"),
        (1, 1, tp, "True Positive (TP)", f"{tp}/{fn+tp} ({tp/(fn+tp)*100:.1f}%)" if (fn+tp)>0 else "0%"),
    ]

    for row, col, count, tag, pct in cell_info:
        bg_val = norm_cm[row, col]
        text_color = "white" if bg_val > 0.55 else "#1e293b"
        sub_color = "#e2e8f0" if bg_val > 0.55 else "#475569"

        ax.text(
            col, row - 0.14,
            f"{count:,}",
            ha="center", va="center",
            color=text_color, fontsize=17, fontweight="bold"
        )
        ax.text(
            col, row + 0.05,
            tag,
            ha="center", va="center",
            color=text_color, fontsize=8.5, fontweight="semibold"
        )
        ax.text(
            col, row + 0.22,
            pct,
            ha="center", va="center",
            color=sub_color, fontsize=8, style="italic"
        )

    # Ticks & Labels
    classes = ["Clean (0)", "Manipulated (1)"]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=10, fontweight="semibold")
    ax.set_yticklabels(classes, fontsize=10, fontweight="semibold")
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold", labelpad=8)

    # Grid styling
    ax.set_xticks(np.arange(len(classes) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(classes) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#cbd5e1", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(which="major", length=0)

    # Title & Subtitle cleanly placed using axis coordinates to prevent any collision
    ax.set_title(
        f"{model_name} — Confusion Matrix",
        fontsize=13, fontweight="bold", color="#0f172a", pad=26
    )
    ax.text(
        0.5, 1.025,
        subtitle,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=8.5, color="#64748b"
    )

    # Performance Stats Banner
    thresh_str = f" | Threshold: {threshold:.3f}" if threshold is not None else ""
    summary_text = (
        f"Accuracy: {accuracy*100:.1f}%   |   Precision: {precision*100:.1f}%   |   "
        f"Recall (Sensitivity): {recall*100:.1f}%\n"
        f"Specificity: {specificity*100:.1f}%   |   F1-Score: {f1:.3f}{thresh_str}"
    )

    ax.text(
        0.5, -0.22,
        summary_text,
        transform=ax.transAxes,
        ha="center", va="center",
        fontsize=8.5,
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="#f8fafc",
            edgecolor="#cbd5e1",
            linewidth=1.0,
        ),
        linespacing=1.4
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_grid(models_data: list[dict], title: str, subtitle: str, out_path: Path):
    """Plot multiple confusion matrices in a clean comparative grid."""
    n = len(models_data)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.4, rows * 4.2), dpi=300)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = np.array([axes])
    elif cols == 1:
        axes = np.array([[ax] for ax in axes])

    classes = ["Clean", "Manipulated"]
    tick_marks = [0, 1]

    for idx, data in enumerate(models_data):
        r, c = idx // cols, idx % cols
        ax = axes[r, c]

        cm = np.array(data["cm"], dtype=int)
        tn, fp = cm[0, 0], cm[0, 1]
        fn, tp = cm[1, 0], cm[1, 1]
        total = np.sum(cm)
        norm_cm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

        ax.imshow(norm_cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                bg = norm_cm[i, j]
                color = "white" if bg > 0.55 else "#0f172a"
                sub_color = "#e2e8f0" if bg > 0.55 else "#475569"
                row_sum = cm[i].sum()
                pct_str = f"({val/row_sum*100:.1f}%)" if row_sum > 0 else ""

                ax.text(j, i - 0.1, f"{val:,}", ha="center", va="center", color=color, fontsize=12, fontweight="bold")
                ax.text(j, i + 0.18, pct_str, ha="center", va="center", color=sub_color, fontsize=7.5)

        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        if c == 0:
            ax.set_ylabel("True Label", fontsize=8.5, fontweight="semibold")
        if r == rows - 1 or idx + cols >= n:
            ax.set_xlabel("Predicted Label", fontsize=8.5, fontweight="semibold")

        acc = data["metrics"].get("accuracy", (tp + tn) / total if total > 0 else 0)
        f1 = data["metrics"].get("f1", 0)
        ax.set_title(
            f"{data['name']}\nAcc: {acc*100:.1f}% | F1: {f1:.3f}",
            fontsize=9.5, fontweight="bold", pad=6
        )

        ax.set_xticks([-0.5, 0.5, 1.5], minor=True)
        ax.set_yticks([-0.5, 0.5, 1.5], minor=True)
        ax.grid(which="minor", color="#cbd5e1", linestyle="-", linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(which="major", length=0)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        r, c = idx // cols, idx % cols
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98, color="#0f172a")
    fig.text(0.5, 0.94, subtitle, ha="center", fontsize=9.5, color="#64748b")
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Baseline Smoke-Test Models Metrics
    with open(PROJECT_ROOT / "models" / "model_metrics.json") as f:
        baseline_metrics = json.load(f)

    # 2. Load Expanded Development Candidates Metrics
    with open(PROJECT_ROOT / "models" / "candidate_all_model_results.json") as f:
        candidate_metrics = json.load(f)

    # 3. Load Pair Detector Metrics
    with open(PROJECT_ROOT / "models" / "pair_detector_metadata.json") as f:
        pair_metadata = json.load(f)

    baseline_grid_data = []
    candidate_grid_data = []
    all_grid_data = []

    # ==========================================
    # Process Baseline Models (Smoke-Test, 45 features)
    # ==========================================
    print("Generating Baseline Models Confusion Matrices...")
    for model_name, data in baseline_metrics.items():
        test_data = data["test"]
        cm = test_data["confusion_matrix"]
        thresh = data.get("threshold", 0.5)

        subfolder_path = OUTPUT_DIR / "baseline_models" / f"confusion_matrix_{model_name.lower()}.png"
        root_path = OUTPUT_DIR / f"baseline_{model_name.lower()}_confusion_matrix.png"

        subtitle = f"Smoke-Test Benchmark | 45 Multi-Domain Features | Held-Out Test Split (N=30)"
        plot_confusion_matrix(
            cm=cm,
            model_name=f"VisionGuard {model_name} (Production Baseline)",
            subtitle=subtitle,
            metrics=test_data,
            out_path=subfolder_path,
            cmap_name="Blues",
            threshold=thresh,
        )
        plot_confusion_matrix(
            cm=cm,
            model_name=f"VisionGuard {model_name} (Production Baseline)",
            subtitle=subtitle,
            metrics=test_data,
            out_path=root_path,
            cmap_name="Blues",
            threshold=thresh,
        )

        grid_item = {
            "name": f"{model_name} (Baseline)",
            "cm": cm,
            "metrics": test_data,
        }
        baseline_grid_data.append(grid_item)
        all_grid_data.append(grid_item)

    plot_grid(
        baseline_grid_data,
        title="VisionGuard Baseline Models — Test Confusion Matrices",
        subtitle="Smoke-Test Evaluation (100 base images / 30 held-out test samples, 45 features)",
        out_path=OUTPUT_DIR / "baseline_models" / "all_baseline_models_comparison.png",
    )

    # ==========================================
    # Process Development Candidates (65 features, 3500 samples)
    # ==========================================
    print("Generating Development Candidate Models Confusion Matrices...")
    for model_name, data in candidate_metrics.items():
        test_data = data["test"]
        cm = test_data["confusion_matrix"]
        thresh = data.get("threshold", 0.5)
        is_best = (model_name == "CatBoost")
        name_tag = f"{model_name} (Best Candidate)" if is_best else f"{model_name} Candidate"

        subfolder_path = OUTPUT_DIR / "development_candidates" / f"confusion_matrix_{model_name.lower()}_candidate.png"
        root_path = OUTPUT_DIR / f"candidate_{model_name.lower()}_confusion_matrix.png"

        subtitle = "Expanded Benchmark | 65 Multi-Domain Features | 3,500 Images (N=525 Test Split)"
        plot_confusion_matrix(
            cm=cm,
            model_name=f"VisionGuard {name_tag}",
            subtitle=subtitle,
            metrics=test_data,
            out_path=subfolder_path,
            cmap_name="Purples" if is_best else "Blues",
            threshold=thresh,
        )
        plot_confusion_matrix(
            cm=cm,
            model_name=f"VisionGuard {name_tag}",
            subtitle=subtitle,
            metrics=test_data,
            out_path=root_path,
            cmap_name="Purples" if is_best else "Blues",
            threshold=thresh,
        )

        grid_item = {
            "name": f"{model_name} (Candidate 65F)",
            "cm": cm,
            "metrics": test_data,
        }
        candidate_grid_data.append(grid_item)
        all_grid_data.append(grid_item)

    plot_grid(
        candidate_grid_data,
        title="VisionGuard Development Candidates — Test Confusion Matrices",
        subtitle="Expanded Benchmark (3,500 samples across 6 attack families, 525 held-out test samples, 65 features)",
        out_path=OUTPUT_DIR / "development_candidates" / "all_candidate_models_comparison.png",
    )

    # ==========================================
    # Process Reference-Aware Pair Detector (130 features)
    # ==========================================
    print("Generating Pair Detector Confusion Matrix...")
    pair_test = pair_metadata["test"]
    pair_cm = pair_test["confusion_matrix"]
    pair_thresh = pair_metadata.get("threshold", 0.05)

    pair_subfolder = OUTPUT_DIR / "pair_detector" / "confusion_matrix_pair_detector.png"
    pair_root = OUTPUT_DIR / "pair_detector_confusion_matrix.png"

    pair_sub = "Reference-Aware Pair Detector | 130 Delta Features | Held-Out Test Pairs (N=525)"
    plot_confusion_matrix(
        cm=pair_cm,
        model_name="Reference-Aware Pair Detector (RandomForest)",
        subtitle=pair_sub,
        metrics=pair_test,
        out_path=pair_subfolder,
        cmap_name="Greens",
        threshold=pair_thresh,
    )
    plot_confusion_matrix(
        cm=pair_cm,
        model_name="Reference-Aware Pair Detector (RandomForest)",
        subtitle=pair_sub,
        metrics=pair_test,
        out_path=pair_root,
        cmap_name="Greens",
        threshold=pair_thresh,
    )

    all_grid_data.append({
        "name": "Pair Detector (130F)",
        "cm": pair_cm,
        "metrics": pair_test,
    })

    # Master Comparative Grid
    plot_grid(
        all_grid_data,
        title="VisionGuard Comprehensive Model Zoo — All Confusion Matrices",
        subtitle="Overview of all Production Baselines, Expanded Development Candidates, and Pair Detector",
        out_path=OUTPUT_DIR / "ALL_MODELS_confusion_matrix_overview.png",
    )

    # ==========================================
    # Write Index README in the folder
    # ==========================================
    readme_content = """# VisionGuard Model Confusion Matrices

This folder contains high-resolution (300 DPI) confusion matrix visualizations and diagnostic performance metrics for all machine learning models in VisionGuard.

---

## 📁 Directory Structure

```
confusion_matrices/
├── ALL_MODELS_confusion_matrix_overview.png   <-- Complete 11-model comparison grid
│
├── [Individual Direct Root Files]
│   ├── baseline_randomforest_confusion_matrix.png
│   ├── baseline_xgboost_confusion_matrix.png
│   ├── baseline_lightgbm_confusion_matrix.png
│   ├── baseline_catboost_confusion_matrix.png
│   ├── baseline_svm_confusion_matrix.png
│   ├── candidate_catboost_confusion_matrix.png
│   ├── candidate_randomforest_confusion_matrix.png
│   ├── candidate_xgboost_confusion_matrix.png
│   ├── candidate_lightgbm_confusion_matrix.png
│   ├── candidate_svm_confusion_matrix.png
│   └── pair_detector_confusion_matrix.png
│
├── baseline_models/                            <-- 45-feature production models
│   ├── confusion_matrix_randomforest.png
│   ├── confusion_matrix_xgboost.png
│   ├── confusion_matrix_lightgbm.png
│   ├── confusion_matrix_catboost.png
│   ├── confusion_matrix_svm.png
│   └── all_baseline_models_comparison.png     <-- Grid of all 5 baseline models
│
├── development_candidates/                     <-- 65-feature expanded models (3,500 images)
│   ├── confusion_matrix_catboost_candidate.png
│   ├── confusion_matrix_randomforest_candidate.png
│   ├── confusion_matrix_xgboost_candidate.png
│   ├── confusion_matrix_lightgbm_candidate.png
│   ├── confusion_matrix_svm_candidate.png
│   └── all_candidate_models_comparison.png     <-- Grid of all 5 candidate models
│
└── pair_detector/                              <-- 130-feature reference-aware detector
    └── confusion_matrix_pair_detector.png
```

---

## 📊 Summary of Model Confusion Metrics (Held-Out Test Splits)

### 1. Production Baseline Models (45 Features, Smoke-Test Benchmark, N=30)
Evaluated on 15 clean and 15 manipulated images from held-out COCO-2017 base IDs:

| Model | TN | FP | FN | TP | Accuracy | Precision | Recall | F1-Score | Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForest (Production)** | **13** | **2** | **4** | **11** | **80.0%** | **84.6%** | **73.3%** | **0.786** | 0.490 |
| **XGBoost** | 12 | 3 | 3 | 12 | 80.0% | 80.0% | 80.0% | 0.800 | 0.269 |
| **LightGBM** | 14 | 1 | 7 | 8 | 73.3% | 88.9% | 53.3% | 0.667 | 0.631 |
| **CatBoost** | 11 | 4 | 4 | 11 | 73.3% | 73.3% | 73.3% | 0.733 | 0.327 |
| **SVM** | 0 | 15 | 3 | 12 | 40.0% | 44.4% | 80.0% | 0.571 | 0.483 |

---

### 2. Expanded Development Candidates (65 Features, 3,500 Images Benchmark, N=525)
Evaluated across 75 clean base images and 450 manipulated variations across 6 attack families:

| Model | TN | FP | FN | TP | Accuracy | Precision | Recall | F1-Score | Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CatBoost (Best Candidate)** | **66** | **9** | **258** | **192** | **49.1%** | **95.5%** | **42.7%** | **0.590** | 0.736 |
| **RandomForest** | 67 | 8 | 269 | 181 | 47.2% | 95.8% | 40.2% | 0.567 | 0.800 |
| **XGBoost** | 69 | 6 | 283 | 167 | 45.0% | 96.5% | 37.1% | 0.536 | 0.827 |
| **LightGBM** | 71 | 4 | 284 | 166 | 45.1% | 97.6% | 36.9% | 0.535 | 0.965 |
| **SVM (RBF + Scaler)** | 70 | 5 | 305 | 145 | 41.0% | 96.7% | 32.2% | 0.483 | 0.865 |

---

### 3. Reference-Aware Pair Detector (130 Signed + Absolute Delta Features, N=525)
Evaluated in the Prompt Injection Lab context comparing candidate image against reference clean image:

| Model | TN | FP | FN | TP | Accuracy | Precision | Recall | F1-Score | Threshold |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForest Pair Detector** | **75** | **0** | **13** | **437** | **97.5%** | **100.0%** | **97.1%** | **0.985** | 0.050 |

---

*Generated automatically for VisionGuard security and model evaluation research.*
"""
    with open(OUTPUT_DIR / "README.md", "w") as f:
        f.write(readme_content)

    print(f"All confusion matrix images successfully generated in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

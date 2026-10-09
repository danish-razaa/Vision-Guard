# VisionGuard Model Confusion Matrices

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

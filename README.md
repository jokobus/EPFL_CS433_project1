# EPFL CS-433: Machine Learning — Project 1
## Myocardial Infarction & Coronary Heart Disease Risk Prediction

This repository contains the complete codebase, empirical evaluation, and reproducible pipeline for **Project 1** in the EPFL Machine Learning course (CS-433, Fall 2026).

---

## 1. Project Overview & Challenges

- **Objective:** Predict whether an individual has experienced coronary heart disease or myocardial infarction (`_MICHD` $\in \{-1, +1\}$) based on lifestyle and clinical variables from the CDC Behavioral Risk Factor Surveillance System (BRFSS 2015).
- **Scale:** 328,135 training samples, 109,379 test samples, 321 raw features.
- **Key Challenges:**
  1. **Extreme Class Imbalance:** Positive cases (`_MICHD = 1`) represent only **8.83%** of the dataset (28,975 positive vs. 299,160 negative). A naive classifier predicting the majority class achieves 91.17% accuracy but an $F_1$ score of $\approx 0.00$, rendering it clinically useless.
  2. **High Feature Missingness:** 147 out of 321 survey features have $>50\%$ missing values due to optional state-specific phone modules.
  3. **Strict Library Constraint:** Only NumPy and standard Python libraries are allowed for modeling, preprocessing, and validation. No external ML toolboxes (`scikit-learn`, `pandas`, `pytorch`, etc.).

---

## 2. Directory Structure

```
projects/project1/
├── implementations.py                # 6 mandatory baseline methods (passes course tests)
├── helpers.py                        # Course-provided utility functions
├── run.py                            # Standalone script reproducing submission.csv
├── exploration_and_experiments.ipynb # Interactive EDA, visualizations & ablation notebook
├── README.md                         # This documentation and student guide
├── submission.csv                    # Final test set predictions (109,379 rows)
│
├── src/                              # Clean, modular pipeline components (NumPy only)
│   ├── __init__.py
│   ├── preprocessing.py              # PreprocessingPipeline: filtering, indicators, imputation, scaling
│   ├── metrics.py                    # Confusion matrix, Acc, Prec, Rec, F1, and threshold optimizer
│   ├── models.py                     # BaseModel wrappers linking implementations.py to pipeline
│   └── cross_validation.py           # Stratified K-Fold CV and grid search routines
│
├── data/
│   ├── dataset_to_release.zip        # Original dataset archive
│   └── dataset/                      # Extracted CSVs (x_train.csv, y_train.csv, x_test.csv)
│
├── grading_tests/                    # Official public grading test suite
│   ├── conftest.py
│   ├── environment.yml
│   ├── INSTRUCTIONS.md
│   └── test_project1_public.py
│
└── latex-example-paper/              # 2-Page IEEE Conference Paper
    ├── IEEEtran.cls
    ├── IEEEtran.bst
    ├── project1_report.tex           # Complete 2-page LaTeX paper
    ├── project1_report.pdf           # Compiled publication-ready PDF report
    ├── literature.bib                # Bibliography
    └── figures/                      # High-res publication plots
        ├── threshold_curve.pdf       # F1 vs decision threshold curve
        └── model_ablation.pdf        # Ablation study comparison bar chart
```

---

## 3. The 6 Mandatory Methods in `implementations.py`

All 6 functions strictly follow the course specifications and return `(w, loss)`:
1. `mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)` — Linear regression via full Gradient Descent.
2. `mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)` — Linear regression via SGD with mini-batch size 1.
3. `least_squares(y, tx)` — Closed-form normal equations: $w = (X^T X)^{-1} X^T y$.
4. `ridge_regression(y, tx, lambda_)` — Regularized normal equations: $w = (X^T X + 2N\lambda I)^{-1} X^T y$. Returns unpenalized MSE loss.
5. `logistic_regression(y, tx, initial_w, max_iters, gamma)` — Logistic regression via Gradient Descent with sigmoid activation and cross-entropy loss ($y \in \{0, 1\}$).
6. `reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma)` — L2 regularized logistic regression. Returns unpenalized cross-entropy loss.

> **Verification:** Passed all 10 public test cases in `grading_tests/test_project1_public.py`.

---

## 4. Methodology & Engineering Strategy

Our modeling and pipeline strategy comprises five principal steps:

### Step 1: Implement & Validate the 6 Core Functions
Implement the formulas in `implementations.py` using vectorization (avoid Python for-loops over samples). Verify your implementations using the small numerical test cases in `grading_tests/test_project1_public.py`.

### Step 2: Principled CDC Survey Recoding & High-Yield Retention
- **Special Survey Codes:** CDC BRFSS uses `88` (count variables) and `888` (alcohol `ALCDAY5`) to denote zero occurrences (0 days or 0 drinks), whereas `7, 9, 77, 99, 777, 999, 9999` indicate "Don't know" or "Refused". Leaving `88/888` as numerical values severely distorts linear models.
- **High-Yield Screening Retention:** Naively dropping columns with $>50\%$ missingness discards the single most predictive feature in the dataset: `CVDASPRN` (daily aspirin therapy for heart disease prevention, correlation $r = -0.273$, positive prevalence of $24.6\%$ vs $4.6\%$). We retain high-yield clinical screening variables (`CVDASPRN`, `BPMEDS`, `DRADVISE`, `CDASSIST`, `ARTTODAY`) regardless of module missingness.
- **Ternary Encoding:** For binary questions, we map Yes $\to +1$, No $\to -1$, and unasked/missing $\to 0$. This ensures a neutral zero baseline for linear models.

### Step 3: Non-Linear Expansions & Clinical Interactions
- **One-Hot Discrete Expansion:** Categorical variables with $3 \le K \le 10$ unique levels as well as `_AGEG5YR` (14 five-year age tiers) are expanded into dummy indicator columns, capturing non-linear step risks.
- **Clinical Explaining-Away Interactions:** Mobility impairment (`DIFFWALK`) is strongly associated with heart disease, but also with arthritis (`HAVARTH3`). Including interaction terms (`DIFFWALK * HAVARTH3`) enables the linear hyperplane to disentangle joint-related disability from cardiac disability.
- **Standardization & Preallocation:** Continuous features are normalized ($z = (x - \mu)/\sigma$), with polynomial expansions on top predictors. All feature matrices are preallocated directly in contiguous `float32`, keeping total memory usage below 600 MB.

### Step 4: Cost-Sensitive Ensembling via Zero-Copy Matrix Subtraction
- **Sample Weighting:** We apply class weight $w_+ = 3.5$ to positive cases ($w_- = 1.0$) to penalize false negatives during training.
- **Zero-Copy Subtraction Mathematics:** Standard 5-fold cross-validation on 328k rows typically creates large temporary array allocations (`tx * weights[:, None]`), triggering out-of-memory errors on 32-bit/64-bit systems. We compute the weighted Gram matrix on the full dataset once:
  $$A_{\text{full}} = X^T X + (w_+ - 1) X_+^T X_+ \in \mathbb{R}^{D \times D}$$
  $$b_{\text{full}} = X^T y + (w_+ - 1) X_+^T y_+ \in \mathbb{R}^D$$
  and for each fold $k$, we obtain the training Gram matrix by direct subtraction:
  $$A_{\text{tr}}^{(k)} = A_{\text{full}} - A_{\text{val}}^{(k)}, \quad b_{\text{tr}}^{(k)} = b_{\text{full}} - b_{\text{val}}^{(k)}$$
  This trains all 5 folds on 328,135 samples in just **60 seconds** with zero extra memory overhead!

### Step 5: Optimal Threshold Tuning
- Continuous out-of-fold predictions from all 5 models are averaged.
- The decision threshold $\tau^*$ is optimized via grid search on validation folds to maximize $F_1 = 2 \cdot \frac{\text{Prec} \cdot \text{Rec}}{\text{Prec} + \text{Rec}}$.
- Setting $\tau^* = -0.0730$ balances precision and recall, raising $F_1$ from baseline 0.4069 to **0.4302** (leaderboard test $F_1 > 0.43$).

---

## 5. Summary of Experimental Results (5-Fold Stratified CV on 328k Rows)

| Model / Configuration | Features | Accuracy | Precision | Recall | $F_1$ Score | Training Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Ridge Regression ($\tau = 0.0$, default) | 321 | **91.71%** | 66.67% | 0.48% | 0.0096 | 28s |
| Least Squares (pinv fallback) | 321 | 84.12% | 28.40% | 35.29% | 0.3146 | 32s |
| Logistic Regression (GD, 500 iters) | 321 | 86.14% | 32.75% | 49.01% | 0.3924 | 240s |
| Initial Baseline Ridge ($\lambda=10^{-4}, \tau^*$) | 321 | 86.85% | 33.80% | 51.10% | 0.4069 | 28s |
| + Survey Recoding & Ternary {-1,0,1} | 321 | 87.11% | 34.62% | 51.45% | 0.4137 | 30s |
| + High-Yield Screening (`CVDASPRN`, `BPMEDS`) | 328 | 87.15% | 34.78% | 52.36% | 0.4180 | 32s |
| + One-Hot Discrete Expansion (`_AGEG5YR`, etc.) | 718 | 87.33% | 35.40% | 53.18% | 0.4250 | 48s |
| **Full Pipeline: 5-Fold Weighted Ridge Ensemble (CV)** | **742** | **87.48%** | **35.96%** | **53.53%** | **0.4302** | **60s** |
| **Official Competition Test Set (AIcrowd Submission #338235)** | **742** | **87.80%** | — | — | **0.4440** | — |

---

## 6. How to Run the Solution

Ensure you are in the `epfl` conda environment:
```bash
conda activate epfl
```

### Reproduce Final Submission
Run the complete end-to-end pipeline:
```bash
python run.py
```
This automatically loads the dataset, cleans features, fits the model, tunes the threshold, and outputs `submission.csv`.

### Interactive Exploration
Launch Jupyter Notebook to view the walkthrough:
```bash
jupyter notebook exploration_and_experiments.ipynb
```

### Compile the LaTeX Report
Compile the 2-page report into PDF:
```bash
cd latex-example-paper
pdflatex project1_report.tex
bibtex project1_report
pdflatex project1_report.tex
pdflatex project1_report.tex
```
The output PDF is generated at `projects/project1/latex-example-paper/project1_report.pdf`.

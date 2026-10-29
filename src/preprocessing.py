# -*- coding: utf-8 -*-
"""Data preprocessing, cleaning, imputation, and feature engineering for Project 1.

EPFL Machine Learning Course (CS-433)
Fall 2026

All operations strictly utilize NumPy and Python standard libraries.
Optimized for BRFSS medical survey data:
- In-place CDC survey code cleaning (88/888 -> 0 for counts, 7/9/77/99/777/999/9999 -> NaN)
- Strategic inclusion of core features (<55% missing) PLUS high-yield screening questions (CVDASPRN, BPMEDS, etc.)
- Ternary {-1, 0, +1} mapping for binary survey questions
- One-hot / dummy encoding of discrete ordinal/nominal features (cardinality 3-10 + _AGEG5YR)
- Clinical comorbidity scores and high-yield pairwise interaction terms
- Targeted quadratic expansion on top continuous predictors
- Preallocated contiguous float32 output matrices for zero memory fragmentation (~590 MB peak RAM)
"""

import os
import gc
import numpy as np

HIGH_YIELD_SCREENING_FEATURES = {
    "CVDASPRN",
    "BPMEDS",
    "DRADVISE",
    "CDASSIST",
    "ARTTODAY",
    "ASPUNSAF",
    "ADENERGY",
    "ADEAT1",
    "CDHOUSE",
    "CDSOCIAL",
    "VIEYEXM2",
}


class PreprocessingPipeline:
    """End-to-end memory-efficient preprocessing pipeline for BRFSS tabular data."""

    def __init__(
        self,
        max_missing_ratio=0.55,
        clean_cdc_codes=True,
        add_missing_indicators=True,
        ternary_binary=True,
        one_hot_categorical=True,
        add_clinical_interactions=True,
        standardize=True,
        add_bias=True,
        feature_names=None,
    ):
        self.max_missing_ratio = max_missing_ratio
        self.clean_cdc_codes = clean_cdc_codes
        self.add_missing_indicators = add_missing_indicators
        self.ternary_binary = ternary_binary
        self.one_hot_categorical = one_hot_categorical
        self.add_clinical_interactions = add_clinical_interactions
        self.standardize = standardize
        self.add_bias = add_bias
        self.feature_names = feature_names

        # Fitted parameters
        self.kept_indices = None
        self.kept_names = None
        self.name_to_col_idx = None
        self.binary_cols = []
        self.cat_cols = []
        self.cat_levels = []
        self.cont_cols = []
        self.cont_means = []
        self.cont_stds = []
        self.cont_quad_cols = []
        self.has_nan_cols = []
        self.interaction_pairs = []
        self.n_out_cols = 0

    def _clean_special_codes_inplace(self, X):
        """Recode CDC survey special codes in-place."""
        n_samples, n_features = X.shape
        for j in range(n_features):
            col = X[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                continue

            unique_vals = np.unique(valid)
            # Recode 88 or 888 (None / zero count) -> 0.0
            if 88.0 in unique_vals and np.any(
                (unique_vals >= 1.0) & (unique_vals <= 30.0)
            ):
                X[col == 88.0, j] = 0.0
            if 888.0 in unique_vals:
                X[col == 888.0, j] = 0.0

            # Recode Don't Know / Refused codes (7, 9, 77, 99, 777, 999, 9999) to NaN
            for special in [7.0, 9.0, 77.0, 99.0, 777.0, 999.0, 9999.0]:
                if special in unique_vals:
                    X[col == special, j] = np.nan

        return X

    def fit(self, X):
        """Fit preprocessing parameters on the training feature matrix."""
        n_samples, n_features = X.shape

        # 1. Filter features: keep core features + high-yield screening features
        nan_ratios = np.isnan(X).mean(axis=0)
        kept = []
        for i in range(n_features):
            fname = (
                self.feature_names[i] if self.feature_names is not None else f"f_{i}"
            )
            if (
                nan_ratios[i] <= self.max_missing_ratio
                or fname in HIGH_YIELD_SCREENING_FEATURES
            ):
                kept.append(i)

        self.kept_indices = np.array(kept, dtype=int)
        X_sub = X[:, self.kept_indices]

        if self.feature_names is not None:
            self.kept_names = [self.feature_names[i] for i in self.kept_indices]
            self.name_to_col_idx = {name: i for i, name in enumerate(self.kept_names)}
        else:
            self.kept_names = [f"f_{i}" for i in self.kept_indices]
            self.name_to_col_idx = {name: i for i, name in enumerate(self.kept_names)}

        # 2. Clean survey special codes in-place on slice
        if self.clean_cdc_codes:
            self._clean_special_codes_inplace(X_sub)

        # 3. Categorize features and compute statistics
        n_kept = X_sub.shape[1]
        self.binary_cols = []
        self.cat_cols = []
        self.cat_levels = []
        self.cont_cols = []
        self.cont_means = []
        self.cont_stds = []
        self.has_nan_cols = []

        for j in range(n_kept):
            col = X_sub[:, j]
            valid = col[~np.isnan(col)]
            fname = self.kept_names[j]
            nr = np.isnan(col).mean()

            # Add missingness indicator for informative missing columns
            if self.add_missing_indicators and (
                0.01 <= nr <= 0.85 or fname in HIGH_YIELD_SCREENING_FEATURES
            ):
                self.has_nan_cols.append(j)

            u = np.unique(valid) if len(valid) > 0 else np.array([])

            # Binary survey questions: {1, 2}
            if self.ternary_binary and len(u) == 2 and set(u).issubset({1.0, 2.0}):
                self.binary_cols.append(j)

            # Discrete categorical/ordinal: 3 to 10 unique integer values OR _AGEG5YR (14 levels)
            elif (
                self.one_hot_categorical
                and (3 <= len(u) <= 10 or fname == "_AGEG5YR")
                and np.all(np.equal(np.mod(u, 1), 0))
            ):
                self.cat_cols.append(j)
                self.cat_levels.append(
                    u[:-1]
                )  # drop one level to avoid multicollinearity

            # Continuous / count features
            else:
                self.cont_cols.append(j)
                med = float(np.nanmedian(col)) if len(valid) > 0 else 0.0
                col_imp = np.where(np.isnan(col), med, col)
                m_v = float(np.mean(col_imp))
                s_v = float(np.std(col_imp))
                s_v = s_v if s_v > 1e-8 else 1.0
                self.cont_means.append(m_v)
                self.cont_stds.append(s_v)

        # Select top continuous features for quadratic expansion (top 25 by variance)
        if len(self.cont_stds) > 0:
            top_cont_idx = np.argsort(self.cont_stds)[-25:]
            self.cont_quad_cols = [self.cont_cols[idx] for idx in top_cont_idx]
        else:
            self.cont_quad_cols = []

        # 4. Clinical interaction pairs
        self.interaction_pairs = []
        if self.add_clinical_interactions and self.name_to_col_idx:
            candidate_pairs = [
                ("_AGEG5YR", "GENHLTH"),
                ("_AGEG5YR", "BPHIGH4"),
                ("_AGEG5YR", "DIABETE3"),
                ("_AGEG5YR", "_SMOKER3"),
                ("_AGEG5YR", "SEX"),
                ("GENHLTH", "PHYSHLTH"),
                ("BPHIGH4", "DIABETE3"),
                ("BPHIGH4", "_BMI5"),
                ("DIFFWALK", "GENHLTH"),
                ("CVDASPRN", "_AGEG5YR"),
                ("CVDSTRK3", "_AGEG5YR"),
                ("BPMEDS", "BPHIGH4"),
                ("FC60_", "_AGEG5YR"),
                ("MAXVO2_", "GENHLTH"),
                ("DIFFWALK", "HAVARTH3"),
            ]
            for n1, n2 in candidate_pairs:
                if n1 in self.name_to_col_idx and n2 in self.name_to_col_idx:
                    self.interaction_pairs.append(
                        (self.name_to_col_idx[n1], self.name_to_col_idx[n2])
                    )

        # Calculate exact total output column dimension
        total_cols = 0
        if self.add_bias:
            total_cols += 1
        if self.add_missing_indicators:
            total_cols += len(self.has_nan_cols)
        if self.ternary_binary:
            total_cols += len(self.binary_cols)
        if self.one_hot_categorical:
            total_cols += sum(len(lvl) for lvl in self.cat_levels)
        total_cols += len(self.cont_cols) + len(self.cont_quad_cols)
        if self.add_clinical_interactions:
            total_cols += len(self.interaction_pairs)

        self.n_out_cols = total_cols
        return self

    def transform(self, X):
        """Transform raw feature matrix directly into preallocated contiguous float32 matrix."""
        n_samples = len(X)
        X_sub = X[:, self.kept_indices]

        if self.clean_cdc_codes:
            self._clean_special_codes_inplace(X_sub)

        tx = np.empty((n_samples, self.n_out_cols), dtype=np.float32)
        out_idx = 0

        # 1. Bias intercept
        if self.add_bias:
            tx[:, out_idx] = 1.0
            out_idx += 1

        # 2. Missingness indicators
        if self.add_missing_indicators and len(self.has_nan_cols) > 0:
            for j in self.has_nan_cols:
                tx[:, out_idx] = np.isnan(X_sub[:, j])
                out_idx += 1

        # 3. Binary features: 1 -> +1 (Yes), 2 -> -1 (No), NaN -> 0
        if self.ternary_binary and len(self.binary_cols) > 0:
            for j in self.binary_cols:
                col = X_sub[:, j]
                b_col = np.zeros(n_samples, dtype=np.float32)
                b_col[col == 1.0] = 1.0
                b_col[col == 2.0] = -1.0
                tx[:, out_idx] = b_col
                out_idx += 1

        # 4. Categorical one-hot features
        if self.one_hot_categorical and len(self.cat_cols) > 0:
            for j, levels in zip(self.cat_cols, self.cat_levels):
                col = X_sub[:, j]
                for val in levels:
                    tx[:, out_idx] = col == val
                    out_idx += 1

        # 5. Continuous features and selected quadratic terms
        quad_set = set(self.cont_quad_cols)
        if len(self.cont_cols) > 0:
            for j, m_v, s_v in zip(self.cont_cols, self.cont_means, self.cont_stds):
                col = X_sub[:, j]
                col_imp = np.where(np.isnan(col), m_v, col)
                c_norm = ((col_imp - m_v) / s_v).astype(np.float32)
                tx[:, out_idx] = c_norm
                out_idx += 1
                if j in quad_set:
                    tx[:, out_idx] = c_norm**2
                    out_idx += 1

        # 6. Clinical interaction terms
        if self.add_clinical_interactions and len(self.interaction_pairs) > 0:
            for i1, i2 in self.interaction_pairs:
                c1 = np.nan_to_num(X_sub[:, i1], nan=0.0)
                c2 = np.nan_to_num(X_sub[:, i2], nan=0.0)
                inter = (c1 * c2).astype(np.float32)
                s = float(np.std(inter))
                if s > 1e-8:
                    tx[:, out_idx] = (inter - float(np.mean(inter))) / s
                else:
                    tx[:, out_idx] = 0.0
                out_idx += 1

        return tx

    def fit_transform(self, X):
        """Fit and transform feature matrix in one call."""
        return self.fit(X).transform(X)


def count_file_lines(path):
    """Count lines in file skipping header."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        f.readline()
        return sum(1 for _ in f)


def load_feature_matrix_preallocated(csv_path, max_samples=None, n_cols=321):
    """Memory-efficient CSV feature loader directly into preallocated float32 array."""
    if max_samples is not None:
        n_rows = max_samples
    else:
        n_rows = count_file_lines(csv_path)

    arr = np.empty((n_rows, n_cols), dtype=np.float32)
    ids = np.empty(n_rows, dtype=int)

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        f.readline()
        for i in range(n_rows):
            line = f.readline()
            if not line:
                break
            parts = line.strip().split(",")
            ids[i] = int(parts[0])
            for j, v in enumerate(parts[1:]):
                arr[i, j] = float(v) if v != "" else np.nan

    return arr, ids


def load_labels_preallocated(y_path, max_samples=None):
    """Memory-efficient label loader directly into preallocated int array."""
    if max_samples is not None:
        n_rows = max_samples
    else:
        n_rows = count_file_lines(y_path)

    y = np.empty(n_rows, dtype=int)
    ids = np.empty(n_rows, dtype=int)

    with open(y_path, "r", encoding="utf-8", errors="ignore") as f:
        f.readline()
        for i in range(n_rows):
            line = f.readline()
            if not line:
                break
            parts = line.strip().split(",")
            ids[i] = int(parts[0])
            y[i] = int(parts[1])

    return y, ids


def load_train_dataset(data_dir, max_samples=None):
    """Load training features and labels with independent memory lifecycle."""
    y_path = os.path.join(data_dir, "y_train.csv")
    x_tr_path = os.path.join(data_dir, "x_train.csv")
    y_train, train_ids = load_labels_preallocated(y_path, max_samples=max_samples)
    X_train, _ = load_feature_matrix_preallocated(x_tr_path, max_samples=max_samples)
    return X_train, y_train, train_ids


def load_test_dataset(data_dir, max_samples=None):
    """Load test features and ids with independent memory lifecycle."""
    x_te_path = os.path.join(data_dir, "x_test.csv")
    X_test, test_ids = load_feature_matrix_preallocated(
        x_te_path, max_samples=max_samples
    )
    return X_test, test_ids


def load_raw_dataset(data_dir, max_samples=None):
    """Convenience loader returning both training and testing data."""
    X_train, y_train, train_ids = load_train_dataset(data_dir, max_samples=max_samples)
    X_test, test_ids = load_test_dataset(data_dir, max_samples=max_samples)
    return X_train, y_train, X_test, train_ids, test_ids

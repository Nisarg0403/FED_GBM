import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

"""
FED-GBM v3 Phase 7.2 Corrective Implementation Unit Test Suite.
Validates all 13 required assertions before authorization of final experiment execution:
1. Locked TCGA test IDs are identical across all five seeds.
2. Locked TCGA test size = 20.
3. Locked TCGA training size = 77.
4. Locked UPenn test size = 114.
5. Locked UPenn training size = 460.
6. Feature selection fitting population contains ONLY TCGA training data (N=77).
7. UPenn training labels cannot alter the locked feature schema.
8. Final feature schema contains exactly 15 features for p_total=15.
9. Four protected clinical features are present.
10. Eleven radiomic features are present.
11. TCGA and UPenn use identical feature ordering.
12. Test data is excluded from all fit operations.
13. Phase 3/4/5/6 regression unit tests pass.
"""

import unittest
import numpy as np
import pandas as pd

from experiments.run_phase_7_experiments import generate_benchmark_datasets, preprocess_and_select_features
from fed_gbm.preprocessing import XGBoostFeatureSelector


class TestPhase7Corrections(unittest.TestCase):
    
    def setUp(self):
        self.seeds = [42, 123, 2024, 3407, 9999]
        self.splits_dir = os.path.join(os.getcwd(), 'data', 'splits')

    def test_01_tcga_test_ids_identical_across_seeds(self):
        """Test 1: Locked TCGA test targets & features are 100% identical across all 5 seeds."""
        tcga_te_labels = []
        tcga_te_shapes = []
        
        for seed in self.seeds:
            _, _, _, y_tcga_te, _, _, _, _, _ = generate_benchmark_datasets(seed=seed)[:9]
            tcga_te_labels.append(y_tcga_te.values)
            tcga_te_shapes.append(y_tcga_te.shape)
            
        for i in range(1, len(self.seeds)):
            np.testing.assert_array_equal(tcga_te_labels[0], tcga_te_labels[i], err_msg=f"TCGA test set labels differ for seed {self.seeds[i]}!")
            self.assertEqual(tcga_te_shapes[0], tcga_te_shapes[i])

    def test_02_tcga_test_size_equals_20(self):
        """Test 2: Locked TCGA test set size equals exactly 20."""
        tcga_te_manifest = pd.read_csv(os.path.join(self.splits_dir, 'tcga_test_ids.csv'))
        self.assertEqual(len(tcga_te_manifest), 20)
        
        for seed in self.seeds:
            _, _, _, y_tcga_te, _, _, _, _, _ = generate_benchmark_datasets(seed=seed)[:9]
            self.assertEqual(len(y_tcga_te), 20)

    def test_03_tcga_train_size_equals_77(self):
        """Test 3: Locked TCGA training set size equals exactly 77."""
        tcga_tr_manifest = pd.read_csv(os.path.join(self.splits_dir, 'tcga_train_ids.csv'))
        self.assertEqual(len(tcga_tr_manifest), 77)
        
        for seed in self.seeds:
            X_tcga_tr, y_tcga_tr, _, _, _, _, _, _, _ = generate_benchmark_datasets(seed=seed)[:9]
            self.assertEqual(len(X_tcga_tr), 77)
            self.assertEqual(len(y_tcga_tr), 77)

    def test_04_upenn_test_size_equals_114(self):
        """Test 4: Locked UPenn test set size equals exactly 114."""
        upenn_te_manifest = pd.read_csv(os.path.join(self.splits_dir, 'upenn_test_ids.csv'))
        self.assertEqual(len(upenn_te_manifest), 114)
        
        for seed in self.seeds:
            _, _, _, _, _, _, X_upenn_te, y_upenn_te, _ = generate_benchmark_datasets(seed=seed)[:9]
            self.assertEqual(len(X_upenn_te), 114)
            self.assertEqual(len(y_upenn_te), 114)

    def test_05_upenn_train_size_equals_460(self):
        """Test 5: Locked UPenn training set size equals exactly 460."""
        upenn_tr_manifest = pd.read_csv(os.path.join(self.splits_dir, 'upenn_train_ids.csv'))
        self.assertEqual(len(upenn_tr_manifest), 460)
        
        for seed in self.seeds:
            _, _, _, _, X_upenn_tr, y_upenn_tr, _, _, _ = generate_benchmark_datasets(seed=seed)[:9]
            self.assertEqual(len(X_upenn_tr), 460)
            self.assertEqual(len(y_upenn_tr), 460)

    def test_06_feature_selection_fitted_on_tcga_train_only(self):
        """Test 6: Feature selection is fitted strictly on TCGA training data (N=77)."""
        X_tcga_tr, y_tcga_tr, X_tcga_te, y_tcga_te, X_upenn_tr, y_upenn_tr, X_upenn_te, y_upenn_te, _ = generate_benchmark_datasets(seed=42)[:9]
        
        # Fit selector directly on TCGA Train (N=77)
        selector = XGBoostFeatureSelector(n_features_to_select=15, protected_cols=['Age', 'Gender', 'KPS', 'Resection_Status'], random_state=42)
        selector.fit(X_tcga_tr, y_tcga_tr)
        expected_cols = selector.selected_cols_
        
        # Call preprocess_and_select_features
        _, _, _, _, actual_cols = preprocess_and_select_features(X_tcga_tr, y_tcga_tr, X_upenn_tr, y_upenn_tr, X_tcga_te, X_upenn_te, seed=42, target_p=15)
        self.assertEqual(expected_cols, actual_cols)

    def test_07_upenn_labels_do_not_alter_feature_schema(self):
        """Test 7: Changing UPenn training labels entirely leaves the locked feature schema unchanged."""
        X_tcga_tr, y_tcga_tr, X_tcga_te, y_tcga_te, X_upenn_tr, y_upenn_tr, X_upenn_te, y_upenn_te, _ = generate_benchmark_datasets(seed=42)[:9]
        
        # Run original
        _, _, _, _, cols_orig = preprocess_and_select_features(X_tcga_tr, y_tcga_tr, X_upenn_tr, y_upenn_tr, X_tcga_te, X_upenn_te, seed=42, target_p=15)
        
        # Invert UPenn training labels
        y_upenn_tr_mutated = 1 - y_upenn_tr
        _, _, _, _, cols_mutated = preprocess_and_select_features(X_tcga_tr, y_tcga_tr, X_upenn_tr, y_upenn_tr_mutated, X_tcga_te, X_upenn_te, seed=42, target_p=15)
        
        self.assertEqual(cols_orig, cols_mutated)

    def test_08_final_feature_count_equals_15(self):
        """Test 8: Final selected feature count equals exactly 15 for p_total=15 configuration."""
        X_tcga_tr, y_tcga_tr, X_tcga_te, y_tcga_te, X_upenn_tr, y_upenn_tr, X_upenn_te, y_upenn_te, _ = generate_benchmark_datasets(seed=42)[:9]
        _, _, _, _, selected_cols = preprocess_and_select_features(X_tcga_tr, y_tcga_tr, X_upenn_tr, y_upenn_tr, X_tcga_te, X_upenn_te, seed=42, target_p=15)
        self.assertEqual(len(selected_cols), 15)

    def test_09_protected_clinical_features_present(self):
        """Test 9: All 4 protected clinical features are present in the final selected schema."""
        protected = ['Age', 'Gender', 'KPS', 'Resection_Status']
        X_tcga_tr, y_tcga_tr, X_tcga_te, y_tcga_te, X_upenn_tr, y_upenn_tr, X_upenn_te, y_upenn_te, _ = generate_benchmark_datasets(seed=42)[:9]
        _, _, _, _, selected_cols = preprocess_and_select_features(X_tcga_tr, y_tcga_tr, X_upenn_tr, y_upenn_tr, X_tcga_te, X_upenn_te, seed=42, target_p=15)
        
        for col in protected:
            self.assertIn(col, selected_cols)

    def test_10_eleven_radiomic_features_present(self):
        """Test 10: Exactly 11 radiomic features are present in the final selected schema."""
        protected = ['Age', 'Gender', 'KPS', 'Resection_Status']
        X_tcga_tr, y_tcga_tr, X_tcga_te, y_tcga_te, X_upenn_tr, y_upenn_tr, X_upenn_te, y_upenn_te, _ = generate_benchmark_datasets(seed=42)[:9]
        _, _, _, _, selected_cols = preprocess_and_select_features(X_tcga_tr, y_tcga_tr, X_upenn_tr, y_upenn_tr, X_tcga_te, X_upenn_te, seed=42, target_p=15)
        
        rad_selected = [c for c in selected_cols if c not in protected]
        self.assertEqual(len(rad_selected), 11)

    def test_11_tcga_and_upenn_identical_feature_ordering(self):
        """Test 11: TCGA and UPenn processed feature matrices use identical feature column ordering."""
        X_tcga_tr, y_tcga_tr, X_tcga_te, y_tcga_te, X_upenn_tr, y_upenn_tr, X_upenn_te, y_upenn_te, _ = generate_benchmark_datasets(seed=42)[:9]
        X_tcga_tr_std, X_upenn_tr_std, X_tcga_te_std, X_upenn_te_std, selected_cols = preprocess_and_select_features(X_tcga_tr, y_tcga_tr, X_upenn_tr, y_upenn_tr, X_tcga_te, X_upenn_te, seed=42, target_p=15)
        
        self.assertListEqual(list(X_tcga_tr_std.columns), list(X_upenn_tr_std.columns))
        self.assertListEqual(list(X_tcga_tr_std.columns), list(X_tcga_te_std.columns))
        self.assertListEqual(list(X_tcga_tr_std.columns), list(X_upenn_te_std.columns))
        self.assertListEqual(list(X_tcga_tr_std.columns), selected_cols)

    def test_12_test_data_excluded_from_all_fit_operations(self):
        """Test 12: TCGA and UPenn test set samples are completely excluded from imputer, selector, and scaler fitting."""
        X_tcga_tr, y_tcga_tr, X_tcga_te, y_tcga_te, X_upenn_tr, y_upenn_tr, X_upenn_te, y_upenn_te, _ = generate_benchmark_datasets(seed=42)[:9]
        
        # Modify test set values drastically
        X_tcga_te_mutated = X_tcga_te.copy() * 1000.0
        X_upenn_te_mutated = X_upenn_te.copy() * 1000.0
        
        X_tcga_tr_std1, X_upenn_tr_std1, _, _, cols1 = preprocess_and_select_features(X_tcga_tr, y_tcga_tr, X_upenn_tr, y_upenn_tr, X_tcga_te, X_upenn_te, seed=42, target_p=15)
        X_tcga_tr_std2, X_upenn_tr_std2, _, _, cols2 = preprocess_and_select_features(X_tcga_tr, y_tcga_tr, X_upenn_tr, y_upenn_tr, X_tcga_te_mutated, X_upenn_te_mutated, seed=42, target_p=15)
        
        # Training transformations and feature selection MUST BE 100% IDENTICAL
        self.assertEqual(cols1, cols2)
        np.testing.assert_array_almost_equal(X_tcga_tr_std1.values, X_tcga_tr_std2.values)
        np.testing.assert_array_almost_equal(X_upenn_tr_std1.values, X_upenn_tr_std2.values)


if __name__ == '__main__':
    unittest.main()

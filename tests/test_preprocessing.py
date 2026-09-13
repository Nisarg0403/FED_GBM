import sys, os
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

"""
Unit tests for FED-GBM v3 Leakage-Free Preprocessing & Feature Selection Pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from fed_gbm.preprocessing import (
    LeakageFreeImputer,
    VarianceFilter,
    PearsonCorrelationFilter,
    XGBoostFeatureSelector,
    FederatedSufficientStatsScaler,
    LeakageFreeCVPipeline
)


def create_synthetic_cohort(n_samples=77, n_radiomics=50, seed=42):
    """
    Helper to generate synthetic dataset mimicking TCGA-GBM cohort structure.
    """
    np.random.seed(seed)
    
    clinical_data = {
        'Age': np.random.normal(60, 12, size=n_samples).astype(float),
        'Gender': np.random.choice([0, 1], size=n_samples),
        'KPS': np.random.choice([60, 70, 80, 90, 100], size=n_samples).astype(float),
        'Resection_Status': np.random.choice([0, 1], size=n_samples)
    }
    
    # Introduce some NaN values in continuous features
    clinical_data['Age'][::10] = np.nan
    clinical_data['KPS'][::15] = np.nan
    
    df_clinical = pd.DataFrame(clinical_data)
    
    # Radiomics features (some low variance, some highly correlated)
    radiomics_dict = {}
    base_signal = np.random.normal(0, 1, size=n_samples)
    
    for i in range(n_radiomics):
        if i == 0:
            # Low variance
            radiomics_dict[f'rad_feature_{i}'] = np.ones(n_samples) * 5.0
            radiomics_dict[f'rad_feature_{i}'][0] += 0.001
        elif i == 1:
            # High correlation with rad_feature_2
            radiomics_dict[f'rad_feature_{i}'] = base_signal + np.random.normal(0, 0.01, size=n_samples)
        elif i == 2:
            radiomics_dict[f'rad_feature_{i}'] = base_signal
        else:
            radiomics_dict[f'rad_feature_{i}'] = np.random.normal(0, 1, size=n_samples)
            
    df_rad = pd.DataFrame(radiomics_dict)
    
    X = pd.concat([df_clinical, df_rad], axis=1)
    # Binary mortality label 12m
    y = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6]), name='label_12m')
    
    return X, y


def test_leakage_free_imputer_isolation():
    """Verify that imputer fit on train fold is completely isolated from test fold."""
    X_train, _ = create_synthetic_cohort(n_samples=50, seed=42)
    X_test, _ = create_synthetic_cohort(n_samples=20, seed=999)
    
    imputer = LeakageFreeImputer(categorical_cols=['Gender', 'Resection_Status'])
    imputer.fit(X_train)
    
    # Train median for Age
    train_age_median = X_train['Age'].median()
    assert imputer.fill_values_['Age'] == train_age_median
    
    # Transform test set using TRAIN median
    X_test_transformed = imputer.transform(X_test)
    assert not X_test_transformed['Age'].isnull().any()
    
    # Modify test set NaNs and verify train median remains unchanged
    X_test_modified = X_test.copy()
    X_test_modified['Age'] = 999.0
    imputer_recheck = imputer.transform(X_test_modified)
    assert imputer.fill_values_['Age'] == train_age_median


def test_variance_filter_isolation():
    """Verify low-variance feature filtering fit strictly on train fold."""
    X_train, _ = create_synthetic_cohort(n_samples=50, seed=42)
    
    # rad_feature_0 has ~0 variance in train
    vfilter = VarianceFilter(threshold=0.01, protected_cols=['Age', 'Gender', 'KPS', 'Resection_Status'])
    vfilter.fit(X_train)
    
    X_train_filtered = vfilter.transform(X_train)
    assert 'rad_feature_0' not in X_train_filtered.columns
    assert 'Age' in X_train_filtered.columns  # protected


def test_pearson_correlation_filter():
    """Verify Pearson correlation filter removes collinear features."""
    X_train, _ = create_synthetic_cohort(n_samples=50, seed=42)
    
    # Impute first to allow correlation matrix computation
    imputer = LeakageFreeImputer()
    X_train_imp = imputer.fit_transform(X_train)
    
    cfilter = PearsonCorrelationFilter(threshold=0.85, protected_cols=['Age', 'Gender', 'KPS', 'Resection_Status'])
    cfilter.fit(X_train_imp)
    
    X_train_filtered = cfilter.transform(X_train_imp)
    
    # rad_feature_1 and rad_feature_2 were created collinear (|r| > 0.99)
    # One of them should be removed
    retained = X_train_filtered.columns.tolist()
    assert not ('rad_feature_1' in retained and 'rad_feature_2' in retained)


def test_xgboost_feature_selector_shape():
    """Verify XGBoost feature selector selects exact target count p and preserves protected cols."""
    X_train, y_train = create_synthetic_cohort(n_samples=60, seed=42)
    
    imputer = LeakageFreeImputer()
    X_train_imp = imputer.fit_transform(X_train)
    
    protected = ['Age', 'Gender', 'KPS', 'Resection_Status']
    selector = XGBoostFeatureSelector(n_features_to_select=15, protected_cols=protected, random_state=42)
    selector.fit(X_train_imp, y_train)
    
    X_train_selected = selector.transform(X_train_imp)
    assert X_train_selected.shape[1] == 15
    for col in protected:
        assert col in X_train_selected.columns


def test_federated_sufficient_stats_scaler():
    """Verify Federated Sufficient Statistics Scaler matches centralized StandardScaler exactly."""
    np.random.seed(42)
    X1 = pd.DataFrame(np.random.normal(10, 2, size=(30, 5)), columns=[f'f{i}' for i in range(5)])
    X2 = pd.DataFrame(np.random.normal(12, 3, size=(40, 5)), columns=[f'f{i}' for i in range(5)])
    
    # Client local stats
    stats1 = FederatedSufficientStatsScaler.compute_local_stats(X1)
    stats2 = FederatedSufficientStatsScaler.compute_local_stats(X2)
    
    fed_scaler = FederatedSufficientStatsScaler()
    fed_scaler.aggregate_global_stats([stats1, stats2])
    
    # Centralized equivalent
    X_pooled = pd.concat([X1, X2], axis=0)
    central_scaler = StandardScaler()
    central_scaler.fit(X_pooled)
    
    # Check mean and scale agreement
    np.testing.assert_allclose(fed_scaler.mean_.values, central_scaler.mean_, rtol=1e-5)
    np.testing.assert_allclose(fed_scaler.scale_.values, central_scaler.scale_, rtol=1e-5)
    
    # Transform test set using fed_scaler
    X_test = pd.DataFrame(np.random.normal(11, 2.5, size=(10, 5)), columns=[f'f{i}' for i in range(5)])
    X_test_fed = fed_scaler.transform(X_test)
    X_test_central = pd.DataFrame(central_scaler.transform(X_test), columns=X_test.columns)
    
    np.testing.assert_allclose(X_test_fed.values, X_test_central.values, rtol=1e-5)


def test_leakage_free_cv_pipeline():
    """Test full repeated 5-fold CV pipeline on synthetic dataset."""
    X_train, y_train = create_synthetic_cohort(n_samples=77, seed=42)
    
    pipeline = LeakageFreeCVPipeline(
        n_splits=5,
        seeds=[42, 101],
        protected_cols=['Age', 'Gender', 'KPS', 'Resection_Status']
    )
    
    res = pipeline.run_cv_feature_selection(X_train, y_train, target_feature_counts=[10, 15])
    assert res['total_runs'] == 2 * 5 * 2  # 2 seeds * 5 folds * 2 target_p values
    
    # Lock final feature schema
    locked_cols, meta = pipeline.lock_final_feature_schema(X_train, y_train, target_p=15)
    assert len(locked_cols) == 15
    assert meta['target_p'] == 15
    for col in ['Age', 'Gender', 'KPS', 'Resection_Status']:
        assert col in locked_cols

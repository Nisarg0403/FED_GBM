"""
Leakage-Free Preprocessing & Feature Selection Pipeline for FED-GBM v3.
Ensures 100% isolation of validation folds and locked test partitions.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier


class LeakageFreeImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values strictly using training set statistics.
    - Continuous features: Median
    - Categorical features: Mode (or default 0)
    """
    def __init__(self, continuous_cols: Optional[List[str]] = None, categorical_cols: Optional[List[str]] = None):
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.fill_values_ = {}

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X_df = pd.DataFrame(X)
        cols = X_df.columns.tolist()
        
        cont_cols = self.continuous_cols if self.continuous_cols is not None else cols
        cat_cols = self.categorical_cols if self.categorical_cols is not None else []
        
        self.fill_values_ = {}
        for col in cols:
            if col in cat_cols:
                mode_vals = X_df[col].mode()
                self.fill_values_[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else 0
            else:
                median_val = X_df[col].median()
                self.fill_values_[col] = median_val if pd.notna(median_val) else 0.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        for col, fill_val in self.fill_values_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].fillna(fill_val)
        return X_df


class VarianceFilter(BaseEstimator, TransformerMixin):
    """
    Removes low-variance features (var <= threshold) fit strictly on training fold.
    """
    def __init__(self, threshold: float = 0.01, protected_cols: Optional[List[str]] = None):
        self.threshold = threshold
        self.protected_cols = protected_cols if protected_cols is not None else []
        self.selected_cols_ = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X_df = pd.DataFrame(X)
        variances = X_df.var(axis=0)
        self.selected_cols_ = [
            col for col in X_df.columns 
            if variances[col] > self.threshold or col in self.protected_cols
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X)
        return X_df[self.selected_cols_].copy()


class PearsonCorrelationFilter(BaseEstimator, TransformerMixin):
    """
    Removes highly collinear features (|r| >= threshold) fit strictly on training fold.
    Protects specified clinical features from being dropped.
    """
    def __init__(self, threshold: float = 0.85, protected_cols: Optional[List[str]] = None):
        self.threshold = threshold
        self.protected_cols = protected_cols if protected_cols is not None else []
        self.selected_cols_ = []

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X_df = pd.DataFrame(X)
        corr_matrix = X_df.corr(method='pearson').abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = set()
        for col in upper_tri.columns:
            if col in self.protected_cols:
                continue
            # If collinear with any other column
            high_corr_cols = upper_tri.index[upper_tri[col] >= self.threshold].tolist()
            if len(high_corr_cols) > 0:
                to_drop.add(col)
                
        self.selected_cols_ = [col for col in X_df.columns if col not in to_drop]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X)
        return X_df[self.selected_cols_].copy()


class XGBoostFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Fits XGBoost gain feature importance ranking on training fold
    and selects the top k features. Always preserves protected clinical features.
    """
    def __init__(self, n_features_to_select: int = 15, protected_cols: Optional[List[str]] = None, random_state: int = 42):
        self.n_features_to_select = n_features_to_select
        self.protected_cols = protected_cols if protected_cols is not None else []
        self.random_state = random_state
        self.selected_cols_ = []

    def fit(self, X: pd.DataFrame, y: pd.Series):
        X_df = pd.DataFrame(X)
        y_series = pd.Series(y)
        
        clf = XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            random_state=self.random_state,
            eval_metric='logloss'
        )
        clf.fit(X_df, y_series)
        
        importances = clf.feature_importances_
        feature_names = X_df.columns.tolist()
        
        # Build ranking dict
        importance_dict = dict(zip(feature_names, importances))
        
        # Separate protected vs radiomics candidate features
        protected = [col for col in feature_names if col in self.protected_cols]
        candidates = [col for col in feature_names if col not in self.protected_cols]
        
        # Sort candidate features by importance descending
        candidates_sorted = sorted(candidates, key=lambda c: importance_dict.get(c, 0.0), reverse=True)
        
        # Select top k radiomics features to reach n_features_to_select total
        k_rad = max(0, self.n_features_to_select - len(protected))
        selected_radiomics = candidates_sorted[:k_rad]
        
        self.selected_cols_ = protected + selected_radiomics
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X)
        return X_df[self.selected_cols_].copy()


class FederatedSufficientStatsScaler:
    """
    Computes local scalar summary statistics S_k = sum(X_k), SS_k = sum(X_k^2)
    on local client training data and aggregates them centrally to derive
    global mean and standard deviation without centralizing raw patient samples.
    """
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        self.mean_ = None
        self.scale_ = None
        self.feature_names_ = None

    @staticmethod
    def compute_local_stats(X: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes local scalar totals S_k and sum of squares SS_k on local train samples.
        """
        X_df = pd.DataFrame(X)
        n_k = len(X_df)
        S_k = X_df.sum(axis=0).to_dict()
        SS_k = (X_df ** 2).sum(axis=0).to_dict()
        return {
            'n_k': n_k,
            'S_k': S_k,
            'SS_k': SS_k,
            'features': X_df.columns.tolist()
        }

    def aggregate_global_stats(self, client_stats_list: List[Dict[str, Any]]):
        """
        Aggregates scalar statistics from all clients to compute global mean & std.
        """
        if not client_stats_list:
            raise ValueError("client_stats_list cannot be empty")
            
        features = client_stats_list[0]['features']
        self.feature_names_ = features
        
        N_total = sum(item['n_k'] for item in client_stats_list)
        if N_total == 0:
            raise ValueError("Total sample count across clients is 0")
            
        S_series = sum(pd.Series(item['S_k']) for item in client_stats_list)
        SS_series = sum(pd.Series(item['SS_k']) for item in client_stats_list)
        
        mean_series = S_series / N_total
        var_series = (SS_series / N_total - mean_series ** 2).clip(lower=0.0)
        std_series = np.sqrt(var_series)
        
        self.mean_ = mean_series
        self.scale_ = std_series.where(std_series > self.epsilon, 1.0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Standardizes input DataFrame using aggregated global statistics.
        """
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Must call aggregate_global_stats before transform")
            
        X_df = pd.DataFrame(X).copy()
        for f in X_df.columns:
            if f in self.mean_ and f in self.scale_:
                X_df[f] = (X_df[f] - self.mean_[f]) / self.scale_[f]
        return X_df


class LeakageFreeCVPipeline:
    """
    Executes Leakage-Free Repeated 5-Fold Stratified Cross-Validation on training data.
    Ensures feature selection and preprocessing are fit EXCLUSIVELY on training folds.
    """
    def __init__(
        self,
        n_splits: int = 5,
        seeds: List[int] = [42, 101, 202, 303, 404, 505, 606, 707, 808, 909],
        protected_cols: List[str] = ['Age', 'Gender', 'KPS', 'Resection_Status'],
        var_threshold: float = 0.01,
        corr_threshold: float = 0.85
    ):
        self.n_splits = n_splits
        self.seeds = seeds
        self.protected_cols = protected_cols
        self.var_threshold = var_threshold
        self.corr_threshold = corr_threshold

    def run_cv_feature_selection(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target_feature_counts: List[int] = [10, 15, 20, 25, 30]
    ) -> Dict[str, Any]:
        """
        Runs repeated 5-fold CV to evaluate feature selection robustness and isolate folds.
        """
        results = []
        
        for seed in self.seeds:
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=seed)
            fold_idx = 0
            
            for train_idx, val_idx in skf.split(X_train, y_train):
                fold_idx += 1
                X_fold_tr = X_train.iloc[train_idx].copy()
                y_fold_tr = y_train.iloc[train_idx].copy()
                X_fold_val = X_train.iloc[val_idx].copy()
                y_fold_val = y_train.iloc[val_idx].copy()
                
                # 1. Fit Imputer strictly on fold-train
                imputer = LeakageFreeImputer(categorical_cols=['Gender', 'Resection_Status'])
                imputer.fit(X_fold_tr)
                X_fold_tr_imp = imputer.transform(X_fold_tr)
                X_fold_val_imp = imputer.transform(X_fold_val)
                
                # 2. Fit Variance Filter strictly on fold-train
                var_filter = VarianceFilter(threshold=self.var_threshold, protected_cols=self.protected_cols)
                var_filter.fit(X_fold_tr_imp)
                X_fold_tr_var = var_filter.transform(X_fold_tr_imp)
                
                # 3. Fit Correlation Filter strictly on fold-train
                corr_filter = PearsonCorrelationFilter(threshold=self.corr_threshold, protected_cols=self.protected_cols)
                corr_filter.fit(X_fold_tr_var)
                X_fold_tr_corr = corr_filter.transform(X_fold_tr_var)
                
                # 4. Fit XGBoost Feature Ranking strictly on fold-train
                for p_total in target_feature_counts:
                    xgb_selector = XGBoostFeatureSelector(
                        n_features_to_select=p_total,
                        protected_cols=self.protected_cols,
                        random_state=seed
                    )
                    xgb_selector.fit(X_fold_tr_corr, y_fold_tr)
                    selected_cols = xgb_selector.selected_cols_
                    
                    results.append({
                        'seed': seed,
                        'fold': fold_idx,
                        'target_p': p_total,
                        'actual_p': len(selected_cols),
                        'selected_cols': selected_cols
                    })
                    
        return {
            'n_seeds': len(self.seeds),
            'n_splits': self.n_splits,
            'total_runs': len(results),
            'fold_results': results
        }

    def lock_final_feature_schema(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        target_p: int = 15
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Fits final feature selection pipeline strictly on the ENTIRE development training cohort (N=77)
        and returns the locked feature list.
        """
        # 1. Imputation
        imputer = LeakageFreeImputer(categorical_cols=['Gender', 'Resection_Status'])
        imputer.fit(X_train)
        X_tr_imp = imputer.transform(X_train)
        
        # 2. Variance Filter
        var_filter = VarianceFilter(threshold=self.var_threshold, protected_cols=self.protected_cols)
        var_filter.fit(X_tr_imp)
        X_tr_var = var_filter.transform(X_tr_imp)
        
        # 3. Correlation Filter
        corr_filter = PearsonCorrelationFilter(threshold=self.corr_threshold, protected_cols=self.protected_cols)
        corr_filter.fit(X_tr_var)
        X_tr_corr = corr_filter.transform(X_tr_var)
        
        # 4. XGBoost Feature Selector
        xgb_selector = XGBoostFeatureSelector(
            n_features_to_select=target_p,
            protected_cols=self.protected_cols,
            random_state=42
        )
        xgb_selector.fit(X_tr_corr, y_train)
        locked_features = xgb_selector.selected_cols_
        
        pipeline_meta = {
            'target_p': target_p,
            'locked_feature_count': len(locked_features),
            'protected_cols': self.protected_cols,
            'radiomics_cols': [c for c in locked_features if c not in self.protected_cols],
            'locked_features': locked_features
        }
        
        return locked_features, pipeline_meta

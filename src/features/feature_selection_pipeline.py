"""Feature selection pipeline with mutual information and RF-based methods."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from datetime import datetime, timedelta
import warnings
from loguru import logger
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum

# Feature selection imports
from sklearn.feature_selection import (
    mutual_info_regression, SelectKBest, RFE, RFECV,
    SelectPercentile, VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from scipy.stats import spearmanr, pearsonr
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.spatial.distance as ssd

from ..config.settings import config


class FeatureSelectionMethod(Enum):
    """Feature selection methods."""
    MUTUAL_INFORMATION = "mutual_information"
    RANDOM_FOREST = "random_forest"
    CORRELATION = "correlation"
    VARIANCE = "variance"
    RECURSIVE_ELIMINATION = "recursive_elimination"
    STATISTICAL = "statistical"
    HYBRID = "hybrid"


class FeatureImportanceMetric(Enum):
    """Feature importance metrics."""
    MUTUAL_INFO = "mutual_info"
    RF_IMPORTANCE = "rf_importance"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    CORRELATION_ABS = "correlation_abs"
    VARIANCE_SCORE = "variance_score"
    STABILITY_SCORE = "stability_score"


@dataclass
class FeatureScore:
    """Feature importance score with metadata."""
    feature_name: str
    score: float
    metric_type: FeatureImportanceMetric
    rank: int
    stability: float
    regime_consistency: float
    redundancy_score: float


@dataclass
class SelectionResult:
    """Feature selection results."""
    selected_features: List[str]
    feature_scores: List[FeatureScore]
    selection_method: FeatureSelectionMethod
    selection_metadata: Dict[str, Any]
    performance_improvement: float
    feature_reduction_ratio: float


class FeatureSelectionPipeline:
    """
    Advanced feature selection pipeline with multiple methods.
    
    Features:
    - Mutual information-based selection
    - Random Forest importance ranking
    - Correlation-based redundancy removal
    - Regime-aware feature selection
    - Stability analysis across market conditions
    - Hybrid selection combining multiple methods
    - Performance validation
    """
    
    def __init__(
        self,
        target_feature_count: int = 50,
        min_feature_count: int = 20,
        max_feature_count: int = 100,
        stability_threshold: float = 0.7,
        redundancy_threshold: float = 0.9,
        mutual_info_percentile: float = 0.8,
        rf_importance_threshold: float = 0.01,
        enable_regime_awareness: bool = True,
        cross_validation_folds: int = 5
    ):
        """
        Initialize feature selection pipeline.
        
        Args:
            target_feature_count: Target number of features to select
            min_feature_count: Minimum features to retain
            max_feature_count: Maximum features to consider
            stability_threshold: Minimum stability score for feature retention
            redundancy_threshold: Correlation threshold for redundancy removal
            mutual_info_percentile: Percentile for mutual information selection
            rf_importance_threshold: Minimum RF importance for feature retention
            enable_regime_awareness: Enable regime-aware feature selection
            cross_validation_folds: Number of CV folds for validation
        """
        self.target_feature_count = target_feature_count
        self.min_feature_count = min_feature_count
        self.max_feature_count = max_feature_count
        self.stability_threshold = stability_threshold
        self.redundancy_threshold = redundancy_threshold
        self.mutual_info_percentile = mutual_info_percentile
        self.rf_importance_threshold = rf_importance_threshold
        self.enable_regime_awareness = enable_regime_awareness
        self.cross_validation_folds = cross_validation_folds
        
        # Selection history and caching
        self.selection_history = []
        self.feature_importance_cache = {}
        self.stability_cache = {}
        
        # Regime-specific feature performance
        self.regime_feature_performance = defaultdict(lambda: defaultdict(list))
        
        # Feature metadata tracking
        self.feature_metadata = {}
        
        logger.info(f"FeatureSelectionPipeline initialized with target {target_feature_count} features")
    
    def select_features(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: List[str],
        method: FeatureSelectionMethod = FeatureSelectionMethod.HYBRID,
        regime_labels: Optional[np.ndarray] = None,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> SelectionResult:
        """
        Select optimal features using specified method.
        
        Args:
            features: Feature matrix (samples x features)
            targets: Target values
            feature_names: Names of features
            method: Selection method to use
            regime_labels: Optional regime labels for regime-aware selection
            validation_data: Optional validation data (features, targets)
            
        Returns:
            Selection results with chosen features and metadata
        """
        try:
            if len(feature_names) != features.shape[1]:
                raise ValueError("Feature names length must match features matrix columns")
            
            if len(features) != len(targets):
                raise ValueError("Features and targets must have same number of samples")
            
            logger.info(f"Starting feature selection with {len(feature_names)} features using {method.value}")
            
            # Preprocess features
            processed_features, scaler = self._preprocess_features(features)
            
            # Apply variance threshold first
            variance_selector = VarianceThreshold(threshold=0.0)  # Remove zero variance
            processed_features = variance_selector.fit_transform(processed_features)
            valid_feature_indices = variance_selector.get_support()
            valid_feature_names = [name for i, name in enumerate(feature_names) if valid_feature_indices[i]]
            
            if len(valid_feature_names) < self.min_feature_count:
                logger.warning(f"Only {len(valid_feature_names)} features passed variance threshold")
                return self._create_fallback_result(valid_feature_names, method)
            
            # Apply selected method
            if method == FeatureSelectionMethod.MUTUAL_INFORMATION:
                result = self._mutual_information_selection(
                    processed_features, targets, valid_feature_names, regime_labels
                )
            elif method == FeatureSelectionMethod.RANDOM_FOREST:
                result = self._random_forest_selection(
                    processed_features, targets, valid_feature_names, regime_labels
                )
            elif method == FeatureSelectionMethod.CORRELATION:
                result = self._correlation_based_selection(
                    processed_features, targets, valid_feature_names
                )
            elif method == FeatureSelectionMethod.RECURSIVE_ELIMINATION:
                result = self._recursive_elimination_selection(
                    processed_features, targets, valid_feature_names
                )
            elif method == FeatureSelectionMethod.STATISTICAL:
                result = self._statistical_selection(
                    processed_features, targets, valid_feature_names
                )
            elif method == FeatureSelectionMethod.HYBRID:
                result = self._hybrid_selection(
                    processed_features, targets, valid_feature_names, regime_labels
                )
            else:
                logger.warning(f"Unknown method {method}, using hybrid selection")
                result = self._hybrid_selection(
                    processed_features, targets, valid_feature_names, regime_labels
                )
            
            # Validate selection if validation data provided
            if validation_data is not None:
                val_features, val_targets = validation_data
                result.performance_improvement = self._validate_selection(
                    result.selected_features, features, targets, val_features, val_targets,
                    feature_names
                )
            
            # Update history
            self.selection_history.append(result)
            
            logger.info(f"Selected {len(result.selected_features)} features with {method.value}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            return self._create_fallback_result(feature_names[:self.target_feature_count], method)
    
    def _preprocess_features(self, features: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
        """Preprocess features for selection algorithms."""
        try:
            # Use RobustScaler to handle outliers
            scaler = RobustScaler()
            processed_features = scaler.fit_transform(features)
            
            # Handle any remaining infinite or NaN values
            processed_features = np.nan_to_num(processed_features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            return processed_features, scaler
            
        except Exception as e:
            logger.warning(f"Error preprocessing features: {e}")
            return features, StandardScaler()
    
    def _mutual_information_selection(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: List[str],
        regime_labels: Optional[np.ndarray] = None
    ) -> SelectionResult:
        """Select features using mutual information."""
        try:
            # Calculate mutual information scores
            mi_scores = mutual_info_regression(
                features, targets, 
                discrete_features=False,
                n_neighbors=min(5, len(features) // 10),
                random_state=42
            )
            
            # Handle edge cases
            mi_scores = np.nan_to_num(mi_scores, nan=0.0)
            
            # Calculate percentile threshold
            threshold = np.percentile(mi_scores[mi_scores > 0], 
                                    self.mutual_info_percentile * 100) if np.any(mi_scores > 0) else 0.0
            
            # Create feature scores
            feature_scores = []
            for i, (name, score) in enumerate(zip(feature_names, mi_scores)):
                stability = self._calculate_feature_stability(features[:, i], regime_labels)
                redundancy = 0.0  # Will be calculated later
                
                feature_score = FeatureScore(
                    feature_name=name,
                    score=score,
                    metric_type=FeatureImportanceMetric.MUTUAL_INFO,
                    rank=0,  # Will be set after sorting
                    stability=stability,
                    regime_consistency=self._calculate_regime_consistency(features[:, i], regime_labels),
                    redundancy_score=redundancy
                )
                feature_scores.append(feature_score)
            
            # Sort by score
            feature_scores.sort(key=lambda x: x.score, reverse=True)
            for i, fs in enumerate(feature_scores):
                fs.rank = i + 1
            
            # Select top features above threshold
            selected_scores = [fs for fs in feature_scores if fs.score >= threshold]
            
            # Limit to target count
            if len(selected_scores) > self.target_feature_count:
                selected_scores = selected_scores[:self.target_feature_count]
            elif len(selected_scores) < self.min_feature_count:
                selected_scores = feature_scores[:self.min_feature_count]
            
            # Remove redundant features
            selected_features = self._remove_redundant_features(
                features, [fs.feature_name for fs in selected_scores], feature_names
            )
            
            # Update redundancy scores
            for fs in selected_scores:
                if fs.feature_name in selected_features:
                    fs.redundancy_score = self._calculate_redundancy_score(
                        features, fs.feature_name, selected_features, feature_names
                    )
            
            return SelectionResult(
                selected_features=selected_features,
                feature_scores=selected_scores,
                selection_method=FeatureSelectionMethod.MUTUAL_INFORMATION,
                selection_metadata={
                    'threshold': threshold,
                    'mean_mi_score': np.mean(mi_scores),
                    'std_mi_score': np.std(mi_scores)
                },
                performance_improvement=0.0,
                feature_reduction_ratio=len(selected_features) / len(feature_names)
            )
            
        except Exception as e:
            logger.error(f"Error in mutual information selection: {e}")
            return self._create_fallback_result(feature_names[:self.target_feature_count], 
                                              FeatureSelectionMethod.MUTUAL_INFORMATION)
    
    def _random_forest_selection(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: List[str],
        regime_labels: Optional[np.ndarray] = None
    ) -> SelectionResult:
        """Select features using Random Forest importance."""
        try:
            # Create Random Forest model
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # Fit model
            rf_model.fit(features, targets)
            
            # Get feature importances
            importances = rf_model.feature_importances_
            
            # Calculate permutation importance for stability
            perm_importances = self._calculate_permutation_importance(
                rf_model, features, targets
            )
            
            # Create feature scores
            feature_scores = []
            for i, (name, importance, perm_imp) in enumerate(zip(feature_names, importances, perm_importances)):
                stability = self._calculate_feature_stability(features[:, i], regime_labels)
                
                feature_score = FeatureScore(
                    feature_name=name,
                    score=importance,
                    metric_type=FeatureImportanceMetric.RF_IMPORTANCE,
                    rank=0,
                    stability=stability,
                    regime_consistency=self._calculate_regime_consistency(features[:, i], regime_labels),
                    redundancy_score=0.0
                )
                feature_scores.append(feature_score)
            
            # Sort by importance
            feature_scores.sort(key=lambda x: x.score, reverse=True)
            for i, fs in enumerate(feature_scores):
                fs.rank = i + 1
            
            # Select features above threshold
            selected_scores = [fs for fs in feature_scores if fs.score >= self.rf_importance_threshold]
            
            if len(selected_scores) > self.target_feature_count:
                selected_scores = selected_scores[:self.target_feature_count]
            elif len(selected_scores) < self.min_feature_count:
                selected_scores = feature_scores[:self.min_feature_count]
            
            # Remove redundant features
            selected_features = self._remove_redundant_features(
                features, [fs.feature_name for fs in selected_scores], feature_names
            )
            
            return SelectionResult(
                selected_features=selected_features,
                feature_scores=selected_scores,
                selection_method=FeatureSelectionMethod.RANDOM_FOREST,
                selection_metadata={
                    'importance_threshold': self.rf_importance_threshold,
                    'mean_importance': np.mean(importances),
                    'std_importance': np.std(importances),
                    'rf_score': rf_model.score(features, targets)
                },
                performance_improvement=0.0,
                feature_reduction_ratio=len(selected_features) / len(feature_names)
            )
            
        except Exception as e:
            logger.error(f"Error in Random Forest selection: {e}")
            return self._create_fallback_result(feature_names[:self.target_feature_count],
                                              FeatureSelectionMethod.RANDOM_FOREST)
    
    def _correlation_based_selection(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: List[str]
    ) -> SelectionResult:
        """Select features based on correlation with target and redundancy removal."""
        try:
            # Calculate correlation with target
            correlations = []
            for i in range(features.shape[1]):
                try:
                    corr, _ = pearsonr(features[:, i], targets)
                    correlations.append(abs(corr) if not np.isnan(corr) else 0.0)
                except:
                    correlations.append(0.0)
            
            correlations = np.array(correlations)
            
            # Create feature scores
            feature_scores = []
            for i, (name, corr) in enumerate(zip(feature_names, correlations)):
                feature_score = FeatureScore(
                    feature_name=name,
                    score=corr,
                    metric_type=FeatureImportanceMetric.CORRELATION_ABS,
                    rank=0,
                    stability=1.0,  # Correlation is relatively stable
                    regime_consistency=1.0,
                    redundancy_score=0.0
                )
                feature_scores.append(feature_score)
            
            # Sort by correlation
            feature_scores.sort(key=lambda x: x.score, reverse=True)
            for i, fs in enumerate(feature_scores):
                fs.rank = i + 1
            
            # Select top features
            selected_scores = feature_scores[:min(self.target_feature_count * 2, len(feature_scores))]
            
            # Remove redundant features based on feature-feature correlations
            selected_features = self._remove_redundant_features(
                features, [fs.feature_name for fs in selected_scores], feature_names
            )
            
            # Limit to target count
            if len(selected_features) > self.target_feature_count:
                selected_features = selected_features[:self.target_feature_count]
            
            return SelectionResult(
                selected_features=selected_features,
                feature_scores=[fs for fs in selected_scores if fs.feature_name in selected_features],
                selection_method=FeatureSelectionMethod.CORRELATION,
                selection_metadata={
                    'mean_correlation': np.mean(correlations),
                    'max_correlation': np.max(correlations),
                    'redundancy_threshold': self.redundancy_threshold
                },
                performance_improvement=0.0,
                feature_reduction_ratio=len(selected_features) / len(feature_names)
            )
            
        except Exception as e:
            logger.error(f"Error in correlation-based selection: {e}")
            return self._create_fallback_result(feature_names[:self.target_feature_count],
                                              FeatureSelectionMethod.CORRELATION)
    
    def _recursive_elimination_selection(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: List[str]
    ) -> SelectionResult:
        """Select features using Recursive Feature Elimination."""
        try:
            # Use Random Forest as base estimator
            base_estimator = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            
            # Use RFECV for automatic feature count selection
            rfecv = RFECV(
                estimator=base_estimator,
                step=1,
                cv=min(self.cross_validation_folds, len(targets) // 20),
                scoring='r2',
                n_jobs=-1
            )
            
            # Fit RFECV
            rfecv.fit(features, targets)
            
            # Get selected features
            selected_mask = rfecv.support_
            selected_features = [name for i, name in enumerate(feature_names) if selected_mask[i]]
            
            # Create feature scores based on ranking
            feature_scores = []
            rankings = rfecv.ranking_
            
            for i, (name, rank) in enumerate(zip(feature_names, rankings)):
                # Convert rank to score (lower rank = higher score)
                score = 1.0 / rank if rank > 0 else 0.0
                
                feature_score = FeatureScore(
                    feature_name=name,
                    score=score,
                    metric_type=FeatureImportanceMetric.RF_IMPORTANCE,
                    rank=rank,
                    stability=1.0,
                    regime_consistency=1.0,
                    redundancy_score=0.0
                )
                feature_scores.append(feature_score)
            
            # Sort by rank
            feature_scores.sort(key=lambda x: x.rank)
            
            return SelectionResult(
                selected_features=selected_features,
                feature_scores=[fs for fs in feature_scores if fs.feature_name in selected_features],
                selection_method=FeatureSelectionMethod.RECURSIVE_ELIMINATION,
                selection_metadata={
                    'optimal_features': rfecv.n_features_,
                    'cv_scores': rfecv.cv_results_['mean_test_score'].tolist() if hasattr(rfecv, 'cv_results_') else [],
                    'grid_scores': rfecv.grid_scores_.tolist() if hasattr(rfecv, 'grid_scores_') else []
                },
                performance_improvement=0.0,
                feature_reduction_ratio=len(selected_features) / len(feature_names)
            )
            
        except Exception as e:
            logger.error(f"Error in recursive elimination selection: {e}")
            return self._create_fallback_result(feature_names[:self.target_feature_count],
                                              FeatureSelectionMethod.RECURSIVE_ELIMINATION)
    
    def _statistical_selection(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: List[str]
    ) -> SelectionResult:
        """Select features using statistical tests."""
        try:
            # Use SelectKBest with mutual information
            selector = SelectKBest(
                score_func=mutual_info_regression,
                k=min(self.target_feature_count, features.shape[1])
            )
            
            # Fit selector
            selected_features_matrix = selector.fit_transform(features, targets)
            
            # Get selected feature indices
            selected_indices = selector.get_support(indices=True)
            selected_features = [feature_names[i] for i in selected_indices]
            
            # Get scores
            scores = selector.scores_
            
            # Create feature scores
            feature_scores = []
            for i, (name, score) in enumerate(zip(feature_names, scores)):
                feature_score = FeatureScore(
                    feature_name=name,
                    score=score if not np.isnan(score) else 0.0,
                    metric_type=FeatureImportanceMetric.MUTUAL_INFO,
                    rank=0,
                    stability=1.0,
                    regime_consistency=1.0,
                    redundancy_score=0.0
                )
                feature_scores.append(feature_score)
            
            # Sort and rank
            feature_scores.sort(key=lambda x: x.score, reverse=True)
            for i, fs in enumerate(feature_scores):
                fs.rank = i + 1
            
            return SelectionResult(
                selected_features=selected_features,
                feature_scores=[fs for fs in feature_scores if fs.feature_name in selected_features],
                selection_method=FeatureSelectionMethod.STATISTICAL,
                selection_metadata={
                    'k_best': selector.k,
                    'mean_score': np.mean(scores[~np.isnan(scores)]) if np.any(~np.isnan(scores)) else 0.0
                },
                performance_improvement=0.0,
                feature_reduction_ratio=len(selected_features) / len(feature_names)
            )
            
        except Exception as e:
            logger.error(f"Error in statistical selection: {e}")
            return self._create_fallback_result(feature_names[:self.target_feature_count],
                                              FeatureSelectionMethod.STATISTICAL)
    
    def _hybrid_selection(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        feature_names: List[str],
        regime_labels: Optional[np.ndarray] = None
    ) -> SelectionResult:
        """Select features using hybrid approach combining multiple methods."""
        try:
            # Run multiple selection methods
            mi_result = self._mutual_information_selection(features, targets, feature_names, regime_labels)
            rf_result = self._random_forest_selection(features, targets, feature_names, regime_labels)
            corr_result = self._correlation_based_selection(features, targets, feature_names)
            
            # Combine results using voting
            feature_votes = defaultdict(int)
            feature_scores_combined = {}
            
            # Weight different methods
            method_weights = {
                FeatureSelectionMethod.MUTUAL_INFORMATION: 0.4,
                FeatureSelectionMethod.RANDOM_FOREST: 0.4,
                FeatureSelectionMethod.CORRELATION: 0.2
            }
            
            # Collect votes and scores
            for result, weight in [(mi_result, method_weights[FeatureSelectionMethod.MUTUAL_INFORMATION]),
                                  (rf_result, method_weights[FeatureSelectionMethod.RANDOM_FOREST]),
                                  (corr_result, method_weights[FeatureSelectionMethod.CORRELATION])]:
                
                for feature in result.selected_features:
                    feature_votes[feature] += weight
                    
                    # Combine scores
                    if feature not in feature_scores_combined:
                        feature_scores_combined[feature] = []
                    
                    # Find score for this feature
                    for fs in result.feature_scores:
                        if fs.feature_name == feature:
                            feature_scores_combined[feature].append((fs.score, weight))
                            break
            
            # Calculate combined scores
            combined_feature_scores = []
            for feature, vote_weight in feature_votes.items():
                if feature in feature_scores_combined:
                    # Weighted average of scores
                    weighted_score = sum(score * weight for score, weight in feature_scores_combined[feature])
                    total_weight = sum(weight for _, weight in feature_scores_combined[feature])
                    avg_score = weighted_score / total_weight if total_weight > 0 else 0.0
                    
                    # Calculate stability across methods
                    stability = self._calculate_cross_method_stability(feature, [mi_result, rf_result, corr_result])
                    
                    feature_score = FeatureScore(
                        feature_name=feature,
                        score=avg_score,
                        metric_type=FeatureImportanceMetric.MUTUAL_INFO,  # Combined
                        rank=0,
                        stability=stability,
                        regime_consistency=self._calculate_regime_consistency(
                            features[:, feature_names.index(feature)], regime_labels
                        ) if feature in feature_names else 1.0,
                        redundancy_score=0.0
                    )
                    combined_feature_scores.append((feature, vote_weight, feature_score))
            
            # Sort by vote weight and score
            combined_feature_scores.sort(key=lambda x: (x[1], x[2].score), reverse=True)
            
            # Select top features
            vote_threshold = max(method_weights.values()) * 1.5  # Features must appear in multiple methods
            selected_candidates = [
                (feature, fs) for feature, vote, fs in combined_feature_scores 
                if vote >= vote_threshold
            ]
            
            if len(selected_candidates) < self.min_feature_count:
                # Relax threshold
                vote_threshold = max(method_weights.values())
                selected_candidates = [
                    (feature, fs) for feature, vote, fs in combined_feature_scores 
                    if vote >= vote_threshold
                ][:self.target_feature_count]
            
            # Remove redundancy
            candidate_features = [feature for feature, _ in selected_candidates]
            selected_features = self._remove_redundant_features(
                features, candidate_features, feature_names
            )
            
            # Limit to target count
            if len(selected_features) > self.target_feature_count:
                selected_features = selected_features[:self.target_feature_count]
            
            # Final feature scores
            final_scores = [fs for feature, fs in selected_candidates if feature in selected_features]
            for i, fs in enumerate(final_scores):
                fs.rank = i + 1
            
            return SelectionResult(
                selected_features=selected_features,
                feature_scores=final_scores,
                selection_method=FeatureSelectionMethod.HYBRID,
                selection_metadata={
                    'vote_threshold': vote_threshold,
                    'method_weights': method_weights,
                    'mi_features': len(mi_result.selected_features),
                    'rf_features': len(rf_result.selected_features),
                    'corr_features': len(corr_result.selected_features)
                },
                performance_improvement=0.0,
                feature_reduction_ratio=len(selected_features) / len(feature_names)
            )
            
        except Exception as e:
            logger.error(f"Error in hybrid selection: {e}")
            return self._create_fallback_result(feature_names[:self.target_feature_count],
                                              FeatureSelectionMethod.HYBRID)
    
    def _remove_redundant_features(
        self,
        features: np.ndarray,
        candidate_features: List[str],
        all_feature_names: List[str]
    ) -> List[str]:
        """Remove redundant features based on correlation clustering."""
        try:
            if len(candidate_features) <= self.min_feature_count:
                return candidate_features
            
            # Get indices of candidate features
            candidate_indices = [all_feature_names.index(name) for name in candidate_features]
            candidate_feature_matrix = features[:, candidate_indices]
            
            # Calculate correlation matrix
            correlation_matrix = np.corrcoef(candidate_feature_matrix.T)
            correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
            
            # Use hierarchical clustering to group similar features
            distance_matrix = 1 - np.abs(correlation_matrix)
            
            # Handle case where all features are identical (distance = 0)
            if np.all(distance_matrix == 0):
                return candidate_features[:self.target_feature_count]
            
            # Perform hierarchical clustering
            try:
                linkage_matrix = linkage(ssd.squareform(distance_matrix), method='average')
                
                # Determine number of clusters based on redundancy threshold
                cluster_threshold = 1 - self.redundancy_threshold
                clusters = fcluster(linkage_matrix, cluster_threshold, criterion='distance')
                
                # Select one feature from each cluster (the one with highest score)
                selected_features = []
                feature_scores = {name: i for i, name in enumerate(candidate_features)}  # Use index as proxy
                
                for cluster_id in np.unique(clusters):
                    cluster_features = [candidate_features[i] for i in range(len(candidate_features)) 
                                      if clusters[i] == cluster_id]
                    
                    # Select feature with lowest index (highest original score)
                    best_feature = min(cluster_features, key=lambda x: feature_scores[x])
                    selected_features.append(best_feature)
                
                return selected_features
                
            except Exception as clustering_error:
                logger.warning(f"Clustering failed: {clustering_error}, using correlation threshold")
                
                # Fallback: simple correlation-based removal
                selected = []
                for feature in candidate_features:
                    feature_idx = candidate_indices[candidate_features.index(feature)]
                    
                    # Check correlation with already selected features
                    is_redundant = False
                    for selected_feature in selected:
                        selected_idx = candidate_indices[candidate_features.index(selected_feature)]
                        corr = abs(correlation_matrix[
                            candidate_indices.index(feature_idx),
                            candidate_indices.index(selected_idx)
                        ])
                        
                        if corr > self.redundancy_threshold:
                            is_redundant = True
                            break
                    
                    if not is_redundant:
                        selected.append(feature)
                    
                    if len(selected) >= self.target_feature_count:
                        break
                
                return selected
            
        except Exception as e:
            logger.warning(f"Error removing redundant features: {e}")
            return candidate_features[:self.target_feature_count]
    
    def _calculate_feature_stability(
        self,
        feature_values: np.ndarray,
        regime_labels: Optional[np.ndarray] = None
    ) -> float:
        """Calculate stability of feature across different conditions."""
        try:
            if regime_labels is None:
                # Use temporal stability (consistency across time)
                if len(feature_values) < 20:
                    return 0.5
                
                # Split into chunks and calculate consistency
                chunk_size = len(feature_values) // 5
                chunk_means = []
                
                for i in range(5):
                    start_idx = i * chunk_size
                    end_idx = (i + 1) * chunk_size if i < 4 else len(feature_values)
                    chunk_mean = np.mean(feature_values[start_idx:end_idx])
                    chunk_means.append(chunk_mean)
                
                # Stability = 1 - coefficient of variation
                mean_of_means = np.mean(chunk_means)
                std_of_means = np.std(chunk_means)
                
                stability = 1 - (std_of_means / (abs(mean_of_means) + 1e-8))
                return max(0.0, min(1.0, stability))
            
            else:
                # Regime-based stability
                unique_regimes = np.unique(regime_labels)
                if len(unique_regimes) < 2:
                    return 1.0
                
                regime_means = []
                for regime in unique_regimes:
                    regime_mask = regime_labels == regime
                    if np.sum(regime_mask) > 0:
                        regime_mean = np.mean(feature_values[regime_mask])
                        regime_means.append(regime_mean)
                
                if len(regime_means) < 2:
                    return 1.0
                
                # Stability across regimes
                mean_across_regimes = np.mean(regime_means)
                std_across_regimes = np.std(regime_means)
                
                stability = 1 - (std_across_regimes / (abs(mean_across_regimes) + 1e-8))
                return max(0.0, min(1.0, stability))
                
        except Exception:
            return 0.5
    
    def _calculate_regime_consistency(
        self,
        feature_values: np.ndarray,
        regime_labels: Optional[np.ndarray] = None
    ) -> float:
        """Calculate how consistently feature behaves across regimes."""
        try:
            if regime_labels is None:
                return 1.0  # No regime info
            
            unique_regimes = np.unique(regime_labels)
            if len(unique_regimes) < 2:
                return 1.0
            
            # Calculate correlation between feature and regime transitions
            regime_transitions = np.diff(regime_labels.astype(int))
            feature_changes = np.diff(feature_values)
            
            if len(regime_transitions) > 5 and len(feature_changes) > 5:
                try:
                    consistency_corr, _ = spearmanr(
                        abs(regime_transitions[:len(feature_changes)]),
                        abs(feature_changes[:len(regime_transitions)])
                    )
                    consistency = 1 - abs(consistency_corr) if not np.isnan(consistency_corr) else 1.0
                    return max(0.0, min(1.0, consistency))
                except:
                    return 1.0
            
            return 1.0
            
        except Exception:
            return 1.0
    
    def _calculate_redundancy_score(
        self,
        features: np.ndarray,
        feature_name: str,
        selected_features: List[str],
        all_feature_names: List[str]
    ) -> float:
        """Calculate redundancy score for a feature."""
        try:
            feature_idx = all_feature_names.index(feature_name)
            feature_values = features[:, feature_idx]
            
            max_correlation = 0.0
            for other_feature in selected_features:
                if other_feature != feature_name:
                    other_idx = all_feature_names.index(other_feature)
                    other_values = features[:, other_idx]
                    
                    try:
                        corr, _ = pearsonr(feature_values, other_values)
                        max_correlation = max(max_correlation, abs(corr) if not np.isnan(corr) else 0.0)
                    except:
                        continue
            
            return max_correlation
            
        except Exception:
            return 0.0
    
    def _calculate_permutation_importance(
        self,
        model,
        features: np.ndarray,
        targets: np.ndarray,
        n_repeats: int = 5
    ) -> np.ndarray:
        """Calculate permutation importance."""
        try:
            baseline_score = model.score(features, targets)
            importances = np.zeros(features.shape[1])
            
            for i in range(features.shape[1]):
                scores = []
                for _ in range(n_repeats):
                    # Permute feature i
                    features_permuted = features.copy()
                    np.random.shuffle(features_permuted[:, i])
                    
                    # Calculate score drop
                    permuted_score = model.score(features_permuted, targets)
                    score_drop = baseline_score - permuted_score
                    scores.append(score_drop)
                
                importances[i] = np.mean(scores)
            
            # Normalize to positive values
            importances = np.maximum(importances, 0)
            return importances
            
        except Exception as e:
            logger.warning(f"Error calculating permutation importance: {e}")
            return np.ones(features.shape[1]) * 0.1
    
    def _calculate_cross_method_stability(
        self,
        feature_name: str,
        method_results: List[SelectionResult]
    ) -> float:
        """Calculate stability of feature across different selection methods."""
        try:
            appearances = sum(1 for result in method_results if feature_name in result.selected_features)
            stability = appearances / len(method_results)
            return stability
            
        except Exception:
            return 0.5
    
    def _validate_selection(
        self,
        selected_features: List[str],
        train_features: np.ndarray,
        train_targets: np.ndarray,
        val_features: np.ndarray,
        val_targets: np.ndarray,
        all_feature_names: List[str]
    ) -> float:
        """Validate feature selection performance."""
        try:
            # Get indices of selected features
            selected_indices = [all_feature_names.index(name) for name in selected_features if name in all_feature_names]
            
            if not selected_indices:
                return 0.0
            
            # Train model with selected features
            selected_train = train_features[:, selected_indices]
            selected_val = val_features[:, selected_indices]
            
            # Simple validation model
            validation_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            
            validation_model.fit(selected_train, train_targets)
            selected_score = validation_model.score(selected_val, val_targets)
            
            # Baseline model with all features
            baseline_model = RandomForestRegressor(
                n_estimators=50,
                max_depth=8,
                random_state=42
            )
            
            baseline_model.fit(train_features, train_targets)
            baseline_score = baseline_model.score(val_features, val_targets)
            
            # Return improvement
            improvement = selected_score - baseline_score
            return improvement
            
        except Exception as e:
            logger.warning(f"Error validating selection: {e}")
            return 0.0
    
    def _create_fallback_result(
        self,
        feature_names: List[str],
        method: FeatureSelectionMethod
    ) -> SelectionResult:
        """Create fallback result when selection fails."""
        # Simple fallback: select first N features
        selected_features = feature_names[:min(self.target_feature_count, len(feature_names))]
        
        feature_scores = [
            FeatureScore(
                feature_name=name,
                score=1.0 / (i + 1),  # Decreasing scores
                metric_type=FeatureImportanceMetric.VARIANCE_SCORE,
                rank=i + 1,
                stability=0.5,
                regime_consistency=0.5,
                redundancy_score=0.0
            )
            for i, name in enumerate(selected_features)
        ]
        
        return SelectionResult(
            selected_features=selected_features,
            feature_scores=feature_scores,
            selection_method=method,
            selection_metadata={'fallback': True},
            performance_improvement=0.0,
            feature_reduction_ratio=len(selected_features) / len(feature_names)
        )
    
    def get_selection_history(self) -> List[SelectionResult]:
        """Get history of feature selections."""
        return self.selection_history.copy()
    
    def get_feature_importance_summary(self) -> Dict[str, Any]:
        """Get summary of feature importance across selections."""
        try:
            if not self.selection_history:
                return {}
            
            # Aggregate feature appearances and scores
            feature_appearances = Counter()
            feature_scores = defaultdict(list)
            
            for result in self.selection_history:
                for fs in result.feature_scores:
                    feature_appearances[fs.feature_name] += 1
                    feature_scores[fs.feature_name].append(fs.score)
            
            # Calculate summary statistics
            summary = {}
            for feature, count in feature_appearances.most_common():
                summary[feature] = {
                    'appearances': count,
                    'avg_score': np.mean(feature_scores[feature]),
                    'std_score': np.std(feature_scores[feature]),
                    'max_score': np.max(feature_scores[feature]),
                    'selection_frequency': count / len(self.selection_history)
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting feature importance summary: {e}")
            return {}
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics."""
        return {
            'target_feature_count': self.target_feature_count,
            'min_feature_count': self.min_feature_count,
            'max_feature_count': self.max_feature_count,
            'selection_history_count': len(self.selection_history),
            'stability_threshold': self.stability_threshold,
            'redundancy_threshold': self.redundancy_threshold,
            'enable_regime_awareness': self.enable_regime_awareness,
            'cross_validation_folds': self.cross_validation_folds,
            'cache_sizes': {
                'feature_importance': len(self.feature_importance_cache),
                'stability': len(self.stability_cache)
            }
        }
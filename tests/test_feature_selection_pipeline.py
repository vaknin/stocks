"""Tests for FeatureSelectionPipeline."""

import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / "src"))

from features.feature_selection_pipeline import (
    FeatureSelectionPipeline,
    FeatureSelectionMethod,
)


class TestFeatureSelectionPipeline(unittest.TestCase):
    def setUp(self):
        # Synthetic dataset with known signal features and redundancy
        rng = np.random.default_rng(42)
        n = 400
        x0 = rng.normal(size=n)
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)  # noise
        x3 = rng.normal(size=n)  # noise
        x0_red = x0 + rng.normal(scale=0.02, size=n)  # redundant with x0

        # Target depends strongly on x0 and x1
        y = 0.8 * x0 - 0.6 * x1 + rng.normal(scale=0.3, size=n)

        self.feature_names = ["x0", "x1", "x2", "x3", "x0_red"]
        self.X = np.column_stack([x0, x1, x2, x3, x0_red]).astype(np.float32)
        self.y = y.astype(np.float32)

        # Two regimes labels (e.g., low/high vol) for API coverage
        self.regimes = np.where(rng.random(n) > 0.5, 1, 0).astype(np.int32)

        self.pipe = FeatureSelectionPipeline(
            target_feature_count=3,
            min_feature_count=2,
            max_feature_count=5,
            redundancy_threshold=0.9,
            cross_validation_folds=3,
        )

    def test_mutual_information_selection(self):
        res = self.pipe.select_features(
            features=self.X,
            targets=self.y,
            feature_names=self.feature_names,
            method=FeatureSelectionMethod.MUTUAL_INFORMATION,
            regime_labels=self.regimes,
        )
        self.assertGreaterEqual(len(res.selected_features), 2)
        # True signals should be present
        self.assertTrue("x0" in res.selected_features or "x0_red" in res.selected_features)
        self.assertIn("x1", res.selected_features)
        self.assertTrue(0.0 <= res.feature_reduction_ratio <= 1.0)

    def test_random_forest_selection(self):
        res = self.pipe.select_features(
            features=self.X,
            targets=self.y,
            feature_names=self.feature_names,
            method=FeatureSelectionMethod.RANDOM_FOREST,
            regime_labels=self.regimes,
        )
        # Ensure redundancy removal prefers one of x0/x0_red, not both
        count_redundant = sum(1 for f in res.selected_features if f in ("x0", "x0_red"))
        self.assertLessEqual(count_redundant, 1)
        self.assertIn("x1", res.selected_features)

    def test_correlation_based_selection(self):
        res = self.pipe.select_features(
            features=self.X,
            targets=self.y,
            feature_names=self.feature_names,
            method=FeatureSelectionMethod.CORRELATION,
        )
        self.assertGreaterEqual(len(res.selected_features), 2)
        self.assertTrue("x0" in res.selected_features or "x0_red" in res.selected_features)
        self.assertIn("x1", res.selected_features)

    def test_hybrid_selection(self):
        res = self.pipe.select_features(
            features=self.X,
            targets=self.y,
            feature_names=self.feature_names,
            method=FeatureSelectionMethod.HYBRID,
            regime_labels=self.regimes,
        )
        self.assertLessEqual(len(res.selected_features), 3)
        self.assertIn(res.selection_method, (
            FeatureSelectionMethod.HYBRID,
            FeatureSelectionMethod.MUTUAL_INFORMATION,
            FeatureSelectionMethod.RANDOM_FOREST,
            FeatureSelectionMethod.CORRELATION,
        ))
        # Should contain core signals
        self.assertTrue("x0" in res.selected_features or "x0_red" in res.selected_features)
        self.assertIn("x1", res.selected_features)


if __name__ == "__main__":
    unittest.main()


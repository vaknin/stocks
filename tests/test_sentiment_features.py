"""Tests for sentiment feature extractor."""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from features.sentiment_features import SentimentFeatureExtractor, SentimentConfig


class TestSentimentFeatureExtractor(unittest.TestCase):
    """Test sentiment feature extractor functionality."""

    def setUp(self):
        self.config = SentimentConfig(
            model_name="yiyanghkust/finbert-tone",
            batch_size=8,
            lookback_days=30,
            short_window_days=7,
            long_window_days=21,
        )
        self.extractor = SentimentFeatureExtractor(self.config)

        # Build synthetic headlines (clearly polarized to reduce flakiness)
        now = datetime.utcnow()
        dates = [now - timedelta(days=i) for i in range(14)]

        positive_texts = [
            "Company posts record profit and raises guidance",
            "Earnings beat expectations; strong revenue growth",
            "Upgraded by analysts with buy rating",
            "Shares surge on positive outlook and robust cash flow",
            "Announces dividend increase and buyback program",
        ]
        negative_texts = [
            "Misses earnings; revenue declines sharply",
            "Downgraded to sell; weak outlook",
            "Faces lawsuit and regulatory probe; losses widen",
            "Shares plunge after guidance cut",
            "Supply issues hurt margins; cuts workforce",
        ]

        # Construct per-ticker DataFrames
        def make_df(texts):
            rows = []
            for i, t in enumerate(texts):
                rows.append({
                    'timestamp': dates[i % len(dates)],
                    'text': t,
                })
            return pd.DataFrame(rows)

        self.df_pos = make_df(positive_texts * 3)  # amplify to ensure signal
        self.df_neg = make_df(negative_texts * 3)

        # Multi-asset dict
        self.multi = {
            'AAPL': self.df_pos.copy(),
            'MSFT': self.df_neg.copy(),
        }

    def test_initialization(self):
        self.assertIsNotNone(self.extractor)
        self.assertGreater(len(self.extractor.get_feature_names()), 0)

    def test_basic_extraction_single(self):
        df = self.df_pos.copy()
        features = self.extractor.extract_features(df, 'AAPL')

        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(len(features), len(self.extractor.get_feature_names()))
        self.assertTrue(np.all(np.isfinite(features)))

        # Positive-heavy corpus should yield higher positive ratio than negative
        feature_names = self.extractor.get_feature_names()
        pos_ratio = features[feature_names.index('sent_pos_ratio')]
        neg_ratio = features[feature_names.index('sent_neg_ratio')]
        self.assertGreater(pos_ratio, neg_ratio)

    def test_basic_extraction_multi_asset(self):
        features = self.extractor.extract_features(self.multi, 'AAPL', reference_tickers=['AAPL', 'MSFT'])
        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(len(features), len(self.extractor.get_feature_names()))
        self.assertTrue(np.all(np.isfinite(features)))

        # Cross-asset breadth should be between 0 and 1
        fn = self.extractor.get_feature_names()
        breadth_pos = features[fn.index('breadth_positive')]
        breadth_neg = features[fn.index('breadth_negative')]
        self.assertGreaterEqual(breadth_pos, 0.0)
        self.assertLessEqual(breadth_pos, 1.0)
        self.assertGreaterEqual(breadth_neg, 0.0)
        self.assertLessEqual(breadth_neg, 1.0)

    def test_caching(self):
        df = self.df_pos.copy()
        self.extractor._feature_cache.clear()
        self.extractor._cache_timestamps.clear()

        f1 = self.extractor.extract_features(df, 'AAPL')
        f2 = self.extractor.extract_features(df, 'AAPL')
        np.testing.assert_array_equal(f1, f2)
        self.assertGreater(len(self.extractor._feature_cache), 0)

    def test_insufficient_and_bad_data(self):
        empty = pd.DataFrame()
        f = self.extractor.extract_features(empty, 'AAPL')
        self.assertEqual(len(f), len(self.extractor.get_feature_names()))
        self.assertTrue(np.all(f == 0.0))

        bad_cols = pd.DataFrame({'headline': ['x'], 'when': [datetime.utcnow()]})
        f2 = self.extractor.extract_features(bad_cols, 'AAPL')
        self.assertTrue(np.all(f2 == 0.0))


if __name__ == '__main__':
    unittest.main()


"""Sentiment feature extractor using finance-domain transformers models.

Generates per-ticker sentiment features from timestamped headlines/news.
Implements robust batching, caching, and bounded numeric feature outputs
for downstream ML models targeting revenue generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class SentimentConfig:
    model_name: str = "yiyanghkust/finbert-tone"
    device: Optional[str] = None  # "cuda", "cpu", or None for auto
    batch_size: int = 16
    lookback_days: int = 90
    short_window_days: int = 7
    long_window_days: int = 30
    cache_ttl_minutes: int = 15


class SentimentFeatureExtractor:
    """
    Finance NLP sentiment feature extractor.

    Input formats supported:
    - Single DataFrame with at least ['timestamp', 'text'] and optional ['ticker']
    - Dict[str, DataFrame] mapping ticker -> DataFrame with the same columns

    Output: fixed-length np.ndarray[float32] with bounded features.
    """

    def __init__(self, config: Optional[SentimentConfig] = None):
        self.config = config or SentimentConfig()
        self._pipeline = None  # Lazy-initialized transformers pipeline

        # Caches
        self._feature_cache: Dict[str, np.ndarray] = {}
        self._cache_timestamps: Dict[str, datetime] = {}

        logger.info(
            f"SentimentFeatureExtractor initialized (model={self.config.model_name}, lookback={self.config.lookback_days}d)"
        )

    # ------------- Public API -------------
    def extract_features(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        ticker: str,
        reference_tickers: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Extract sentiment features for a given ticker.

        Args:
            data: DataFrame or dict of DataFrames with columns ['timestamp', 'text'] and optional ['ticker']
            ticker: Primary ticker to extract features for
            reference_tickers: Optional list of other tickers to compute breadth/dispersion

        Returns:
            np.ndarray[float32] of fixed length features
        """
        try:
            primary_df, multi_asset = self._normalize_input(data, ticker)
            if primary_df is None or len(primary_df) == 0:
                return self._default_features()

            # Determine cache key and serve from cache if valid
            cache_key = self._cache_key(ticker, primary_df)
            if self._is_cache_valid(cache_key):
                return self._feature_cache[cache_key]

            # Restrict to lookback window
            cutoff = primary_df["timestamp"].max() - timedelta(days=self.config.lookback_days)
            recent_df = primary_df[primary_df["timestamp"] >= cutoff].copy()

            if len(recent_df) == 0:
                features = self._default_features()
                self._store_cache(cache_key, features)
                return features

            # Run transformer model to get class probabilities per text
            probs, max_probs = self._score_texts(recent_df["text"].tolist())
            if probs.size == 0:
                features = self._default_features()
                self._store_cache(cache_key, features)
                return features

            # Compute per-item sentiment in [-1, 1]
            sent_scores = self._to_scalar_sentiment(probs)

            # Merge back into DataFrame for time grouping
            recent_df = recent_df.reset_index(drop=True)
            recent_df["sentiment"] = sent_scores
            recent_df["conf_max"] = max_probs

            # Daily aggregates
            recent_df["date"] = recent_df["timestamp"].dt.date
            daily_mean = recent_df.groupby("date")["sentiment"].mean()
            daily_std = recent_df.groupby("date")["sentiment"].std().fillna(0.0)
            daily_count = recent_df.groupby("date")["sentiment"].size()

            # Class distribution across the window
            class_ratios = self._class_ratios(probs)
            pos_ratio, neg_ratio, neu_ratio = class_ratios

            # Entropy and disagreement
            entropy = self._entropy_from_probs(probs)
            disagreement = float(np.mean(np.std(probs, axis=1))) if len(probs) else 0.0
            top_conf = float(np.mean(max_probs)) if len(max_probs) else 0.0

            # Mean, std, and volatility of daily sentiment
            mean_sent = float(np.mean(sent_scores))
            std_sent = float(np.std(sent_scores))
            daily_vol = float(np.std(daily_mean)) if len(daily_mean) > 1 else 0.0

            # Short vs long momentum (7 vs 30 days)
            short_cutoff = recent_df["timestamp"].max() - timedelta(days=self.config.short_window_days)
            long_cutoff = recent_df["timestamp"].max() - timedelta(days=self.config.long_window_days)

            short_sent = recent_df[recent_df["timestamp"] >= short_cutoff]["sentiment"].mean()
            long_sent = recent_df[recent_df["timestamp"] >= long_cutoff]["sentiment"].mean()
            short_sent = float(0.0 if math.isnan(short_sent) else short_sent)
            long_sent = float(0.0 if math.isnan(long_sent) else long_sent)
            momentum_sl = short_sent - long_sent

            # Recency-weighted sentiment (EWMA)
            rec_weighted = self._recency_weighted_sentiment(recent_df)

            # Surprise: last 3d mean vs trailing 14d mean
            surprise = self._surprise_score(daily_mean)

            # Burstiness: share of days with |daily_mean| above threshold
            burst_threshold = 0.25  # moderately strong signal threshold
            burst_ratio = float(np.mean(np.abs(daily_mean) > burst_threshold)) if len(daily_mean) else 0.0

            # News flow intensity (normalized headline count per day)
            avg_per_day = float(np.mean(daily_count)) if len(daily_count) else 0.0
            news_intensity = min(avg_per_day / 50.0, 1.0)  # cap scale (50 items/day ~ 1.0)

            # Cross-asset breadth/dispersion if multi-asset available
            breadth_pos, breadth_neg, cs_spread, cs_dispersion = 0.0, 0.0, 0.0, 0.0
            if multi_asset and reference_tickers:
                breadth_pos, breadth_neg, cs_spread, cs_dispersion = self._cross_asset_breadth(
                    multi_asset, reference_tickers
                )

            # Simple trend of daily sentiment (slope)
            trend_slope = self._slope_of_series(daily_mean)

            # Last-day positivity (fraction positive labels)
            last_day = recent_df["date"].max()
            last_mask = recent_df["date"] == last_day
            last_probs = probs[last_mask.values]
            last_pos_frac = float(np.mean(last_probs[:, 0])) if last_probs.size else 0.0

            # Aggregate features vector
            feats = np.array(
                [
                    # Distribution
                    pos_ratio,
                    neg_ratio,
                    neu_ratio,
                    mean_sent,
                    std_sent,
                    entropy,
                    disagreement,
                    top_conf,
                    # Dynamics
                    momentum_sl,
                    rec_weighted,
                    surprise,
                    daily_vol,
                    burst_ratio,
                    news_intensity,
                    trend_slope,
                    last_pos_frac,
                    # Cross-asset
                    breadth_pos,
                    breadth_neg,
                    cs_spread,
                    cs_dispersion,
                ],
                dtype=np.float32,
            )

            # Clean and bound
            feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
            feats = np.clip(feats, -1.0, 1.0)

            self._store_cache(cache_key, feats)
            return feats
        except Exception as e:
            logger.error(f"Error extracting sentiment features for {ticker}: {e}")
            return self._default_features()

    def get_feature_names(self) -> List[str]:
        return [
            # Distribution
            "sent_pos_ratio",
            "sent_neg_ratio",
            "sent_neu_ratio",
            "sent_mean",
            "sent_std",
            "sent_entropy",
            "sent_disagreement",
            "sent_top_confidence",
            # Dynamics
            "sent_momentum_7v30",
            "sent_recency_weighted",
            "sent_surprise_3v14",
            "sent_daily_volatility",
            "sent_burst_ratio",
            "news_flow_intensity",
            "sent_trend_slope",
            "last_day_pos_frac",
            # Cross-asset
            "breadth_positive",
            "breadth_negative",
            "cross_asset_sent_spread",
            "cross_asset_sent_dispersion",
        ]

    # ------------- Internal helpers -------------
    def _normalize_input(
        self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], ticker: str
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, pd.DataFrame]]]:
        if isinstance(data, dict):
            if ticker not in data:
                logger.warning(f"Ticker {ticker} not present in provided multi-asset data")
                return None, data
            df = data[ticker].copy()
            multi_asset = data
        else:
            df = data.copy()
            multi_asset = None

        # Required columns
        if "timestamp" not in df.columns or "text" not in df.columns:
            return None, multi_asset

        # Ensure dtypes
        if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            except Exception:
                return None, multi_asset

        # Drop NaNs and empty texts
        df = df.dropna(subset=["timestamp", "text"]).copy()
        df = df[df["text"].astype(str).str.strip() != ""]

        return df, multi_asset

    def _load_pipeline(self):
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline
            pipe_device = 0 if (self.config.device or "").lower() == "cuda" else -1
            self._pipeline = pipeline(
                task="text-classification",
                model=self.config.model_name,
                return_all_scores=True,
                device=pipe_device,
                truncation=True,
                max_length=128,
            )
            logger.info(f"Loaded transformers pipeline: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to load transformers pipeline: {e}")
            self._pipeline = None

    def _score_texts(self, texts: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        """Return (probs[class_order], max_prob) for each text.

        Class order: [positive, negative, neutral]
        """
        self._load_pipeline()
        if self._pipeline is None or len(texts) == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # Run in batches to control memory
        all_probs: List[List[float]] = []
        all_max: List[float] = []
        bs = max(1, self.config.batch_size)
        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            try:
                outputs = self._pipeline(batch)
                # outputs is list of lists of dicts: [{label, score} * 3]
                for scores in outputs:
                    prob_map = {s["label"].lower(): float(s["score"]) for s in scores}
                    # normalize keys to expected
                    pos = prob_map.get("positive", prob_map.get("pos", 0.0))
                    neg = prob_map.get("negative", prob_map.get("neg", 0.0))
                    neu = prob_map.get("neutral", prob_map.get("neu", 0.0))
                    # safety normalize
                    total = pos + neg + neu
                    if total <= 0:
                        pos, neg, neu = 0.0, 0.0, 1.0
                        total = 1.0
                    pos, neg, neu = pos / total, neg / total, neu / total
                    all_probs.append([pos, neg, neu])
                    all_max.append(max(pos, neg, neu))
            except Exception as e:
                logger.debug(f"Pipeline batch scoring failed: {e}")
                # fill neutrals for batch
                for _ in batch:
                    all_probs.append([0.0, 0.0, 1.0])
                    all_max.append(1.0)

        return np.array(all_probs, dtype=np.float32), np.array(all_max, dtype=np.float32)

    def _to_scalar_sentiment(self, probs: np.ndarray) -> np.ndarray:
        # Map class probabilities to scalar: pos - neg in [-1, 1]
        if probs.size == 0:
            return np.zeros((0,), dtype=np.float32)
        pos = probs[:, 0]
        neg = probs[:, 1]
        score = pos - neg
        score = np.clip(score, -1.0, 1.0)
        return score.astype(np.float32)

    def _class_ratios(self, probs: np.ndarray) -> Tuple[float, float, float]:
        if probs.size == 0:
            return 0.0, 0.0, 1.0
        mean_probs = probs.mean(axis=0)
        return float(mean_probs[0]), float(mean_probs[1]), float(mean_probs[2])

    def _entropy_from_probs(self, probs: np.ndarray) -> float:
        if probs.size == 0:
            return 0.0
        eps = 1e-12
        ent = -np.sum(probs * np.log(probs + eps), axis=1)
        # Normalize to [0,1] for 3 classes: max entropy = ln(3)
        ent_norm = ent / math.log(3)
        return float(np.mean(np.clip(ent_norm, 0.0, 1.0)))

    def _recency_weighted_sentiment(self, df: pd.DataFrame) -> float:
        # Exponential decay with half-life ~ 7 days
        if len(df) == 0:
            return 0.0
        last_ts = df["timestamp"].max()
        half_life_days = 7.0
        decay_lambda = math.log(2) / half_life_days
        weights = np.exp(-decay_lambda * ((last_ts - df["timestamp"]).dt.total_seconds() / 86400.0))
        wmean = float(np.average(df["sentiment"], weights=weights)) if weights.sum() > 0 else 0.0
        return max(-1.0, min(1.0, wmean))

    def _surprise_score(self, daily_mean: pd.Series) -> float:
        if len(daily_mean) < 5:
            return 0.0
        # Align by date order
        dm = daily_mean.sort_index()
        recent = dm.tail(3).mean()
        trailing = dm.head(len(dm) - 3).tail(14).mean() if len(dm) > 3 else 0.0
        recent = float(0.0 if math.isnan(recent) else recent)
        trailing = float(0.0 if math.isnan(trailing) else trailing)
        return max(-1.0, min(1.0, recent - trailing))

    def _cross_asset_breadth(
        self, multi_asset: Dict[str, pd.DataFrame], reference_tickers: List[str]
    ) -> Tuple[float, float, float, float]:
        # Compute short-window mean sentiment for each reference asset
        pos_count = 0
        neg_count = 0
        sentiments: List[float] = []

        for t in reference_tickers[:20]:  # safety cap
            if t not in multi_asset:
                continue
            df = multi_asset[t]
            if not {"timestamp", "text"}.issubset(df.columns):
                continue
            df_local = df.dropna(subset=["timestamp", "text"]).copy()
            if len(df_local) == 0:
                continue
            cutoff = df_local["timestamp"].max() - timedelta(days=self.config.short_window_days)
            df_local = df_local[df_local["timestamp"] >= cutoff]
            if len(df_local) == 0:
                continue
            probs, _ = self._score_texts(df_local["text"].tolist())
            s = float(np.mean(self._to_scalar_sentiment(probs))) if probs.size else 0.0
            sentiments.append(s)
            if s > 0:
                pos_count += 1
            elif s < 0:
                neg_count += 1

        if not sentiments:
            return 0.0, 0.0, 0.0, 0.0
        breadth_pos = pos_count / max(1, len(sentiments))
        breadth_neg = neg_count / max(1, len(sentiments))
        spread = float(np.mean(sentiments))
        dispersion = float(np.std(sentiments))
        return (
            max(0.0, min(1.0, breadth_pos)),
            max(0.0, min(1.0, breadth_neg)),
            max(-1.0, min(1.0, spread)),
            max(0.0, min(1.0, dispersion)),
        )

    def _slope_of_series(self, s: pd.Series) -> float:
        if s is None or len(s) < 3:
            return 0.0
        y = s.values
        x = np.arange(len(y))
        try:
            slope = float(np.polyfit(x, y, 1)[0])
        except Exception:
            slope = 0.0
        return max(-1.0, min(1.0, slope))

    def _cache_key(self, ticker: str, df: pd.DataFrame) -> str:
        last_ts = df["timestamp"].max()
        return f"{ticker}_{len(df)}_{str(last_ts)}"

    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._feature_cache or key not in self._cache_timestamps:
            return False
        age_min = (datetime.now() - self._cache_timestamps[key]).seconds / 60
        return age_min < self.config.cache_ttl_minutes

    def _store_cache(self, key: str, features: np.ndarray) -> None:
        self._feature_cache[key] = features
        self._cache_timestamps[key] = datetime.now()
        self._evict_expired()

    def _evict_expired(self) -> None:
        now = datetime.now()
        expired = [
            k
            for k, ts in self._cache_timestamps.items()
            if (now - ts).seconds / 60 >= self.config.cache_ttl_minutes
        ]
        for k in expired:
            self._feature_cache.pop(k, None)
            self._cache_timestamps.pop(k, None)

    def _default_features(self) -> np.ndarray:
        return np.zeros(len(self.get_feature_names()), dtype=np.float32)


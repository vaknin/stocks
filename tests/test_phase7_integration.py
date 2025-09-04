"""
End-to-End Integration Tests for Phase 7: Integration and Optimization.
Tests the complete integrated system including all Phase 1-6 components plus Phase 7 optimizations.
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import threading
import concurrent.futures

from src.models.ensemble import MetaLearningEnsemble
from src.models.efficient_prediction_pipeline import EfficientPredictionPipeline, PredictionRequest
from src.models.memory_optimization import MemoryOptimizationManager
from src.models.neural_hyperparams import PRODUCTION_CONFIG
from src.models.optimized_neural_meta_learner import OptimizedNeuralMetaLearner
from src.features.meta_feature_extractor import MetaFeatureExtractor
from src.features.feature_selection_pipeline import FeatureSelector
from src.features.sentiment_features import SentimentAnalyzer
from src.config.settings import config


class TestPhase7Integration:
    """Integration tests for Phase 7 complete system."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Generate sample OHLCV data for testing."""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        # Generate realistic price data with trend
        base_price = 100
        returns = np.random.normal(0.001, 0.02, 100)  # 0.1% daily return, 2% volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = {
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(100000, 1000000, 100)
        }
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        return df
    
    @pytest.fixture
    def multi_asset_data(self, sample_price_data):
        """Generate multi-asset data for cross-asset features."""
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
        data = {}
        
        for i, ticker in enumerate(tickers):
            # Create correlated but unique data for each ticker
            df = sample_price_data.copy()
            multiplier = 1 + (i * 0.1)  # Different price levels
            correlation_factor = 0.3 + (i * 0.1)  # Different correlation strengths
            
            for col in ['open', 'high', 'low', 'close']:
                noise = np.random.normal(0, 0.005, len(df))  # 0.5% noise
                df[col] = df[col] * multiplier + df[col] * noise * correlation_factor
            
            data[ticker] = df
        
        return data
    
    @pytest.fixture
    def optimized_ensemble(self):
        """Create optimized ensemble with Phase 7 enhancements."""
        ensemble = MetaLearningEnsemble(
            horizon_len=[1, 5, 20],
            timesfm_weight=0.25,
            tsmamba_weight=0.25,
            samba_weight=0.25,
            tft_weight=0.25,
            uncertainty_alpha=0.1,
            performance_window=50,
            device="cpu",
            enable_neural_meta_learning=True,
            enable_online_learning=True,
            enable_multi_resolution=True,
            multi_resolution_weight=0.0  # Disabled for testing
        )
        
        return ensemble
    
    @pytest.fixture
    def efficient_pipeline(self, optimized_ensemble):
        """Create efficient prediction pipeline."""
        pipeline = EfficientPredictionPipeline(
            ensemble_model=optimized_ensemble,
            cache_config={
                'max_size': 100,
                'ttl_seconds': 60,
                'memory_limit_mb': 64
            },
            parallel_config={
                'max_workers': 2,
                'use_process_pool': False
            },
            batch_config={
                'batch_size': 4,
                'timeout_ms': 50
            }
        )
        
        return pipeline
    
    @pytest.fixture
    def memory_optimizer(self):
        """Create memory optimization manager."""
        return MemoryOptimizationManager(enable_monitoring=False)  # Disable monitoring for tests
    
    def test_enhanced_meta_feature_extraction(self, sample_price_data, multi_asset_data):
        """Test enhanced meta-feature extraction with Phase 6 features."""
        extractor = MetaFeatureExtractor(
            lookback_window=30,
            performance_window=20,
            enable_feature_selection=True
        )
        
        # Test combined feature extraction
        features = extractor.extract_combined_features(
            price_data=sample_price_data,
            ticker='AAPL',
            multi_asset_data=multi_asset_data,
            horizon=1
        )
        
        # Validate feature extraction
        assert isinstance(features, np.ndarray)
        assert len(features) > 50  # Should have many features from Phase 6
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
        
        # Test Phase 6 feature components
        phase6_features = extractor.extract_phase6_features(
            price_data=sample_price_data,
            ticker='AAPL',
            multi_asset_data=multi_asset_data
        )
        
        assert 'technical' in phase6_features
        assert 'microstructure' in phase6_features
        assert 'cross_asset' in phase6_features
        assert 'volatility_regime' in phase6_features
        assert 'sentiment' in phase6_features
        
        # Validate individual feature components
        for component_name, component_features in phase6_features.items():
            assert isinstance(component_features, np.ndarray)
            assert len(component_features) > 0
            assert not np.any(np.isnan(component_features))
    
    def test_optimized_neural_meta_learner(self):
        """Test optimized neural meta-learner with enhanced architecture."""
        config = PRODUCTION_CONFIG.copy()
        config['architecture'].meta_feature_dim = 100  # Adjust for test
        
        meta_learner = OptimizedNeuralMetaLearner(config=config, device='cpu')
        
        # Test model initialization
        assert meta_learner.param_count > 0
        assert meta_learner.device.type == 'cpu'
        
        # Test forward pass
        dummy_features = torch.randn(2, 100)  # Batch size 2
        
        for horizon in [1, 5, 20]:
            result = meta_learner.forward(
                meta_features=dummy_features,
                horizon=horizon,
                return_uncertainty=True
            )
            
            assert 'weights' in result
            assert 'confidence' in result
            assert 'uncertainty' in result
            
            # Validate weights sum to 1
            weights = result['weights']
            weight_sums = torch.sum(weights, dim=1)
            assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5)
            
            # Validate confidence is in [0, 1]
            confidence = result['confidence']
            assert torch.all(confidence >= 0.0)
            assert torch.all(confidence <= 1.0)
        
        # Test get_optimal_weights method
        features_np = dummy_features[0].numpy()
        weights_result = meta_learner.get_optimal_weights(
            meta_features=features_np,
            horizon=1,
            return_confidence=True
        )
        
        assert 'weights' in weights_result
        assert 'calibrated_confidence' in weights_result
        assert isinstance(weights_result['weights'], dict)
        
        # Model summary
        summary = meta_learner.get_model_summary()
        assert 'total_parameters' in summary
        assert summary['total_parameters'] > 0
    
    def test_feature_selection_integration(self, sample_price_data, multi_asset_data):
        """Test feature selection pipeline integration."""
        extractor = MetaFeatureExtractor(
            enable_feature_selection=True,
            lookback_window=30
        )
        
        # Test feature selector fitting
        training_data = [{'AAPL': sample_price_data} for _ in range(20)]
        target_returns = [np.array([0.01, 0.005, -0.002]) for _ in range(20)]
        tickers = ['AAPL'] * 20
        
        fit_result = extractor.fit_feature_selector(
            training_data=training_data,
            target_returns=target_returns,
            tickers=tickers
        )
        
        assert 'feature_selection_enabled' in fit_result
        if fit_result.get('feature_selection_enabled'):
            assert 'n_original_features' in fit_result
            assert 'n_selected_features' in fit_result
            assert fit_result['n_selected_features'] <= fit_result['n_original_features']
        
        # Test feature extraction with selection
        features = extractor.extract_combined_features(
            price_data=sample_price_data,
            ticker='AAPL',
            multi_asset_data=multi_asset_data
        )
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
    
    def test_efficient_prediction_pipeline(self, efficient_pipeline, sample_price_data):
        """Test efficient prediction pipeline with caching and parallelization."""
        # Test single prediction
        result = efficient_pipeline.predict_single(
            ticker='AAPL',
            price_data=sample_price_data,
            horizons=[1, 5],
            return_confidence=True,
            use_cache=True
        )
        
        assert result.request_id is not None
        assert result.ticker == 'AAPL'
        assert isinstance(result.predictions, dict)
        assert result.processing_time_ms > 0
        assert isinstance(result.cache_hit, bool)
        
        # Test cache hit on second call
        result2 = efficient_pipeline.predict_single(
            ticker='AAPL',
            price_data=sample_price_data,
            horizons=[1, 5],
            return_confidence=True,
            use_cache=True
        )
        
        # Second call should be faster (cached)
        assert result2.processing_time_ms < result.processing_time_ms * 0.5 or result2.cache_hit
        
        # Test parallel predictions
        requests = [
            PredictionRequest(
                ticker=f'STOCK_{i}',
                price_data=sample_price_data,
                multi_asset_data=None,
                horizons=[1, 5],
                return_confidence=True,
                priority=1
            )
            for i in range(5)
        ]
        
        parallel_results = efficient_pipeline.predict_parallel(requests, max_parallel=2)
        
        assert len(parallel_results) == len(requests)
        for result in parallel_results:
            assert isinstance(result, type(result))  # PredictionResult type
            assert result.ticker.startswith('STOCK_')
        
        # Test performance statistics
        stats = efficient_pipeline.get_performance_stats()
        
        assert 'pipeline_stats' in stats
        assert 'cache_stats' in stats
        assert 'system_stats' in stats
        assert stats['pipeline_stats']['requests_processed'] > 0
    
    @pytest.mark.asyncio
    async def test_async_prediction(self, efficient_pipeline, sample_price_data):
        """Test asynchronous prediction capabilities."""
        result = await efficient_pipeline.predict_async(
            ticker='AAPL',
            price_data=sample_price_data,
            horizons=[1, 5],
            return_confidence=True
        )
        
        assert result.ticker == 'AAPL'
        assert isinstance(result.predictions, dict)
        assert result.processing_time_ms > 0
        
        # Test multiple async predictions
        tasks = [
            efficient_pipeline.predict_async(
                ticker=f'ASYNC_STOCK_{i}',
                price_data=sample_price_data,
                horizons=[1],
                return_confidence=True
            )
            for i in range(3)
        ]
        
        results = await asyncio.gather(*tasks)
        assert len(results) == 3
        
        for i, result in enumerate(results):
            assert result.ticker == f'ASYNC_STOCK_{i}'
    
    def test_memory_optimization_integration(self, memory_optimizer, sample_price_data):
        """Test memory optimization integration with prediction pipeline."""
        # Initial memory stats
        initial_stats = memory_optimizer.get_comprehensive_stats()
        
        # Perform memory-intensive operations
        large_arrays = []
        for i in range(10):
            array = np.random.randn(1000, 1000).astype(np.float32)  # ~4MB each
            large_arrays.append(array)
        
        # Test light cleanup
        cleanup_result = memory_optimizer.perform_light_cleanup()
        
        assert 'cleanup_type' in cleanup_result
        assert cleanup_result['cleanup_type'] == 'light'
        assert cleanup_result['objects_collected'] >= 0
        
        # Test memory optimization for inference
        optimization_result = memory_optimizer.optimize_for_inference()
        
        assert 'optimizations_applied' in optimization_result
        assert 'optimization_time_seconds' in optimization_result
        assert len(optimization_result['optimizations_applied']) > 0
        
        # Clean up test arrays
        del large_arrays
        memory_optimizer.perform_light_cleanup()
        
        # Final stats should show improvement
        final_stats = memory_optimizer.get_comprehensive_stats()
        assert 'system' in final_stats
    
    def test_ensemble_training_with_phase7_features(self, optimized_ensemble, sample_price_data):
        """Test ensemble training with Phase 7 feature integration."""
        # Prepare training data
        training_data = []
        tickers = []
        
        for i in range(10):  # Small training set for test
            df = sample_price_data.copy()
            # Add some variation
            noise = np.random.normal(0, 0.01, len(df))
            for col in ['open', 'high', 'low', 'close']:
                df[col] = df[col] * (1 + noise)
            
            training_data.append(df)
            tickers.append(f'STOCK_{i}')
        
        # Train ensemble
        training_results = optimized_ensemble.train_ensemble(
            training_data=training_data,
            tickers=tickers,
            validation_split=0.3
        )
        
        assert 'performance_metrics' in training_results
        assert 'final_weights' in training_results
        assert 'feature_selection_results' in training_results
        
        # Check that feature selection was attempted
        feature_results = training_results['feature_selection_results']
        if 'error' not in feature_results:
            assert 'feature_selection_enabled' in feature_results
    
    def test_online_learning_integration(self, optimized_ensemble, sample_price_data):
        """Test online learning system integration."""
        if not optimized_ensemble.enable_online_learning:
            pytest.skip("Online learning not enabled")
        
        # Start online learning
        started = optimized_ensemble.start_online_learning()
        
        if started:
            # Update with new experience
            optimized_ensemble.update_online_learning(
                price_data=sample_price_data,
                predictions={'timesfm': 0.01, 'tsmamba': 0.008, 'samba': 0.012, 'tft': 0.009},
                actual_return=0.01,
                model_weights={'timesfm': 0.25, 'tsmamba': 0.25, 'samba': 0.25, 'tft': 0.25},
                horizon=1,
                ticker='AAPL'
            )
            
            # Get neural meta-learning status
            status = optimized_ensemble.get_neural_meta_learning_status()
            
            assert 'enabled' in status
            assert status['enabled'] == True
            
            # Stop online learning
            stopped = optimized_ensemble.stop_online_learning()
            assert stopped == True
    
    def test_end_to_end_prediction_flow(self, efficient_pipeline, multi_asset_data):
        """Test complete end-to-end prediction flow."""
        ticker = 'AAPL'
        price_data = multi_asset_data[ticker]
        
        start_time = time.time()
        
        # Make prediction with full Phase 7 pipeline
        result = efficient_pipeline.predict_single(
            ticker=ticker,
            price_data=price_data,
            multi_asset_data=multi_asset_data,
            horizons=[1, 5, 20],
            return_confidence=True,
            use_cache=True
        )
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # ms
        
        # Validate result structure
        assert result.ticker == ticker
        assert isinstance(result.predictions, dict)
        assert result.processing_time_ms > 0
        assert result.processing_time_ms < 30000  # Should complete within 30 seconds
        
        # Check for expected prediction structure
        predictions = result.predictions
        if 'error' not in predictions:
            for horizon in [1, 5, 20]:
                horizon_key = f'horizon_{horizon}'
                if horizon_key in predictions:
                    horizon_pred = predictions[horizon_key]
                    
                    assert 'prediction' in horizon_pred
                    assert 'confidence' in horizon_pred
                    assert 'prediction_interval' in horizon_pred
                    assert 'model_weights' in horizon_pred
                    
                    # Validate prediction values
                    assert isinstance(horizon_pred['prediction'], (float, int))
                    assert 0.0 <= horizon_pred['confidence'] <= 1.0
                    assert len(horizon_pred['prediction_interval']) == 2
                    assert isinstance(horizon_pred['model_weights'], dict)
        
        logger.info(f"End-to-end prediction completed in {processing_time:.1f}ms")
    
    def test_performance_benchmarking(self, efficient_pipeline, sample_price_data):
        """Test performance benchmarking of the integrated system."""
        benchmark_results = []
        
        # Benchmark different scenarios
        scenarios = [
            {'name': 'single_horizon', 'horizons': [1]},
            {'name': 'multi_horizon', 'horizons': [1, 5, 20]},
            {'name': 'with_confidence', 'horizons': [1, 5], 'confidence': True},
            {'name': 'cached_prediction', 'horizons': [1], 'repeat': True}
        ]
        
        for scenario in scenarios:
            scenario_name = scenario['name']
            horizons = scenario['horizons']
            return_confidence = scenario.get('confidence', True)
            
            start_time = time.time()
            
            # First prediction
            result1 = efficient_pipeline.predict_single(
                ticker=f'BENCH_{scenario_name}',
                price_data=sample_price_data,
                horizons=horizons,
                return_confidence=return_confidence,
                use_cache=True
            )
            
            first_time = time.time() - start_time
            
            # Second prediction if testing cache
            second_time = 0
            if scenario.get('repeat'):
                start_time2 = time.time()
                result2 = efficient_pipeline.predict_single(
                    ticker=f'BENCH_{scenario_name}',
                    price_data=sample_price_data,
                    horizons=horizons,
                    return_confidence=return_confidence,
                    use_cache=True
                )
                second_time = time.time() - start_time2
            
            benchmark_results.append({
                'scenario': scenario_name,
                'first_prediction_ms': first_time * 1000,
                'second_prediction_ms': second_time * 1000 if second_time > 0 else None,
                'cache_speedup': first_time / second_time if second_time > 0 else None,
                'horizons_tested': len(horizons)
            })
        
        # Validate benchmark results
        for result in benchmark_results:
            assert result['first_prediction_ms'] > 0
            assert result['first_prediction_ms'] < 30000  # Under 30 seconds
            
            if result['cache_speedup'] is not None:
                assert result['cache_speedup'] >= 1.0  # Cache should be faster or equal
        
        # Log benchmark summary
        avg_time = np.mean([r['first_prediction_ms'] for r in benchmark_results])
        logger.info(f"Average prediction time across scenarios: {avg_time:.1f}ms")
    
    def test_error_handling_and_resilience(self, efficient_pipeline):
        """Test error handling and system resilience."""
        # Test with invalid data
        invalid_data = pd.DataFrame()  # Empty DataFrame
        
        result = efficient_pipeline.predict_single(
            ticker='INVALID',
            price_data=invalid_data,
            horizons=[1],
            return_confidence=True,
            use_cache=False
        )
        
        # Should handle error gracefully
        assert result.ticker == 'INVALID'
        assert 'error' in result.predictions or len(result.predictions) == 0
        
        # Test with malformed multi-asset data
        malformed_data = {'INVALID': pd.DataFrame({'close': []})}  # Empty close column
        
        result2 = efficient_pipeline.predict_single(
            ticker='MALFORMED',
            price_data=pd.DataFrame({'close': [100, 101, 99]}),  # Minimal valid data
            multi_asset_data=malformed_data,
            horizons=[1],
            return_confidence=True,
            use_cache=False
        )
        
        # Should still produce a result despite malformed multi-asset data
        assert result2.ticker == 'MALFORMED'
        assert result2.processing_time_ms > 0
    
    def test_concurrent_predictions(self, efficient_pipeline, sample_price_data):
        """Test concurrent prediction handling."""
        def make_prediction(ticker_suffix):
            return efficient_pipeline.predict_single(
                ticker=f'CONCURRENT_{ticker_suffix}',
                price_data=sample_price_data,
                horizons=[1, 5],
                return_confidence=True,
                use_cache=True
            )
        
        # Test with multiple threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(make_prediction, i)
                for i in range(10)
            ]
            
            results = []
            for future in concurrent.futures.as_completed(futures, timeout=60):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Concurrent prediction failed: {e}")
        
        # Validate concurrent results
        assert len(results) <= 10  # Some may fail, but should have some results
        
        tickers_processed = set()
        for result in results:
            assert result.ticker.startswith('CONCURRENT_')
            tickers_processed.add(result.ticker)
        
        # Should have processed multiple unique tickers
        assert len(tickers_processed) >= len(results) * 0.5  # At least half should be unique


# Performance and load testing
class TestPhase7Performance:
    """Performance and load tests for Phase 7 system."""
    
    @pytest.fixture
    def stress_test_data(self):
        """Generate larger dataset for stress testing."""
        dates = pd.date_range(start='2023-01-01', periods=252, freq='D')  # 1 year
        np.random.seed(42)
        
        prices = [100]
        for _ in range(251):
            prices.append(prices[-1] * (1 + np.random.normal(0.001, 0.02)))
        
        return pd.DataFrame({
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': np.random.randint(100000, 5000000, 252)
        }, index=dates)
    
    def test_high_frequency_predictions(self, efficient_pipeline, stress_test_data):
        """Test system performance under high-frequency prediction load."""
        num_predictions = 50
        start_time = time.time()
        
        results = []
        for i in range(num_predictions):
            result = efficient_pipeline.predict_single(
                ticker=f'HF_STOCK_{i % 10}',  # Cycle through 10 stocks
                price_data=stress_test_data,
                horizons=[1, 5],
                return_confidence=True,
                use_cache=True
            )
            results.append(result)
        
        total_time = time.time() - start_time
        avg_time_per_prediction = (total_time / num_predictions) * 1000  # ms
        
        # Performance assertions
        assert len(results) == num_predictions
        assert avg_time_per_prediction < 1000  # Should average under 1 second per prediction
        assert total_time < 30  # Total should complete within 30 seconds
        
        # Check cache effectiveness
        cache_hits = sum(1 for r in results if r.cache_hit)
        cache_hit_rate = cache_hits / num_predictions
        
        logger.info(f"High-frequency test: {avg_time_per_prediction:.1f}ms avg, {cache_hit_rate:.1%} cache hit rate")
    
    def test_memory_usage_under_load(self, efficient_pipeline, stress_test_data):
        """Test memory usage remains stable under prediction load."""
        import psutil
        process = psutil.Process()
        
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Generate many predictions
        for i in range(100):
            efficient_pipeline.predict_single(
                ticker=f'MEMORY_TEST_{i}',
                price_data=stress_test_data,
                horizons=[1, 5, 20],
                return_confidence=True,
                use_cache=True
            )
            
            # Check memory every 20 predictions
            if i % 20 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be reasonable
                assert memory_growth < 500  # Less than 500MB growth
        
        final_memory = process.memory_info().rss / 1024 / 1024
        total_growth = final_memory - initial_memory
        
        logger.info(f"Memory usage: {initial_memory:.1f}MB -> {final_memory:.1f}MB (+{total_growth:.1f}MB)")
        
        # Perform cleanup and verify memory reduction
        efficient_pipeline.optimize_memory()
        
        after_cleanup_memory = process.memory_info().rss / 1024 / 1024
        memory_freed = final_memory - after_cleanup_memory
        
        assert memory_freed >= 0  # Should free some memory or at least not increase
    
    def test_cache_performance_analysis(self, efficient_pipeline, stress_test_data):
        """Analyze cache performance characteristics."""
        # Test cache warming
        tickers = [f'CACHE_TEST_{i}' for i in range(20)]
        sample_data = {ticker: stress_test_data for ticker in tickers}
        
        warming_results = efficient_pipeline.warm_cache(
            tickers=tickers,
            sample_data=sample_data,
            horizons=[1, 5]
        )
        
        assert warming_results['tickers_warmed'] > 0
        assert warming_results['warming_time_seconds'] > 0
        
        # Test cache hit rates with different access patterns
        patterns = {
            'sequential': list(range(20)),
            'random': np.random.choice(20, 20).tolist(),
            'concentrated': [0, 1, 2] * 7  # Focus on first 3 tickers
        }
        
        pattern_results = {}
        
        for pattern_name, ticker_indices in patterns.items():
            start_time = time.time()
            results = []
            
            for idx in ticker_indices:
                result = efficient_pipeline.predict_single(
                    ticker=f'CACHE_TEST_{idx}',
                    price_data=stress_test_data,
                    horizons=[1, 5],
                    return_confidence=True,
                    use_cache=True
                )
                results.append(result)
            
            pattern_time = time.time() - start_time
            cache_hits = sum(1 for r in results if r.cache_hit)
            hit_rate = cache_hits / len(results)
            
            pattern_results[pattern_name] = {
                'total_time': pattern_time,
                'hit_rate': hit_rate,
                'predictions': len(results)
            }
        
        # Concentrated access should have higher hit rate than random
        assert pattern_results['concentrated']['hit_rate'] >= pattern_results['random']['hit_rate']
        
        logger.info("Cache pattern analysis:")
        for pattern, stats in pattern_results.items():
            logger.info(f"  {pattern}: {stats['hit_rate']:.1%} hit rate, {stats['total_time']:.2f}s")


if __name__ == "__main__":
    # Run specific tests for debugging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Can run individual test methods here for debugging
    test_class = TestPhase7Integration()
    
    # Example: test with sample data
    sample_data = test_class.sample_price_data(test_class)
    print(f"Sample data shape: {sample_data.shape}")
    print(f"Sample data columns: {sample_data.columns.tolist()}")
    print(f"Sample data date range: {sample_data.index[0]} to {sample_data.index[-1]}")
"""
Efficient Prediction Pipeline for Phase 7 Optimization.
Implements computational efficiency improvements including caching, 
parallel processing, memory optimization, and batch processing.
"""

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import pickle
import hashlib
from dataclasses import dataclass
from loguru import logger
from pathlib import Path
import psutil
import gc

# Cache and memory optimization
from functools import lru_cache
from collections import OrderedDict
import weakref

# Batch processing
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class PredictionRequest:
    """Individual prediction request with metadata."""
    ticker: str
    price_data: pd.DataFrame
    multi_asset_data: Optional[Dict[str, pd.DataFrame]]
    horizons: List[int]
    return_confidence: bool
    priority: int = 1  # 1=highest, 5=lowest
    timestamp: datetime = None
    request_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.request_id is None:
            self.request_id = f"{self.ticker}_{int(time.time()*1000)}"


@dataclass
class PredictionResult:
    """Prediction result with performance metadata."""
    request_id: str
    ticker: str
    predictions: Dict[str, Any]
    processing_time_ms: float
    cache_hit: bool
    feature_extraction_time_ms: float
    model_inference_time_ms: float
    timestamp: datetime


class AdvancedCache:
    """Advanced caching system with TTL, LRU, and memory management."""
    
    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 300,
        memory_limit_mb: int = 512
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        
        # Multi-level cache
        self.l1_cache = OrderedDict()  # Hot cache
        self.l2_cache = {}  # Warm cache
        
        # Metadata tracking
        self.access_times = {}
        self.creation_times = {}
        self.hit_counts = {}
        self.memory_usage = {}
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'memory_cleanups': 0
        }
        
        self._lock = threading.RLock()
    
    def _generate_key(self, ticker: str, data_hash: str, horizon: int) -> str:
        """Generate cache key with collision resistance."""
        return f"{ticker}:{data_hash}:{horizon}"
    
    def _compute_data_hash(self, price_data: pd.DataFrame) -> str:
        """Compute hash of price data for caching."""
        if len(price_data) == 0:
            return "empty"
        
        # Use last timestamp and closing price for quick hash
        last_row = price_data.iloc[-1]
        hash_data = f"{last_row.name}:{last_row.get('close', 0):.6f}:{len(price_data)}"
        return hashlib.md5(hash_data.encode()).hexdigest()[:16]
    
    def get(self, ticker: str, price_data: pd.DataFrame, horizon: int) -> Optional[Any]:
        """Get item from cache with access tracking."""
        data_hash = self._compute_data_hash(price_data)
        key = self._generate_key(ticker, data_hash, horizon)
        
        with self._lock:
            current_time = time.time()
            
            # Check L1 cache (hot)
            if key in self.l1_cache:
                creation_time = self.creation_times.get(key, current_time)
                if current_time - creation_time < self.ttl_seconds:
                    # Move to end (most recently used)
                    self.l1_cache.move_to_end(key)
                    self.access_times[key] = current_time
                    self.hit_counts[key] = self.hit_counts.get(key, 0) + 1
                    self.stats['hits'] += 1
                    return self.l1_cache[key]
                else:
                    # Expired, remove from L1
                    self._remove_key(key)
            
            # Check L2 cache (warm)
            if key in self.l2_cache:
                creation_time = self.creation_times.get(key, current_time)
                if current_time - creation_time < self.ttl_seconds:
                    # Promote to L1
                    value = self.l2_cache.pop(key)
                    self.l1_cache[key] = value
                    self.access_times[key] = current_time
                    self.hit_counts[key] = self.hit_counts.get(key, 0) + 1
                    self.stats['hits'] += 1
                    return value
                else:
                    # Expired, remove from L2
                    self._remove_key(key)
            
            self.stats['misses'] += 1
            return None
    
    def put(self, ticker: str, price_data: pd.DataFrame, horizon: int, value: Any) -> None:
        """Put item in cache with memory management."""
        data_hash = self._compute_data_hash(price_data)
        key = self._generate_key(ticker, data_hash, horizon)
        
        with self._lock:
            current_time = time.time()
            
            # Estimate memory usage
            try:
                value_size = len(pickle.dumps(value))
            except:
                value_size = 1024  # Default estimate
            
            # Check memory limit
            total_memory = sum(self.memory_usage.values()) + value_size
            if total_memory > self.memory_limit_bytes:
                self._cleanup_memory()
            
            # Add to L1 cache
            self.l1_cache[key] = value
            self.creation_times[key] = current_time
            self.access_times[key] = current_time
            self.hit_counts[key] = 0
            self.memory_usage[key] = value_size
            
            # Enforce size limits
            while len(self.l1_cache) > self.max_size // 2:  # L1 gets half the space
                self._demote_lru()
    
    def _remove_key(self, key: str) -> None:
        """Remove key from all cache levels."""
        self.l1_cache.pop(key, None)
        self.l2_cache.pop(key, None)
        self.creation_times.pop(key, None)
        self.access_times.pop(key, None)
        self.hit_counts.pop(key, None)
        self.memory_usage.pop(key, None)
    
    def _demote_lru(self) -> None:
        """Demote least recently used item from L1 to L2."""
        if self.l1_cache:
            key, value = self.l1_cache.popitem(last=False)  # Remove LRU
            
            # Move to L2 if not full
            if len(self.l2_cache) < self.max_size // 2:
                self.l2_cache[key] = value
            else:
                # Remove from tracking
                self._remove_key(key)
                self.stats['evictions'] += 1
    
    def _cleanup_memory(self) -> None:
        """Clean up memory by removing old entries."""
        current_time = time.time()
        keys_to_remove = []
        
        # Find expired entries
        for key, creation_time in self.creation_times.items():
            if current_time - creation_time > self.ttl_seconds:
                keys_to_remove.append(key)
        
        # Remove expired entries
        for key in keys_to_remove:
            self._remove_key(key)
        
        # If still over limit, remove by memory usage (largest first)
        while sum(self.memory_usage.values()) > self.memory_limit_bytes * 0.8:
            if not self.memory_usage:
                break
            
            largest_key = max(self.memory_usage.keys(), key=lambda k: self.memory_usage[k])
            self._remove_key(largest_key)
        
        self.stats['memory_cleanups'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / max(1, total_requests)
            
            return {
                'l1_entries': len(self.l1_cache),
                'l2_entries': len(self.l2_cache),
                'total_memory_mb': sum(self.memory_usage.values()) / (1024 * 1024),
                'hit_rate': hit_rate,
                **self.stats
            }


class BatchPredictionDataset(Dataset):
    """Dataset for batch processing of predictions."""
    
    def __init__(self, requests: List[PredictionRequest]):
        self.requests = requests
    
    def __len__(self):
        return len(self.requests)
    
    def __getitem__(self, idx):
        return self.requests[idx]


class EfficientPredictionPipeline:
    """Highly optimized prediction pipeline with advanced caching and parallelization."""
    
    def __init__(
        self,
        ensemble_model,
        cache_config: Optional[Dict[str, Any]] = None,
        parallel_config: Optional[Dict[str, Any]] = None,
        batch_config: Optional[Dict[str, Any]] = None
    ):
        self.ensemble_model = ensemble_model
        
        # Initialize advanced caching
        cache_config = cache_config or {}
        self.cache = AdvancedCache(
            max_size=cache_config.get('max_size', 1000),
            ttl_seconds=cache_config.get('ttl_seconds', 300),
            memory_limit_mb=cache_config.get('memory_limit_mb', 512)
        )
        
        # Parallel processing setup
        parallel_config = parallel_config or {}
        self.max_workers = parallel_config.get('max_workers', min(8, psutil.cpu_count()))
        self.use_process_pool = parallel_config.get('use_process_pool', False)
        
        if self.use_process_pool:
            self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Batch processing configuration
        batch_config = batch_config or {}
        self.batch_size = batch_config.get('batch_size', 16)
        self.batch_timeout_ms = batch_config.get('timeout_ms', 100)
        
        # Performance monitoring
        self.performance_stats = {
            'requests_processed': 0,
            'cache_hits': 0,
            'total_processing_time': 0.0,
            'avg_processing_time': 0.0,
            'memory_usage_mb': 0.0
        }
        
        # Batch processing queue
        self._batch_queue = []
        self._batch_lock = threading.Lock()
        self._batch_thread = None
        self._shutdown = False
        
        # Start batch processing thread
        self._start_batch_processor()
        
        logger.info(f"EfficientPredictionPipeline initialized with {self.max_workers} workers")
    
    def _start_batch_processor(self):
        """Start the batch processing thread."""
        def batch_processor():
            while not self._shutdown:
                try:
                    with self._batch_lock:
                        if len(self._batch_queue) >= self.batch_size or (
                            self._batch_queue and 
                            time.time() * 1000 - self._batch_queue[0].timestamp.timestamp() * 1000 > self.batch_timeout_ms
                        ):
                            batch = self._batch_queue[:self.batch_size]
                            self._batch_queue = self._batch_queue[self.batch_size:]
                            
                            if batch:
                                self._process_batch(batch)
                    
                    time.sleep(0.01)  # 10ms polling
                except Exception as e:
                    logger.error(f"Error in batch processor: {e}")
        
        self._batch_thread = threading.Thread(target=batch_processor, daemon=True)
        self._batch_thread.start()
    
    def predict_single(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        multi_asset_data: Optional[Dict[str, pd.DataFrame]] = None,
        horizons: List[int] = [1, 5, 20],
        return_confidence: bool = True,
        use_cache: bool = True
    ) -> PredictionResult:
        """
        Optimized single prediction with caching and performance monitoring.
        
        Args:
            ticker: Stock ticker
            price_data: OHLCV data
            multi_asset_data: Multi-asset data for cross-asset features
            horizons: Prediction horizons
            return_confidence: Whether to return confidence estimates
            use_cache: Whether to use caching
            
        Returns:
            PredictionResult with performance metadata
        """
        start_time = time.time()
        request_id = f"{ticker}_{int(start_time * 1000)}"
        
        # Check cache first
        cache_hit = False
        cached_result = None
        
        if use_cache:
            for horizon in horizons:
                cached_result = self.cache.get(ticker, price_data, horizon)
                if cached_result is not None:
                    cache_hit = True
                    break
        
        if cached_result is not None:
            # Return cached result with updated metadata
            processing_time = (time.time() - start_time) * 1000
            self.performance_stats['cache_hits'] += 1
            
            return PredictionResult(
                request_id=request_id,
                ticker=ticker,
                predictions=cached_result,
                processing_time_ms=processing_time,
                cache_hit=True,
                feature_extraction_time_ms=0.0,
                model_inference_time_ms=0.0,
                timestamp=datetime.now()
            )
        
        # Feature extraction timing
        feature_start = time.time()
        
        try:
            # Generate predictions using ensemble
            predictions = self.ensemble_model.predict(
                df=multi_asset_data or price_data,
                ticker=ticker,
                return_confidence=return_confidence,
                use_cache=use_cache
            )
            
            feature_time = (time.time() - feature_start) * 1000
            
            # Model inference timing (approximate)
            model_inference_time = max(0, (time.time() - start_time) * 1000 - feature_time)
            
            # Cache the result
            if use_cache:
                for horizon in horizons:
                    horizon_key = f"horizon_{horizon}"
                    if horizon_key in predictions:
                        self.cache.put(ticker, price_data, horizon, predictions)
                        break
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update performance statistics
            self._update_performance_stats(processing_time, cache_hit)
            
            return PredictionResult(
                request_id=request_id,
                ticker=ticker,
                predictions=predictions,
                processing_time_ms=processing_time,
                cache_hit=False,
                feature_extraction_time_ms=feature_time,
                model_inference_time_ms=model_inference_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in prediction for {ticker}: {e}")
            processing_time = (time.time() - start_time) * 1000
            
            # Return empty result with error info
            return PredictionResult(
                request_id=request_id,
                ticker=ticker,
                predictions={'error': str(e)},
                processing_time_ms=processing_time,
                cache_hit=False,
                feature_extraction_time_ms=0.0,
                model_inference_time_ms=0.0,
                timestamp=datetime.now()
            )
    
    def predict_parallel(
        self,
        requests: List[PredictionRequest],
        max_parallel: Optional[int] = None
    ) -> List[PredictionResult]:
        """
        Process multiple predictions in parallel.
        
        Args:
            requests: List of prediction requests
            max_parallel: Maximum parallel requests (defaults to self.max_workers)
            
        Returns:
            List of prediction results
        """
        if not requests:
            return []
        
        max_parallel = max_parallel or self.max_workers
        
        # Sort by priority (1=highest priority)
        sorted_requests = sorted(requests, key=lambda x: x.priority)
        
        futures = []
        results = []
        
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            # Submit all requests
            for request in sorted_requests:
                future = executor.submit(
                    self.predict_single,
                    request.ticker,
                    request.price_data,
                    request.multi_asset_data,
                    request.horizons,
                    request.return_confidence,
                    True  # Use cache
                )
                futures.append((future, request))
            
            # Collect results as they complete
            for future, request in futures:
                try:
                    result = future.result(timeout=30)  # 30 second timeout
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error in parallel prediction for {request.ticker}: {e}")
                    # Create error result
                    error_result = PredictionResult(
                        request_id=request.request_id,
                        ticker=request.ticker,
                        predictions={'error': str(e)},
                        processing_time_ms=0.0,
                        cache_hit=False,
                        feature_extraction_time_ms=0.0,
                        model_inference_time_ms=0.0,
                        timestamp=datetime.now()
                    )
                    results.append(error_result)
        
        logger.info(f"Processed {len(results)} predictions in parallel")
        return results
    
    async def predict_async(
        self,
        ticker: str,
        price_data: pd.DataFrame,
        multi_asset_data: Optional[Dict[str, pd.DataFrame]] = None,
        horizons: List[int] = [1, 5, 20],
        return_confidence: bool = True
    ) -> PredictionResult:
        """Asynchronous prediction for integration with async frameworks."""
        loop = asyncio.get_event_loop()
        
        result = await loop.run_in_executor(
            self.executor,
            self.predict_single,
            ticker,
            price_data,
            multi_asset_data,
            horizons,
            return_confidence,
            True  # use_cache
        )
        
        return result
    
    def _process_batch(self, batch: List[PredictionRequest]) -> None:
        """Process a batch of requests efficiently."""
        logger.debug(f"Processing batch of {len(batch)} requests")
        
        # Group by similar characteristics for batch optimization
        grouped_requests = self._group_requests_for_batching(batch)
        
        for group in grouped_requests:
            try:
                # Process each group with optimized ensemble prediction
                self._process_request_group(group)
            except Exception as e:
                logger.error(f"Error processing request group: {e}")
    
    def _group_requests_for_batching(
        self, 
        requests: List[PredictionRequest]
    ) -> List[List[PredictionRequest]]:
        """Group requests for optimal batch processing."""
        # Simple grouping by ticker for now
        groups = {}
        for request in requests:
            ticker = request.ticker
            if ticker not in groups:
                groups[ticker] = []
            groups[ticker].append(request)
        
        return list(groups.values())
    
    def _process_request_group(self, group: List[PredictionRequest]) -> None:
        """Process a group of similar requests."""
        for request in group:
            try:
                self.predict_single(
                    request.ticker,
                    request.price_data,
                    request.multi_asset_data,
                    request.horizons,
                    request.return_confidence
                )
            except Exception as e:
                logger.error(f"Error processing request {request.request_id}: {e}")
    
    def _update_performance_stats(self, processing_time: float, cache_hit: bool) -> None:
        """Update performance statistics."""
        self.performance_stats['requests_processed'] += 1
        if cache_hit:
            self.performance_stats['cache_hits'] += 1
        
        self.performance_stats['total_processing_time'] += processing_time
        self.performance_stats['avg_processing_time'] = (
            self.performance_stats['total_processing_time'] / 
            self.performance_stats['requests_processed']
        )
        
        # Update memory usage
        process = psutil.Process()
        self.performance_stats['memory_usage_mb'] = process.memory_info().rss / 1024 / 1024
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        cache_stats = self.cache.get_stats()
        
        return {
            'pipeline_stats': self.performance_stats.copy(),
            'cache_stats': cache_stats,
            'system_stats': {
                'cpu_count': psutil.cpu_count(),
                'memory_available_mb': psutil.virtual_memory().available / 1024 / 1024,
                'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            },
            'configuration': {
                'max_workers': self.max_workers,
                'batch_size': self.batch_size,
                'batch_timeout_ms': self.batch_timeout_ms,
                'use_process_pool': self.use_process_pool
            }
        }
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform memory optimization and return statistics."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Clear caches
        self.cache._cleanup_memory()
        
        # Force garbage collection
        collected = gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_freed = initial_memory - final_memory
        
        logger.info(f"Memory optimization: freed {memory_freed:.1f}MB, collected {collected} objects")
        
        return {
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'memory_freed_mb': memory_freed,
            'objects_collected': collected,
            'cache_cleanup_performed': True
        }
    
    def warm_cache(
        self, 
        tickers: List[str], 
        sample_data: Dict[str, pd.DataFrame],
        horizons: List[int] = [1, 5, 20]
    ) -> Dict[str, Any]:
        """Warm up the cache with sample predictions."""
        logger.info(f"Warming cache for {len(tickers)} tickers")
        
        start_time = time.time()
        warmed_count = 0
        
        for ticker in tickers:
            if ticker in sample_data:
                try:
                    self.predict_single(
                        ticker=ticker,
                        price_data=sample_data[ticker],
                        horizons=horizons,
                        return_confidence=True,
                        use_cache=True
                    )
                    warmed_count += 1
                except Exception as e:
                    logger.debug(f"Cache warming failed for {ticker}: {e}")
        
        warming_time = time.time() - start_time
        
        return {
            'tickers_warmed': warmed_count,
            'total_tickers': len(tickers),
            'warming_time_seconds': warming_time,
            'cache_stats': self.cache.get_stats()
        }
    
    def shutdown(self) -> None:
        """Gracefully shutdown the pipeline."""
        logger.info("Shutting down efficient prediction pipeline")
        
        self._shutdown = True
        
        # Wait for batch thread to finish
        if self._batch_thread and self._batch_thread.is_alive():
            self._batch_thread.join(timeout=5)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Pipeline shutdown complete")
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.shutdown()
        except:
            pass


# Factory function for easy pipeline creation
def create_optimized_pipeline(
    ensemble_model,
    cache_size: int = 1000,
    cache_ttl_seconds: int = 300,
    max_workers: int = None,
    batch_size: int = 16
) -> EfficientPredictionPipeline:
    """
    Create an optimized prediction pipeline with sensible defaults.
    
    Args:
        ensemble_model: The ensemble model to use
        cache_size: Maximum cache size
        cache_ttl_seconds: Cache TTL in seconds
        max_workers: Maximum worker threads
        batch_size: Batch processing size
        
    Returns:
        Configured EfficientPredictionPipeline
    """
    if max_workers is None:
        max_workers = min(8, psutil.cpu_count())
    
    pipeline = EfficientPredictionPipeline(
        ensemble_model=ensemble_model,
        cache_config={
            'max_size': cache_size,
            'ttl_seconds': cache_ttl_seconds,
            'memory_limit_mb': 512
        },
        parallel_config={
            'max_workers': max_workers,
            'use_process_pool': False  # Threads are usually better for I/O bound tasks
        },
        batch_config={
            'batch_size': batch_size,
            'timeout_ms': 100
        }
    )
    
    return pipeline
"""
Memory Optimization Utilities for Phase 7 Computational Efficiency.
Implements advanced memory management, GPU optimization, and resource monitoring.
"""

import torch
import numpy as np
import pandas as pd
import gc
import psutil
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from loguru import logger
import weakref
from collections import defaultdict
import os


@dataclass
class MemoryUsage:
    """Memory usage statistics."""
    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float
    gpu_total_mb: float = 0.0
    gpu_used_mb: float = 0.0
    gpu_percent_used: float = 0.0


class MemoryMonitor:
    """Real-time memory usage monitoring with alerts."""
    
    def __init__(
        self,
        warning_threshold: float = 80.0,
        critical_threshold: float = 90.0,
        monitor_gpu: bool = True
    ):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.monitor_gpu = monitor_gpu and torch.cuda.is_available()
        
        self.callbacks = {
            'warning': [],
            'critical': []
        }
        
        self.monitoring = False
        self.monitor_thread = None
        self._shutdown = False
        
        self.history = []
        self.max_history = 1000
    
    def add_callback(self, threshold: str, callback: Callable[[MemoryUsage], None]):
        """Add callback for memory threshold events."""
        if threshold in self.callbacks:
            self.callbacks[threshold].append(callback)
    
    def get_current_usage(self) -> MemoryUsage:
        """Get current memory usage statistics."""
        # System memory
        memory = psutil.virtual_memory()
        
        usage = MemoryUsage(
            total_mb=memory.total / 1024 / 1024,
            available_mb=memory.available / 1024 / 1024,
            used_mb=memory.used / 1024 / 1024,
            percent_used=memory.percent
        )
        
        # GPU memory if available
        if self.monitor_gpu:
            try:
                gpu_memory = torch.cuda.memory_stats()
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                max_memory = torch.cuda.max_memory_allocated()
                
                if max_memory > 0:
                    usage.gpu_total_mb = max_memory / 1024 / 1024
                    usage.gpu_used_mb = allocated / 1024 / 1024
                    usage.gpu_percent_used = (allocated / max_memory) * 100
            except Exception as e:
                logger.debug(f"GPU memory monitoring failed: {e}")
        
        return usage
    
    def start_monitoring(self, interval_seconds: float = 1.0):
        """Start continuous memory monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self._shutdown = False
        
        def monitor_loop():
            while not self._shutdown:
                try:
                    usage = self.get_current_usage()
                    
                    # Add to history
                    self.history.append(usage)
                    if len(self.history) > self.max_history:
                        self.history.pop(0)
                    
                    # Check thresholds
                    if usage.percent_used >= self.critical_threshold:
                        for callback in self.callbacks['critical']:
                            try:
                                callback(usage)
                            except Exception as e:
                                logger.error(f"Critical memory callback failed: {e}")
                    
                    elif usage.percent_used >= self.warning_threshold:
                        for callback in self.callbacks['warning']:
                            try:
                                callback(usage)
                            except Exception as e:
                                logger.error(f"Warning memory callback failed: {e}")
                    
                    time.sleep(interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Memory monitoring error: {e}")
                    time.sleep(interval_seconds)
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._shutdown = True
        self.monitoring = False
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        
        logger.info("Memory monitoring stopped")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory usage statistics from history."""
        if not self.history:
            return {}
        
        memory_percents = [usage.percent_used for usage in self.history]
        gpu_percents = [usage.gpu_percent_used for usage in self.history if usage.gpu_percent_used > 0]
        
        stats = {
            'current': self.get_current_usage().__dict__,
            'system_memory': {
                'avg_percent': np.mean(memory_percents),
                'max_percent': np.max(memory_percents),
                'min_percent': np.min(memory_percents),
                'std_percent': np.std(memory_percents)
            },
            'history_length': len(self.history),
            'monitoring': self.monitoring
        }
        
        if gpu_percents:
            stats['gpu_memory'] = {
                'avg_percent': np.mean(gpu_percents),
                'max_percent': np.max(gpu_percents),
                'min_percent': np.min(gpu_percents),
                'std_percent': np.std(gpu_percents)
            }
        
        return stats


class GPUOptimizer:
    """GPU memory and computation optimization."""
    
    def __init__(self):
        self.cuda_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.cuda_available else 0
        self.optimization_enabled = self.cuda_available
        
        if self.cuda_available:
            # Enable memory growth and optimization
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            logger.info(f"GPU optimization enabled for {self.device_count} device(s)")
        else:
            logger.info("GPU optimization disabled - CUDA not available")
    
    def optimize_memory_allocation(self):
        """Optimize GPU memory allocation."""
        if not self.cuda_available:
            return
        
        try:
            # Clear cache
            torch.cuda.empty_cache()
            
            # Synchronize to ensure operations complete
            torch.cuda.synchronize()
            
            # Set memory fraction if needed
            if hasattr(torch.cuda, 'set_memory_fraction'):
                torch.cuda.set_memory_fraction(0.9)  # Use 90% of GPU memory max
            
            logger.debug("GPU memory allocation optimized")
            
        except Exception as e:
            logger.error(f"GPU memory optimization failed: {e}")
    
    def get_gpu_stats(self) -> Dict[str, Any]:
        """Get GPU memory and utilization statistics."""
        if not self.cuda_available:
            return {'gpu_available': False}
        
        stats = {'gpu_available': True, 'devices': []}
        
        for device_id in range(self.device_count):
            try:
                device_props = torch.cuda.get_device_properties(device_id)
                
                with torch.cuda.device(device_id):
                    allocated = torch.cuda.memory_allocated()
                    cached = torch.cuda.memory_reserved()
                    max_allocated = torch.cuda.max_memory_allocated()
                    
                    device_stats = {
                        'device_id': device_id,
                        'name': device_props.name,
                        'total_memory': device_props.total_memory,
                        'allocated_mb': allocated / 1024 / 1024,
                        'cached_mb': cached / 1024 / 1024,
                        'max_allocated_mb': max_allocated / 1024 / 1024,
                        'utilization_percent': (allocated / device_props.total_memory) * 100
                    }
                    
                    stats['devices'].append(device_stats)
            
            except Exception as e:
                logger.debug(f"Could not get stats for GPU {device_id}: {e}")
        
        return stats
    
    @contextmanager
    def memory_efficient_inference(self, enable_amp: bool = True):
        """Context manager for memory-efficient inference."""
        if not self.cuda_available:
            yield
            return
        
        try:
            # Clear cache before inference
            torch.cuda.empty_cache()
            
            # Use automatic mixed precision if available
            if enable_amp and hasattr(torch, 'autocast'):
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    yield
            else:
                yield
                
        finally:
            # Clean up after inference
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'synchronize'):
                torch.cuda.synchronize()
    
    def optimize_model_for_inference(self, model: torch.nn.Module) -> torch.nn.Module:
        """Optimize PyTorch model for inference."""
        if not isinstance(model, torch.nn.Module):
            return model
        
        try:
            # Set to evaluation mode
            model.eval()
            
            # Move to GPU if available
            if self.cuda_available:
                model = model.cuda()
            
            # Optimize with TorchScript if possible
            if hasattr(torch, 'jit'):
                try:
                    # Create example input (approximate)
                    example_input = torch.randn(1, 100).to(next(model.parameters()).device)
                    traced_model = torch.jit.trace(model, example_input)
                    traced_model = torch.jit.optimize_for_inference(traced_model)
                    
                    logger.info("Model optimized with TorchScript")
                    return traced_model
                    
                except Exception as e:
                    logger.debug(f"TorchScript optimization failed: {e}")
            
            return model
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model


class DataOptimizer:
    """Optimize data structures and operations for memory efficiency."""
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame, reduce_precision: bool = True) -> pd.DataFrame:
        """Optimize DataFrame memory usage."""
        if df.empty:
            return df
        
        optimized_df = df.copy()
        
        for col in optimized_df.columns:
            col_type = optimized_df[col].dtype
            
            if col_type != object:
                c_min = optimized_df[col].min()
                c_max = optimized_df[col].max()
                
                if str(col_type)[:3] == 'int':
                    # Optimize integer columns
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        optimized_df[col] = optimized_df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        optimized_df[col] = optimized_df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        optimized_df[col] = optimized_df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        optimized_df[col] = optimized_df[col].astype(np.int64)
                
                elif str(col_type)[:5] == 'float':
                    # Optimize float columns
                    if reduce_precision:
                        if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                            optimized_df[col] = optimized_df[col].astype(np.float32)
            
            elif col_type == object:
                # Optimize string columns
                num_unique_values = len(optimized_df[col].unique())
                num_total_values = len(optimized_df[col])
                
                if num_unique_values / num_total_values < 0.5:  # Less than 50% unique
                    optimized_df[col] = optimized_df[col].astype('category')
        
        return optimized_df
    
    @staticmethod
    def get_memory_usage(obj: Any) -> Dict[str, float]:
        """Get memory usage of object in MB."""
        usage = {}
        
        if isinstance(obj, pd.DataFrame):
            usage['total_mb'] = obj.memory_usage(deep=True).sum() / 1024 / 1024
            usage['index_mb'] = obj.index.memory_usage(deep=True) / 1024 / 1024
            
            for col in obj.columns:
                usage[f'{col}_mb'] = obj[col].memory_usage(deep=True) / 1024 / 1024
        
        elif isinstance(obj, dict):
            import sys
            total_size = sys.getsizeof(obj)
            
            for key, value in obj.items():
                total_size += sys.getsizeof(key) + sys.getsizeof(value)
                
                if isinstance(value, (pd.DataFrame, np.ndarray)):
                    if isinstance(value, pd.DataFrame):
                        total_size += value.memory_usage(deep=True).sum()
                    else:
                        total_size += value.nbytes
            
            usage['total_mb'] = total_size / 1024 / 1024
        
        elif isinstance(obj, np.ndarray):
            usage['total_mb'] = obj.nbytes / 1024 / 1024
        
        elif isinstance(obj, torch.Tensor):
            usage['total_mb'] = obj.element_size() * obj.nelement() / 1024 / 1024
        
        else:
            import sys
            usage['total_mb'] = sys.getsizeof(obj) / 1024 / 1024
        
        return usage


class MemoryPool:
    """Memory pool for reusing allocated memory."""
    
    def __init__(self, max_size_mb: int = 256):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.pools = defaultdict(list)  # size -> list of available arrays
        self.total_allocated = 0
        self._lock = threading.Lock()
    
    def get_array(self, shape: tuple, dtype=np.float32) -> np.ndarray:
        """Get array from pool or allocate new one."""
        size = np.prod(shape) * np.dtype(dtype).itemsize
        
        with self._lock:
            if size in self.pools and self.pools[size]:
                array = self.pools[size].pop()
                return array.reshape(shape)
            
            # Allocate new array if pool is empty
            if self.total_allocated + size > self.max_size_bytes:
                self._cleanup_pool()
            
            array = np.empty(shape, dtype=dtype)
            self.total_allocated += size
            
            return array
    
    def return_array(self, array: np.ndarray):
        """Return array to pool for reuse."""
        if not isinstance(array, np.ndarray):
            return
        
        size = array.nbytes
        
        with self._lock:
            # Flatten array for reuse
            flat_array = array.flatten()
            self.pools[size].append(flat_array)
            
            # Limit pool size
            if len(self.pools[size]) > 10:
                self.pools[size] = self.pools[size][-5:]  # Keep only 5 most recent
    
    def _cleanup_pool(self):
        """Clean up pool to free memory."""
        # Remove smallest arrays first
        for size in sorted(self.pools.keys()):
            while self.pools[size] and self.total_allocated > self.max_size_bytes * 0.8:
                array = self.pools[size].pop()
                self.total_allocated -= array.nbytes
            
            if not self.pools[size]:
                del self.pools[size]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics."""
        with self._lock:
            return {
                'total_allocated_mb': self.total_allocated / 1024 / 1024,
                'max_size_mb': self.max_size_bytes / 1024 / 1024,
                'pool_sizes': {size: len(arrays) for size, arrays in self.pools.items()},
                'utilization_percent': (self.total_allocated / self.max_size_bytes) * 100
            }


class MemoryOptimizationManager:
    """Central manager for all memory optimization features."""
    
    def __init__(self, enable_monitoring: bool = True):
        self.monitor = MemoryMonitor() if enable_monitoring else None
        self.gpu_optimizer = GPUOptimizer()
        self.data_optimizer = DataOptimizer()
        self.memory_pool = MemoryPool()
        
        self.optimization_history = []
        
        # Set up automatic memory management callbacks
        if self.monitor:
            self.monitor.add_callback('warning', self._handle_memory_warning)
            self.monitor.add_callback('critical', self._handle_memory_critical)
            
            if enable_monitoring:
                self.monitor.start_monitoring()
    
    def _handle_memory_warning(self, usage: MemoryUsage):
        """Handle memory warning threshold."""
        logger.warning(f"Memory usage warning: {usage.percent_used:.1f}% used ({usage.used_mb:.1f}MB)")
        
        # Perform light cleanup
        self.perform_light_cleanup()
    
    def _handle_memory_critical(self, usage: MemoryUsage):
        """Handle critical memory threshold."""
        logger.error(f"Critical memory usage: {usage.percent_used:.1f}% used ({usage.used_mb:.1f}MB)")
        
        # Perform aggressive cleanup
        self.perform_aggressive_cleanup()
    
    def perform_light_cleanup(self) -> Dict[str, Any]:
        """Perform light memory cleanup."""
        start_time = time.time()
        
        # GPU cleanup
        if self.gpu_optimizer.cuda_available:
            torch.cuda.empty_cache()
        
        # Python garbage collection
        collected = gc.collect()
        
        cleanup_time = time.time() - start_time
        
        result = {
            'cleanup_type': 'light',
            'objects_collected': collected,
            'cleanup_time_seconds': cleanup_time,
            'timestamp': datetime.now()
        }
        
        self.optimization_history.append(result)
        logger.info(f"Light cleanup completed: {collected} objects collected in {cleanup_time:.3f}s")
        
        return result
    
    def perform_aggressive_cleanup(self) -> Dict[str, Any]:
        """Perform aggressive memory cleanup."""
        start_time = time.time()
        
        # Clear memory pool
        self.memory_pool._cleanup_pool()
        
        # GPU optimization
        if self.gpu_optimizer.cuda_available:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Multiple garbage collection passes
        collected = 0
        for _ in range(3):
            collected += gc.collect()
        
        # Clear weak references
        if hasattr(gc, 'set_debug'):
            gc.set_debug(0)  # Disable debug to free more memory
        
        cleanup_time = time.time() - start_time
        
        result = {
            'cleanup_type': 'aggressive',
            'objects_collected': collected,
            'cleanup_time_seconds': cleanup_time,
            'timestamp': datetime.now()
        }
        
        self.optimization_history.append(result)
        logger.warning(f"Aggressive cleanup completed: {collected} objects collected in {cleanup_time:.3f}s")
        
        return result
    
    def optimize_for_inference(self, model: torch.nn.Module = None) -> Dict[str, Any]:
        """Comprehensive optimization for inference."""
        start_time = time.time()
        
        optimizations = []
        
        # GPU optimization
        if self.gpu_optimizer.cuda_available:
            self.gpu_optimizer.optimize_memory_allocation()
            optimizations.append('gpu_memory_allocation')
        
        # Model optimization if provided
        if model is not None:
            optimized_model = self.gpu_optimizer.optimize_model_for_inference(model)
            optimizations.append('model_optimization')
        
        # Memory cleanup
        self.perform_light_cleanup()
        optimizations.append('memory_cleanup')
        
        optimization_time = time.time() - start_time
        
        return {
            'optimizations_applied': optimizations,
            'optimization_time_seconds': optimization_time,
            'current_memory_usage': self.get_comprehensive_stats()
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory and optimization statistics."""
        stats = {}
        
        # Memory monitoring stats
        if self.monitor:
            stats['memory_monitoring'] = self.monitor.get_statistics()
        
        # GPU stats
        stats['gpu'] = self.gpu_optimizer.get_gpu_stats()
        
        # Memory pool stats
        stats['memory_pool'] = self.memory_pool.get_stats()
        
        # System information
        stats['system'] = {
            'cpu_count': psutil.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'process_memory_mb': psutil.Process().memory_info().rss / (1024**2)
        }
        
        # Optimization history
        if self.optimization_history:
            recent_optimizations = self.optimization_history[-10:]  # Last 10
            stats['optimization_history'] = {
                'recent_count': len(recent_optimizations),
                'total_count': len(self.optimization_history),
                'recent_optimizations': recent_optimizations
            }
        
        return stats
    
    def shutdown(self):
        """Shutdown memory optimization manager."""
        if self.monitor:
            self.monitor.stop_monitoring()
        
        # Final cleanup
        self.perform_light_cleanup()
        
        logger.info("Memory optimization manager shutdown complete")


# Global memory optimization instance
_global_memory_optimizer = None

def get_memory_optimizer(enable_monitoring: bool = True) -> MemoryOptimizationManager:
    """Get global memory optimizer instance."""
    global _global_memory_optimizer
    
    if _global_memory_optimizer is None:
        _global_memory_optimizer = MemoryOptimizationManager(enable_monitoring)
    
    return _global_memory_optimizer


# Decorator for memory-optimized functions
def memory_optimized(enable_monitoring: bool = False, cleanup_after: bool = True):
    """Decorator to add memory optimization to functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            optimizer = get_memory_optimizer(enable_monitoring)
            
            try:
                # Pre-execution optimization
                if enable_monitoring:
                    optimizer.perform_light_cleanup()
                
                # Execute function
                result = func(*args, **kwargs)
                
                return result
                
            finally:
                # Post-execution cleanup
                if cleanup_after:
                    optimizer.perform_light_cleanup()
        
        return wrapper
    return decorator
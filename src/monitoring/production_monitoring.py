"""
Production Monitoring and Infrastructure for Phase 7 Deployment.
Comprehensive monitoring, alerting, and production readiness infrastructure.
"""

import asyncio
import threading
import time
import psutil
import logging
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from pathlib import Path
from collections import deque, defaultdict
import numpy as np
import pandas as pd
from loguru import logger
import socket
import requests
import os
import sys
from contextlib import asynccontextmanager

# Metrics and monitoring
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, Summary
from prometheus_client.core import CollectorRegistry


@dataclass
class PerformanceMetrics:
    """Performance metrics for monitoring."""
    timestamp: datetime
    prediction_latency_ms: float
    cache_hit_rate: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: float = 0.0
    active_connections: int = 0
    requests_per_second: float = 0.0
    error_rate: float = 0.0
    model_accuracy: Optional[float] = None


@dataclass
class SystemHealth:
    """System health status."""
    overall_status: str  # healthy, degraded, unhealthy
    uptime_seconds: float
    last_health_check: datetime
    component_status: Dict[str, str]
    active_alerts: List[str]
    performance_score: float  # 0-100


@dataclass
class AlertRule:
    """Alert rule configuration."""
    name: str
    condition: str  # Python expression
    threshold: float
    duration_seconds: int
    severity: str  # critical, warning, info
    message_template: str
    enabled: bool = True
    cooldown_seconds: int = 300  # 5 minutes


class PrometheusMetrics:
    """Prometheus metrics collector."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or prometheus_client.REGISTRY
        
        # Request metrics
        self.request_count = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint'],
            registry=self.registry
        )
        
        # Prediction metrics
        self.prediction_count = Counter(
            'predictions_total',
            'Total predictions made',
            ['ticker', 'horizon'],
            registry=self.registry
        )
        
        self.prediction_latency = Histogram(
            'prediction_latency_seconds',
            'Prediction processing latency',
            ['ticker'],
            registry=self.registry
        )
        
        self.cache_hits = Counter(
            'cache_hits_total',
            'Cache hits',
            ['cache_type'],
            registry=self.registry
        )
        
        # System metrics
        self.memory_usage = Gauge(
            'memory_usage_bytes',
            'Memory usage in bytes',
            registry=self.registry
        )
        
        self.cpu_usage = Gauge(
            'cpu_usage_percent',
            'CPU usage percentage',
            registry=self.registry
        )
        
        self.gpu_usage = Gauge(
            'gpu_usage_percent',
            'GPU usage percentage',
            registry=self.registry
        )
        
        # Model performance metrics
        self.model_accuracy = Gauge(
            'model_accuracy',
            'Model prediction accuracy',
            ['model_name', 'horizon'],
            registry=self.registry
        )
        
        self.ensemble_weight = Gauge(
            'ensemble_weight',
            'Current ensemble model weights',
            ['model_name'],
            registry=self.registry
        )
        
        # Business metrics
        self.portfolio_value = Gauge(
            'portfolio_value_usd',
            'Current portfolio value in USD',
            registry=self.registry
        )
        
        self.active_positions = Gauge(
            'active_positions',
            'Number of active positions',
            registry=self.registry
        )


class AlertManager:
    """Alert management and notification system."""
    
    def __init__(
        self,
        rules: List[AlertRule],
        notification_handlers: Optional[List[Callable]] = None
    ):
        self.rules = rules
        self.notification_handlers = notification_handlers or []
        
        # Alert state tracking
        self.active_alerts = {}  # rule_name -> alert_start_time
        self.alert_history = deque(maxlen=1000)
        self.cooldown_tracker = {}  # rule_name -> last_alert_time
        
        self._monitoring = False
        self._monitor_thread = None
        
        logger.info(f"AlertManager initialized with {len(self.rules)} rules")
    
    def add_notification_handler(self, handler: Callable[[str, AlertRule, Dict], None]):
        """Add notification handler for alerts."""
        self.notification_handlers.append(handler)
    
    def start_monitoring(self, metrics_provider: Callable[[], Dict[str, Any]]):
        """Start alert monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._metrics_provider = metrics_provider
        
        def monitor_loop():
            while self._monitoring:
                try:
                    self._check_alert_rules()
                    time.sleep(5)  # Check every 5 seconds
                except Exception as e:
                    logger.error(f"Alert monitoring error: {e}")
                    time.sleep(10)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop alert monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        
        logger.info("Alert monitoring stopped")
    
    def _check_alert_rules(self):
        """Check all alert rules against current metrics."""
        current_metrics = self._metrics_provider()
        current_time = datetime.now()
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            try:
                # Evaluate rule condition
                condition_met = self._evaluate_condition(rule.condition, current_metrics)
                
                if condition_met:
                    # Check if alert should trigger
                    if rule.name not in self.active_alerts:
                        self.active_alerts[rule.name] = current_time
                    else:
                        # Check duration threshold
                        alert_duration = (current_time - self.active_alerts[rule.name]).total_seconds()
                        
                        if alert_duration >= rule.duration_seconds:
                            # Trigger alert if not in cooldown
                            if self._can_send_alert(rule.name, current_time):
                                self._send_alert(rule, current_metrics, current_time)
                
                else:
                    # Condition not met, clear active alert
                    if rule.name in self.active_alerts:
                        del self.active_alerts[rule.name]
            
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.name}: {e}")
    
    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Safely evaluate alert condition."""
        try:
            # Create safe evaluation context
            context = {
                'metrics': metrics,
                'np': np,
                'abs': abs,
                'max': max,
                'min': min,
                'len': len
            }
            
            # Add metrics directly to context for easier access
            for key, value in metrics.items():
                if isinstance(key, str) and key.replace('_', '').isalnum():
                    context[key] = value
            
            return bool(eval(condition, {"__builtins__": {}}, context))
        
        except Exception as e:
            logger.debug(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _can_send_alert(self, rule_name: str, current_time: datetime) -> bool:
        """Check if alert can be sent (not in cooldown)."""
        rule = next(r for r in self.rules if r.name == rule_name)
        
        if rule_name not in self.cooldown_tracker:
            return True
        
        last_alert_time = self.cooldown_tracker[rule_name]
        time_since_last = (current_time - last_alert_time).total_seconds()
        
        return time_since_last >= rule.cooldown_seconds
    
    def _send_alert(self, rule: AlertRule, metrics: Dict[str, Any], timestamp: datetime):
        """Send alert notification."""
        try:
            # Format alert message
            message = self._format_alert_message(rule, metrics)
            
            alert_data = {
                'rule_name': rule.name,
                'severity': rule.severity,
                'message': message,
                'timestamp': timestamp,
                'metrics': metrics
            }
            
            # Send to all notification handlers
            for handler in self.notification_handlers:
                try:
                    handler(rule.severity, rule, alert_data)
                except Exception as e:
                    logger.error(f"Notification handler error: {e}")
            
            # Update tracking
            self.cooldown_tracker[rule.name] = timestamp
            self.alert_history.append(alert_data)
            
            logger.warning(f"Alert triggered: {rule.name} - {message}")
            
        except Exception as e:
            logger.error(f"Error sending alert for {rule.name}: {e}")
    
    def _format_alert_message(self, rule: AlertRule, metrics: Dict[str, Any]) -> str:
        """Format alert message with metrics."""
        try:
            return rule.message_template.format(**metrics)
        except Exception:
            return rule.message_template
    
    def get_alert_status(self) -> Dict[str, Any]:
        """Get current alert status."""
        return {
            'active_alerts': len(self.active_alerts),
            'active_alert_names': list(self.active_alerts.keys()),
            'alert_history_count': len(self.alert_history),
            'recent_alerts': list(self.alert_history)[-10:],
            'rules_enabled': sum(1 for r in self.rules if r.enabled),
            'total_rules': len(self.rules)
        }


class ProductionMonitor:
    """Main production monitoring system."""
    
    def __init__(
        self,
        ensemble_model=None,
        prediction_pipeline=None,
        memory_optimizer=None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.ensemble_model = ensemble_model
        self.prediction_pipeline = prediction_pipeline
        self.memory_optimizer = memory_optimizer
        
        self.config = config or {}
        
        # Monitoring components
        self.prometheus_metrics = PrometheusMetrics()
        self.alert_manager = self._create_alert_manager()
        
        # Performance tracking
        self.performance_history = deque(maxlen=1440)  # 24 hours at 1-minute intervals
        self.system_start_time = datetime.now()
        
        # Health checking
        self.health_checkers = []
        self.last_health_check = None
        self.health_status = SystemHealth(
            overall_status='starting',
            uptime_seconds=0,
            last_health_check=datetime.now(),
            component_status={},
            active_alerts=[],
            performance_score=0.0
        )
        
        # Monitoring state
        self._monitoring = False
        self._monitor_thread = None
        
        logger.info("ProductionMonitor initialized")
    
    def _create_alert_manager(self) -> AlertManager:
        """Create alert manager with default rules."""
        default_rules = [
            AlertRule(
                name="high_memory_usage",
                condition="memory_usage_mb > 2000",  # 2GB
                threshold=2000,
                duration_seconds=60,
                severity="warning",
                message_template="High memory usage: {memory_usage_mb:.1f}MB"
            ),
            
            AlertRule(
                name="critical_memory_usage",
                condition="memory_usage_mb > 4000",  # 4GB
                threshold=4000,
                duration_seconds=30,
                severity="critical",
                message_template="Critical memory usage: {memory_usage_mb:.1f}MB"
            ),
            
            AlertRule(
                name="high_prediction_latency",
                condition="prediction_latency_ms > 5000",  # 5 seconds
                threshold=5000,
                duration_seconds=120,
                severity="warning",
                message_template="High prediction latency: {prediction_latency_ms:.1f}ms"
            ),
            
            AlertRule(
                name="low_cache_hit_rate",
                condition="cache_hit_rate < 0.3",  # Below 30%
                threshold=0.3,
                duration_seconds=300,
                severity="info",
                message_template="Low cache hit rate: {cache_hit_rate:.1%}"
            ),
            
            AlertRule(
                name="high_error_rate",
                condition="error_rate > 0.05",  # Above 5%
                threshold=0.05,
                duration_seconds=60,
                severity="critical",
                message_template="High error rate: {error_rate:.1%}"
            ),
            
            AlertRule(
                name="model_accuracy_degradation",
                condition="model_accuracy is not None and model_accuracy < 0.6",  # Below 60%
                threshold=0.6,
                duration_seconds=600,  # 10 minutes
                severity="warning",
                message_template="Model accuracy degraded: {model_accuracy:.1%}"
            )
        ]
        
        alert_manager = AlertManager(default_rules)
        
        # Add notification handlers
        alert_manager.add_notification_handler(self._log_alert_handler)
        
        # Add webhook handler if configured
        webhook_url = self.config.get('alert_webhook_url')
        if webhook_url:
            alert_manager.add_notification_handler(
                lambda severity, rule, data: self._webhook_alert_handler(webhook_url, severity, rule, data)
            )
        
        return alert_manager
    
    def start_monitoring(self):
        """Start comprehensive monitoring."""
        if self._monitoring:
            return
        
        logger.info("Starting production monitoring...")
        
        self._monitoring = True
        
        # Start alert monitoring
        self.alert_manager.start_monitoring(self._get_current_metrics)
        
        # Start performance monitoring
        def monitor_loop():
            while self._monitoring:
                try:
                    # Collect performance metrics
                    metrics = self._collect_performance_metrics()
                    self.performance_history.append(metrics)
                    
                    # Update Prometheus metrics
                    self._update_prometheus_metrics(metrics)
                    
                    # Update health status
                    self._update_health_status()
                    
                    time.sleep(60)  # Collect every minute
                    
                except Exception as e:
                    logger.error(f"Monitoring loop error: {e}")
                    time.sleep(60)
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        logger.info("Production monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        logger.info("Stopping production monitoring...")
        
        self._monitoring = False
        
        if self.alert_manager:
            self.alert_manager.stop_monitoring()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
        
        logger.info("Production monitoring stopped")
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        current_time = datetime.now()
        
        # System metrics
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024  # MB
        cpu_usage = process.cpu_percent()
        
        # GPU metrics (if available)
        gpu_usage = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.utilization()
        except Exception:
            pass
        
        # Prediction pipeline metrics
        prediction_latency = 0.0
        cache_hit_rate = 0.0
        
        if self.prediction_pipeline:
            stats = self.prediction_pipeline.get_performance_stats()
            pipeline_stats = stats.get('pipeline_stats', {})
            cache_stats = stats.get('cache_stats', {})
            
            prediction_latency = pipeline_stats.get('avg_processing_time', 0)
            cache_hit_rate = cache_stats.get('hit_rate', 0)
        
        # Model accuracy (simplified - would need actual accuracy tracking)
        model_accuracy = None
        if hasattr(self, '_recent_accuracy'):
            model_accuracy = self._recent_accuracy
        
        return PerformanceMetrics(
            timestamp=current_time,
            prediction_latency_ms=prediction_latency,
            cache_hit_rate=cache_hit_rate,
            memory_usage_mb=memory_usage,
            cpu_usage_percent=cpu_usage,
            gpu_usage_percent=gpu_usage,
            model_accuracy=model_accuracy
        )
    
    def _update_prometheus_metrics(self, metrics: PerformanceMetrics):
        """Update Prometheus metrics."""
        self.prometheus_metrics.memory_usage.set(metrics.memory_usage_mb * 1024 * 1024)  # Convert to bytes
        self.prometheus_metrics.cpu_usage.set(metrics.cpu_usage_percent)
        self.prometheus_metrics.gpu_usage.set(metrics.gpu_usage_percent)
    
    def _update_health_status(self):
        """Update overall system health status."""
        current_time = datetime.now()
        uptime = (current_time - self.system_start_time).total_seconds()
        
        # Check component health
        component_status = {}
        
        if self.ensemble_model:
            try:
                ensemble_status = self.ensemble_model.get_ensemble_status()
                component_status['ensemble'] = 'healthy' if ensemble_status.get('enabled') else 'degraded'
            except Exception:
                component_status['ensemble'] = 'unhealthy'
        
        if self.prediction_pipeline:
            try:
                pipeline_stats = self.prediction_pipeline.get_performance_stats()
                avg_time = pipeline_stats.get('pipeline_stats', {}).get('avg_processing_time', 0)
                component_status['prediction_pipeline'] = 'healthy' if avg_time < 5000 else 'degraded'
            except Exception:
                component_status['prediction_pipeline'] = 'unhealthy'
        
        if self.memory_optimizer:
            try:
                mem_stats = self.memory_optimizer.get_comprehensive_stats()
                mem_usage = mem_stats.get('system', {}).get('process_memory_mb', 0)
                component_status['memory'] = 'healthy' if mem_usage < 2000 else 'degraded'
            except Exception:
                component_status['memory'] = 'unhealthy'
        
        # Calculate overall status
        unhealthy_count = sum(1 for status in component_status.values() if status == 'unhealthy')
        degraded_count = sum(1 for status in component_status.values() if status == 'degraded')
        
        if unhealthy_count > 0:
            overall_status = 'unhealthy'
        elif degraded_count > len(component_status) / 2:
            overall_status = 'degraded'
        else:
            overall_status = 'healthy'
        
        # Calculate performance score
        performance_score = self._calculate_performance_score()
        
        # Get active alerts
        alert_status = self.alert_manager.get_alert_status()
        active_alerts = alert_status.get('active_alert_names', [])
        
        self.health_status = SystemHealth(
            overall_status=overall_status,
            uptime_seconds=uptime,
            last_health_check=current_time,
            component_status=component_status,
            active_alerts=active_alerts,
            performance_score=performance_score
        )
        
        self.last_health_check = current_time
    
    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        if not self.performance_history:
            return 50.0  # Default neutral score
        
        recent_metrics = list(self.performance_history)[-10:]  # Last 10 minutes
        
        scores = []
        
        for metrics in recent_metrics:
            score = 100.0
            
            # Memory usage penalty
            if metrics.memory_usage_mb > 2000:
                score -= min(30, (metrics.memory_usage_mb - 2000) / 100)
            
            # Latency penalty
            if metrics.prediction_latency_ms > 1000:
                score -= min(25, (metrics.prediction_latency_ms - 1000) / 200)
            
            # Cache hit rate bonus/penalty
            if metrics.cache_hit_rate > 0.7:
                score += 10
            elif metrics.cache_hit_rate < 0.3:
                score -= 15
            
            # CPU usage penalty
            if metrics.cpu_usage_percent > 80:
                score -= min(20, (metrics.cpu_usage_percent - 80) / 5)
            
            # Model accuracy bonus/penalty
            if metrics.model_accuracy is not None:
                if metrics.model_accuracy > 0.7:
                    score += 15
                elif metrics.model_accuracy < 0.5:
                    score -= 25
            
            scores.append(max(0, min(100, score)))
        
        return np.mean(scores)
    
    def _get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics for alert evaluation."""
        if not self.performance_history:
            return {}
        
        current = self.performance_history[-1]
        
        return {
            'memory_usage_mb': current.memory_usage_mb,
            'cpu_usage_percent': current.cpu_usage_percent,
            'gpu_usage_percent': current.gpu_usage_percent,
            'prediction_latency_ms': current.prediction_latency_ms,
            'cache_hit_rate': current.cache_hit_rate,
            'model_accuracy': current.model_accuracy,
            'error_rate': 0.0,  # Would need actual error tracking
            'uptime_seconds': (datetime.now() - self.system_start_time).total_seconds()
        }
    
    def _log_alert_handler(self, severity: str, rule: AlertRule, alert_data: Dict[str, Any]):
        """Log alert notifications."""
        message = f"[{severity.upper()}] {alert_data['message']} (Rule: {rule.name})"
        
        if severity == 'critical':
            logger.error(message)
        elif severity == 'warning':
            logger.warning(message)
        else:
            logger.info(message)
    
    def _webhook_alert_handler(
        self, 
        webhook_url: str, 
        severity: str, 
        rule: AlertRule, 
        alert_data: Dict[str, Any]
    ):
        """Send alert to webhook."""
        try:
            payload = {
                'severity': severity,
                'rule_name': rule.name,
                'message': alert_data['message'],
                'timestamp': alert_data['timestamp'].isoformat(),
                'system': 'phase7-trading-system',
                'metrics': alert_data.get('metrics', {})
            }
            
            response = requests.post(
                webhook_url,
                json=payload,
                timeout=10,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code != 200:
                logger.error(f"Webhook alert failed: {response.status_code}")
            
        except Exception as e:
            logger.error(f"Webhook alert error: {e}")
    
    def get_monitoring_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        current_metrics = self._get_current_metrics()
        alert_status = self.alert_manager.get_alert_status()
        
        # Performance history for charts
        history_data = []
        for metrics in list(self.performance_history)[-60:]:  # Last hour
            history_data.append({
                'timestamp': metrics.timestamp.isoformat(),
                'memory_usage_mb': metrics.memory_usage_mb,
                'cpu_usage_percent': metrics.cpu_usage_percent,
                'prediction_latency_ms': metrics.prediction_latency_ms,
                'cache_hit_rate': metrics.cache_hit_rate
            })
        
        return {
            'health_status': asdict(self.health_status),
            'current_metrics': current_metrics,
            'performance_history': history_data,
            'alert_status': alert_status,
            'system_info': {
                'start_time': self.system_start_time.isoformat(),
                'python_version': sys.version,
                'platform': sys.platform,
                'hostname': socket.gethostname()
            }
        }
    
    def export_metrics_for_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        return prometheus_client.generate_latest(self.prometheus_metrics.registry)
    
    def save_monitoring_report(self, filepath: str):
        """Save comprehensive monitoring report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'system_uptime_seconds': (datetime.now() - self.system_start_time).total_seconds(),
            'health_status': asdict(self.health_status),
            'performance_summary': self._generate_performance_summary(),
            'alert_summary': self.alert_manager.get_alert_status(),
            'configuration': self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Monitoring report saved to {filepath}")
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Generate performance summary statistics."""
        if not self.performance_history:
            return {}
        
        metrics_list = list(self.performance_history)
        
        # Calculate statistics
        memory_values = [m.memory_usage_mb for m in metrics_list]
        latency_values = [m.prediction_latency_ms for m in metrics_list]
        cache_rates = [m.cache_hit_rate for m in metrics_list]
        
        return {
            'memory_usage': {
                'avg_mb': np.mean(memory_values),
                'max_mb': np.max(memory_values),
                'min_mb': np.min(memory_values),
                'std_mb': np.std(memory_values)
            },
            'prediction_latency': {
                'avg_ms': np.mean(latency_values),
                'max_ms': np.max(latency_values),
                'min_ms': np.min(latency_values),
                'p95_ms': np.percentile(latency_values, 95)
            },
            'cache_performance': {
                'avg_hit_rate': np.mean(cache_rates),
                'min_hit_rate': np.min(cache_rates),
                'max_hit_rate': np.max(cache_rates)
            },
            'data_points': len(metrics_list),
            'time_range_hours': (metrics_list[-1].timestamp - metrics_list[0].timestamp).total_seconds() / 3600 if len(metrics_list) > 1 else 0
        }


# Convenience functions
def create_production_monitor(
    ensemble_model=None,
    prediction_pipeline=None,
    memory_optimizer=None,
    config: Optional[Dict[str, Any]] = None
) -> ProductionMonitor:
    """Create production monitor with default configuration."""
    return ProductionMonitor(
        ensemble_model=ensemble_model,
        prediction_pipeline=prediction_pipeline,
        memory_optimizer=memory_optimizer,
        config=config or {}
    )


def setup_logging_for_production():
    """Set up production-ready logging configuration."""
    from loguru import logger
    import sys
    
    # Remove default handler
    logger.remove()
    
    # Add structured logging for production
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="INFO",
        serialize=False
    )
    
    # Add file logging with rotation
    logger.add(
        "logs/phase7_production.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
        level="INFO",
        rotation="100 MB",
        retention="7 days",
        compression="gz"
    )
    
    # Add error-only file logging
    logger.add(
        "logs/phase7_errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message} | {extra}",
        level="ERROR",
        rotation="50 MB",
        retention="30 days"
    )
    
    logger.info("Production logging configured")


# Global monitoring instance
_global_monitor = None

def get_production_monitor() -> Optional[ProductionMonitor]:
    """Get global production monitor instance."""
    return _global_monitor


def initialize_production_monitoring(
    ensemble_model=None,
    prediction_pipeline=None,
    memory_optimizer=None,
    config: Optional[Dict[str, Any]] = None
) -> ProductionMonitor:
    """Initialize global production monitoring."""
    global _global_monitor
    
    if _global_monitor is None:
        _global_monitor = create_production_monitor(
            ensemble_model=ensemble_model,
            prediction_pipeline=prediction_pipeline,
            memory_optimizer=memory_optimizer,
            config=config
        )
        
        _global_monitor.start_monitoring()
    
    return _global_monitor


def shutdown_production_monitoring():
    """Shutdown global production monitoring."""
    global _global_monitor
    
    if _global_monitor is not None:
        _global_monitor.stop_monitoring()
        _global_monitor = None
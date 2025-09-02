#!/usr/bin/env python3
"""
Model Performance Monitoring and Degradation Detection
Comprehensive monitoring system for AI model performance and reliability.
"""

import sys
import json
import pickle
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

import numpy as np
import pandas as pd
from loguru import logger

# Add src to path for imports  
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import config

warnings.filterwarnings('ignore')


class ModelHealth(Enum):
    """Model health status levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"
    FAILED = "failed"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    model_name: str
    timestamp: datetime
    
    # Accuracy metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # Prediction quality
    mean_absolute_error: float
    mean_squared_error: float
    directional_accuracy: float  # For stock predictions
    
    # Reliability metrics
    prediction_confidence: float
    prediction_consistency: float
    data_coverage: float
    
    # Performance metrics
    inference_time_ms: float
    memory_usage_mb: float
    
    # Model-specific metrics
    sharpe_ratio: Optional[float] = None
    hit_rate: Optional[float] = None
    max_drawdown: Optional[float] = None


@dataclass
class ModelAlert:
    """Model monitoring alert."""
    timestamp: datetime
    model_name: str
    severity: AlertSeverity
    alert_type: str
    message: str
    metric_name: str
    current_value: float
    threshold_value: float
    trend: str  # "improving", "stable", "degrading"
    recommended_action: str


@dataclass
class ModelBaselineMetrics:
    """Baseline metrics for model comparison."""
    model_name: str
    established_date: datetime
    baseline_accuracy: float
    baseline_mae: float
    baseline_mse: float
    baseline_confidence: float
    baseline_inference_time: float
    expected_ranges: Dict[str, Tuple[float, float]]  # metric_name: (min, max)


class ModelMonitor:
    """Comprehensive model performance monitoring system."""
    
    def __init__(self, monitoring_config: Optional[Dict] = None):
        """Initialize model monitor.
        
        Args:
            monitoring_config: Configuration for monitoring thresholds
        """
        self.config = monitoring_config or self._get_default_config()
        
        # Storage paths
        self.metrics_file = Path("model_metrics.json")
        self.baselines_file = Path("model_baselines.json") 
        self.alerts_file = Path("model_alerts.json")
        
        # In-memory storage
        self.metrics_history: Dict[str, List[ModelMetrics]] = {}
        self.model_baselines: Dict[str, ModelBaselineMetrics] = {}
        self.alerts: List[ModelAlert] = []
        
        # Load existing data
        self._load_historical_data()
        
        logger.info("Model monitor initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default monitoring configuration."""
        return {
            # Degradation thresholds (relative to baseline)
            'accuracy_degradation_threshold': 0.10,      # 10% drop triggers warning
            'accuracy_critical_threshold': 0.20,         # 20% drop triggers critical
            'mae_increase_threshold': 0.15,              # 15% increase triggers warning
            'confidence_drop_threshold': 0.10,           # 10% confidence drop
            'inference_time_increase_threshold': 2.0,    # 2x slower triggers warning
            'memory_increase_threshold': 1.5,            # 1.5x memory usage
            
            # Model-specific thresholds
            'min_directional_accuracy': 0.55,            # 55% minimum for stock predictions
            'min_hit_rate': 0.50,                        # 50% minimum hit rate
            'max_acceptable_drawdown': 0.25,             # 25% maximum drawdown
            'min_sharpe_ratio': 0.8,                     # 0.8 minimum Sharpe ratio
            
            # Monitoring intervals
            'evaluation_frequency_hours': 6,              # Every 6 hours
            'baseline_update_days': 30,                   # Update baseline monthly
            'alert_cooldown_hours': 2,                    # 2 hours between similar alerts
            'metric_retention_days': 90,                  # Keep 90 days of metrics
            
            # Quality gates
            'min_data_coverage': 0.8,                     # 80% data availability required
            'max_prediction_variance': 0.5,               # Consistency check
            'confidence_threshold': 0.7,                  # Minimum confidence for predictions
            
            # Performance bounds
            'max_inference_time_ms': 5000,                # 5 second max inference
            'max_memory_usage_mb': 2048,                  # 2GB max memory usage
        }
    
    def register_model_baseline(
        self,
        model_name: str,
        baseline_metrics: Dict[str, float],
        expected_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """Register baseline performance metrics for a model.
        
        Args:
            model_name: Name of the model
            baseline_metrics: Dictionary of baseline metric values
            expected_ranges: Expected ranges for each metric
        """
        if expected_ranges is None:
            expected_ranges = self._calculate_expected_ranges(baseline_metrics)
        
        baseline = ModelBaselineMetrics(
            model_name=model_name,
            established_date=datetime.now(),
            baseline_accuracy=baseline_metrics.get('accuracy', 0.0),
            baseline_mae=baseline_metrics.get('mae', float('inf')),
            baseline_mse=baseline_metrics.get('mse', float('inf')),
            baseline_confidence=baseline_metrics.get('confidence', 0.0),
            baseline_inference_time=baseline_metrics.get('inference_time_ms', 1000.0),
            expected_ranges=expected_ranges
        )
        
        self.model_baselines[model_name] = baseline
        self._save_baselines()
        
        logger.info(f"Baseline registered for model {model_name}")
    
    def record_model_performance(
        self,
        model_name: str,
        predictions: np.ndarray,
        actuals: np.ndarray,
        prediction_metadata: Dict[str, Any],
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> ModelMetrics:
        """Record model performance metrics.
        
        Args:
            model_name: Name of the model
            predictions: Model predictions array
            actuals: Actual values array
            prediction_metadata: Additional prediction context
            performance_metrics: Pre-calculated performance metrics
            
        Returns:
            ModelMetrics object with calculated metrics
        """
        timestamp = datetime.now()
        
        # Calculate core metrics
        metrics = self._calculate_metrics(
            predictions, actuals, prediction_metadata, performance_metrics
        )
        
        # Create metrics object
        model_metrics = ModelMetrics(
            model_name=model_name,
            timestamp=timestamp,
            **metrics
        )
        
        # Store metrics
        if model_name not in self.metrics_history:
            self.metrics_history[model_name] = []
        
        self.metrics_history[model_name].append(model_metrics)
        
        # Cleanup old metrics
        self._cleanup_old_metrics()
        
        # Save to disk
        self._save_metrics()
        
        # Check for degradation
        alerts = self._check_for_degradation(model_metrics)
        if alerts:
            self.alerts.extend(alerts)
            self._save_alerts()
        
        logger.debug(f"Recorded performance for {model_name}: accuracy={metrics['accuracy']:.3f}")
        
        return model_metrics
    
    def _calculate_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        metadata: Dict[str, Any],
        performance_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Calculate comprehensive model metrics."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        
        metrics = {}
        
        # Handle different types of predictions (classification vs regression)
        if len(np.unique(actuals)) <= 10:  # Classification
            # Convert continuous predictions to classes if needed
            if predictions.dtype in [np.float32, np.float64]:
                pred_classes = (predictions > 0.5).astype(int) if len(np.unique(actuals)) == 2 else np.round(predictions).astype(int)
            else:
                pred_classes = predictions
            
            metrics['accuracy'] = accuracy_score(actuals, pred_classes)
            metrics['precision'] = precision_score(actuals, pred_classes, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(actuals, pred_classes, average='weighted', zero_division=0)
            metrics['f1_score'] = f1_score(actuals, pred_classes, average='weighted', zero_division=0)
        else:  # Regression
            metrics['accuracy'] = self._calculate_regression_accuracy(predictions, actuals)
            metrics['precision'] = 0.0  # Not applicable for regression
            metrics['recall'] = 0.0     # Not applicable for regression
            metrics['f1_score'] = 0.0   # Not applicable for regression
        
        # Error metrics
        metrics['mean_absolute_error'] = mean_absolute_error(actuals, predictions)
        metrics['mean_squared_error'] = mean_squared_error(actuals, predictions)
        
        # Directional accuracy for financial predictions
        metrics['directional_accuracy'] = self._calculate_directional_accuracy(predictions, actuals)
        
        # Prediction quality metrics
        metrics['prediction_confidence'] = metadata.get('confidence', np.mean(np.abs(predictions)))
        metrics['prediction_consistency'] = self._calculate_consistency(predictions)
        metrics['data_coverage'] = metadata.get('data_coverage', 1.0)
        
        # Performance metrics
        metrics['inference_time_ms'] = metadata.get('inference_time_ms', 0.0)
        metrics['memory_usage_mb'] = metadata.get('memory_usage_mb', 0.0)
        
        # Trading-specific metrics (if available)
        if performance_metrics:
            metrics['sharpe_ratio'] = performance_metrics.get('sharpe_ratio')
            metrics['hit_rate'] = performance_metrics.get('hit_rate')
            metrics['max_drawdown'] = performance_metrics.get('max_drawdown')
        
        return metrics
    
    def _calculate_regression_accuracy(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate accuracy for regression problems."""
        # Use MAPE (Mean Absolute Percentage Error) inverted to accuracy
        mask = actuals != 0  # Avoid division by zero
        if np.sum(mask) == 0:
            return 0.0
        
        mape = np.mean(np.abs((actuals[mask] - predictions[mask]) / actuals[mask]))
        accuracy = max(0.0, 1.0 - mape)
        return min(1.0, accuracy)
    
    def _calculate_directional_accuracy(self, predictions: np.ndarray, actuals: np.ndarray) -> float:
        """Calculate directional accuracy for financial predictions."""
        if len(predictions) < 2:
            return 0.0
        
        # Calculate direction of change
        pred_direction = np.sign(np.diff(predictions))
        actual_direction = np.sign(np.diff(actuals))
        
        # Calculate accuracy of directional predictions
        correct_directions = np.sum(pred_direction == actual_direction)
        total_predictions = len(pred_direction)
        
        return correct_directions / total_predictions if total_predictions > 0 else 0.0
    
    def _calculate_consistency(self, predictions: np.ndarray) -> float:
        """Calculate prediction consistency (inverse of variance)."""
        if len(predictions) < 2:
            return 1.0
        
        variance = np.var(predictions)
        # Normalize variance to 0-1 scale (higher = more consistent)
        consistency = 1.0 / (1.0 + variance)
        return consistency
    
    def _check_for_degradation(self, current_metrics: ModelMetrics) -> List[ModelAlert]:
        """Check current metrics against baseline for degradation."""
        alerts = []
        model_name = current_metrics.model_name
        
        # Skip if no baseline established
        if model_name not in self.model_baselines:
            return alerts
        
        baseline = self.model_baselines[model_name]
        timestamp = current_metrics.timestamp
        
        # Check accuracy degradation
        if baseline.baseline_accuracy > 0:
            accuracy_drop = (baseline.baseline_accuracy - current_metrics.accuracy) / baseline.baseline_accuracy
            
            if accuracy_drop > self.config['accuracy_critical_threshold']:
                alerts.append(ModelAlert(
                    timestamp=timestamp,
                    model_name=model_name,
                    severity=AlertSeverity.CRITICAL,
                    alert_type="ACCURACY_DEGRADATION",
                    message=f"Critical accuracy degradation: {accuracy_drop:.1%} below baseline",
                    metric_name="accuracy",
                    current_value=current_metrics.accuracy,
                    threshold_value=baseline.baseline_accuracy * (1 - self.config['accuracy_critical_threshold']),
                    trend=self._calculate_trend(model_name, "accuracy"),
                    recommended_action="Immediate model retraining required"
                ))
            
            elif accuracy_drop > self.config['accuracy_degradation_threshold']:
                alerts.append(ModelAlert(
                    timestamp=timestamp,
                    model_name=model_name,
                    severity=AlertSeverity.WARNING,
                    alert_type="ACCURACY_DEGRADATION",
                    message=f"Accuracy degradation detected: {accuracy_drop:.1%} below baseline",
                    metric_name="accuracy",
                    current_value=current_metrics.accuracy,
                    threshold_value=baseline.baseline_accuracy * (1 - self.config['accuracy_degradation_threshold']),
                    trend=self._calculate_trend(model_name, "accuracy"),
                    recommended_action="Consider model retraining or feature engineering"
                ))
        
        # Check MAE increase
        if baseline.baseline_mae > 0:
            mae_increase = (current_metrics.mean_absolute_error - baseline.baseline_mae) / baseline.baseline_mae
            
            if mae_increase > self.config['mae_increase_threshold']:
                alerts.append(ModelAlert(
                    timestamp=timestamp,
                    model_name=model_name,
                    severity=AlertSeverity.WARNING,
                    alert_type="ERROR_INCREASE",
                    message=f"Mean Absolute Error increased by {mae_increase:.1%}",
                    metric_name="mean_absolute_error",
                    current_value=current_metrics.mean_absolute_error,
                    threshold_value=baseline.baseline_mae * (1 + self.config['mae_increase_threshold']),
                    trend=self._calculate_trend(model_name, "mean_absolute_error"),
                    recommended_action="Review data quality and model parameters"
                ))
        
        # Check confidence degradation
        if baseline.baseline_confidence > 0:
            confidence_drop = (baseline.baseline_confidence - current_metrics.prediction_confidence) / baseline.baseline_confidence
            
            if confidence_drop > self.config['confidence_drop_threshold']:
                alerts.append(ModelAlert(
                    timestamp=timestamp,
                    model_name=model_name,
                    severity=AlertSeverity.WARNING,
                    alert_type="CONFIDENCE_DROP",
                    message=f"Prediction confidence dropped by {confidence_drop:.1%}",
                    metric_name="prediction_confidence",
                    current_value=current_metrics.prediction_confidence,
                    threshold_value=baseline.baseline_confidence * (1 - self.config['confidence_drop_threshold']),
                    trend=self._calculate_trend(model_name, "prediction_confidence"),
                    recommended_action="Check input data quality and model calibration"
                ))
        
        # Check performance degradation
        if baseline.baseline_inference_time > 0:
            time_increase = current_metrics.inference_time_ms / baseline.baseline_inference_time
            
            if time_increase > self.config['inference_time_increase_threshold']:
                alerts.append(ModelAlert(
                    timestamp=timestamp,
                    model_name=model_name,
                    severity=AlertSeverity.WARNING,
                    alert_type="PERFORMANCE_DEGRADATION",
                    message=f"Inference time increased by {time_increase:.1f}x",
                    metric_name="inference_time_ms",
                    current_value=current_metrics.inference_time_ms,
                    threshold_value=baseline.baseline_inference_time * self.config['inference_time_increase_threshold'],
                    trend=self._calculate_trend(model_name, "inference_time_ms"),
                    recommended_action="Optimize model or increase computational resources"
                ))
        
        # Check trading-specific metrics
        if current_metrics.directional_accuracy < self.config['min_directional_accuracy']:
            alerts.append(ModelAlert(
                timestamp=timestamp,
                model_name=model_name,
                severity=AlertSeverity.CRITICAL,
                alert_type="DIRECTIONAL_ACCURACY_LOW",
                message=f"Directional accuracy {current_metrics.directional_accuracy:.1%} below minimum",
                metric_name="directional_accuracy",
                current_value=current_metrics.directional_accuracy,
                threshold_value=self.config['min_directional_accuracy'],
                trend=self._calculate_trend(model_name, "directional_accuracy"),
                recommended_action="Model unsuitable for trading - retrain immediately"
            ))
        
        if current_metrics.sharpe_ratio is not None and current_metrics.sharpe_ratio < self.config['min_sharpe_ratio']:
            alerts.append(ModelAlert(
                timestamp=timestamp,
                model_name=model_name,
                severity=AlertSeverity.WARNING,
                alert_type="SHARPE_RATIO_LOW",
                message=f"Sharpe ratio {current_metrics.sharpe_ratio:.2f} below minimum",
                metric_name="sharpe_ratio",
                current_value=current_metrics.sharpe_ratio,
                threshold_value=self.config['min_sharpe_ratio'],
                trend=self._calculate_trend(model_name, "sharpe_ratio"),
                recommended_action="Review trading strategy and risk management"
            ))
        
        return alerts
    
    def _calculate_trend(self, model_name: str, metric_name: str, window: int = 10) -> str:
        """Calculate trend for a specific metric."""
        if (model_name not in self.metrics_history or 
            len(self.metrics_history[model_name]) < 3):
            return "insufficient_data"
        
        recent_metrics = self.metrics_history[model_name][-window:]
        values = [getattr(m, metric_name, 0) for m in recent_metrics]
        
        if len(values) < 3:
            return "insufficient_data"
        
        # Calculate linear trend
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        slope = coeffs[0]
        
        # Classify trend
        threshold = np.std(values) * 0.1  # 10% of standard deviation
        
        if slope > threshold:
            return "improving"
        elif slope < -threshold:
            return "degrading" 
        else:
            return "stable"
    
    def get_model_health_status(self, model_name: str) -> Dict[str, Any]:
        """Get comprehensive health status for a model."""
        if model_name not in self.metrics_history:
            return {
                'model_name': model_name,
                'health_status': ModelHealth.FAILED,
                'message': 'No performance data available'
            }
        
        recent_metrics = self.metrics_history[model_name][-1]  # Most recent
        recent_alerts = [a for a in self.alerts 
                        if a.model_name == model_name and 
                        a.timestamp > datetime.now() - timedelta(hours=24)]
        
        # Determine health status
        critical_alerts = [a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL]
        warning_alerts = [a for a in recent_alerts if a.severity == AlertSeverity.WARNING]
        
        if critical_alerts:
            health_status = ModelHealth.FAILED
            message = f"{len(critical_alerts)} critical issues detected"
        elif len(warning_alerts) >= 3:
            health_status = ModelHealth.POOR
            message = f"{len(warning_alerts)} warnings - degraded performance"
        elif warning_alerts:
            health_status = ModelHealth.DEGRADED
            message = f"{len(warning_alerts)} warning(s) - monitoring required"
        elif recent_metrics.accuracy >= 0.8 and recent_metrics.directional_accuracy >= 0.6:
            health_status = ModelHealth.EXCELLENT
            message = "Performing optimally"
        elif recent_metrics.accuracy >= 0.7 and recent_metrics.directional_accuracy >= 0.55:
            health_status = ModelHealth.GOOD
            message = "Performing within acceptable range"
        else:
            health_status = ModelHealth.DEGRADED
            message = "Performance below optimal levels"
        
        return {
            'model_name': model_name,
            'health_status': health_status,
            'message': message,
            'last_updated': recent_metrics.timestamp,
            'current_accuracy': recent_metrics.accuracy,
            'current_directional_accuracy': recent_metrics.directional_accuracy,
            'active_alerts': len(recent_alerts),
            'critical_alerts': len(critical_alerts),
            'warning_alerts': len(warning_alerts),
            'trend_summary': {
                'accuracy_trend': self._calculate_trend(model_name, 'accuracy'),
                'error_trend': self._calculate_trend(model_name, 'mean_absolute_error'),
                'confidence_trend': self._calculate_trend(model_name, 'prediction_confidence')
            }
        }
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report for all models."""
        report = {
            'generated_at': datetime.now(),
            'monitoring_period_days': self.config['metric_retention_days'],
            'total_models': len(self.metrics_history),
            'models': {},
            'system_summary': {
                'healthy_models': 0,
                'degraded_models': 0,
                'failed_models': 0,
                'total_alerts_24h': 0,
                'critical_alerts_24h': 0
            },
            'recommendations': []
        }
        
        # Generate status for each model
        for model_name in self.metrics_history.keys():
            model_status = self.get_model_health_status(model_name)
            report['models'][model_name] = model_status
            
            # Update system summary
            if model_status['health_status'] in [ModelHealth.EXCELLENT, ModelHealth.GOOD]:
                report['system_summary']['healthy_models'] += 1
            elif model_status['health_status'] == ModelHealth.DEGRADED:
                report['system_summary']['degraded_models'] += 1
            else:
                report['system_summary']['failed_models'] += 1
        
        # Count recent alerts
        cutoff_time = datetime.now() - timedelta(hours=24)
        recent_alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        report['system_summary']['total_alerts_24h'] = len(recent_alerts)
        report['system_summary']['critical_alerts_24h'] = len([a for a in recent_alerts if a.severity == AlertSeverity.CRITICAL])
        
        # Generate recommendations
        report['recommendations'] = self._generate_system_recommendations(report)
        
        return report
    
    def _generate_system_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate system-wide recommendations."""
        recommendations = []
        summary = report['system_summary']
        
        if summary['failed_models'] > 0:
            recommendations.append(f"URGENT: {summary['failed_models']} model(s) have failed and require immediate attention")
        
        if summary['critical_alerts_24h'] > 0:
            recommendations.append(f"Address {summary['critical_alerts_24h']} critical alert(s) immediately")
        
        if summary['degraded_models'] > summary['healthy_models']:
            recommendations.append("Majority of models are degraded - review training data and procedures")
        
        if summary['total_alerts_24h'] > 10:
            recommendations.append("High alert volume - consider adjusting monitoring thresholds")
        
        # Model-specific recommendations
        for model_name, model_data in report['models'].items():
            if model_data['health_status'] == ModelHealth.FAILED:
                recommendations.append(f"Model {model_name}: Immediate retraining required")
            elif (model_data['current_accuracy'] < 0.6 or 
                  model_data['current_directional_accuracy'] < 0.5):
                recommendations.append(f"Model {model_name}: Performance below trading threshold")
        
        if not recommendations:
            recommendations.append("All models operating within acceptable parameters")
        
        return recommendations[:10]  # Limit to top 10
    
    def _calculate_expected_ranges(self, baseline_metrics: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
        """Calculate expected ranges based on baseline metrics."""
        ranges = {}
        
        for metric_name, baseline_value in baseline_metrics.items():
            if metric_name in ['accuracy', 'precision', 'recall', 'f1_score', 'directional_accuracy']:
                # Performance metrics - allow small degradation
                ranges[metric_name] = (baseline_value * 0.85, 1.0)
            elif metric_name in ['mean_absolute_error', 'mean_squared_error']:
                # Error metrics - allow moderate increase
                ranges[metric_name] = (0.0, baseline_value * 1.5)
            elif metric_name == 'prediction_confidence':
                # Confidence - allow moderate decrease
                ranges[metric_name] = (baseline_value * 0.8, 1.0)
            elif metric_name == 'inference_time_ms':
                # Performance - allow 2x increase
                ranges[metric_name] = (0.0, baseline_value * 2.0)
            else:
                # Default range
                ranges[metric_name] = (baseline_value * 0.5, baseline_value * 1.5)
        
        return ranges
    
    def _load_historical_data(self):
        """Load historical metrics and baselines from disk."""
        try:
            # Load metrics
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                
                for model_name, metrics_list in data.items():
                    self.metrics_history[model_name] = [
                        ModelMetrics(**{**m, 'timestamp': datetime.fromisoformat(m['timestamp'])})
                        for m in metrics_list
                    ]
            
            # Load baselines
            if self.baselines_file.exists():
                with open(self.baselines_file, 'r') as f:
                    data = json.load(f)
                
                for model_name, baseline_data in data.items():
                    self.model_baselines[model_name] = ModelBaselineMetrics(
                        **{**baseline_data, 'established_date': datetime.fromisoformat(baseline_data['established_date'])}
                    )
            
            # Load alerts
            if self.alerts_file.exists():
                with open(self.alerts_file, 'r') as f:
                    data = json.load(f)
                
                self.alerts = [
                    ModelAlert(
                        **{**alert_data, 
                          'timestamp': datetime.fromisoformat(alert_data['timestamp']),
                          'severity': AlertSeverity(alert_data['severity'])}
                    )
                    for alert_data in data
                ]
                
        except Exception as e:
            logger.warning(f"Could not load historical monitoring data: {e}")
    
    def _save_metrics(self):
        """Save metrics to disk."""
        try:
            data = {}
            for model_name, metrics_list in self.metrics_history.items():
                data[model_name] = [
                    {**asdict(m), 'timestamp': m.timestamp.isoformat()}
                    for m in metrics_list
                ]
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Could not save metrics: {e}")
    
    def _save_baselines(self):
        """Save baselines to disk."""
        try:
            data = {
                model_name: {**asdict(baseline), 'established_date': baseline.established_date.isoformat()}
                for model_name, baseline in self.model_baselines.items()
            }
            
            with open(self.baselines_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Could not save baselines: {e}")
    
    def _save_alerts(self):
        """Save alerts to disk."""
        try:
            data = [
                {**asdict(alert), 
                 'timestamp': alert.timestamp.isoformat(),
                 'severity': alert.severity.value}
                for alert in self.alerts
            ]
            
            with open(self.alerts_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Could not save alerts: {e}")
    
    def _cleanup_old_metrics(self):
        """Clean up old metrics to maintain storage limits."""
        cutoff_date = datetime.now() - timedelta(days=self.config['metric_retention_days'])
        
        for model_name in self.metrics_history:
            self.metrics_history[model_name] = [
                m for m in self.metrics_history[model_name]
                if m.timestamp > cutoff_date
            ]
        
        # Clean up old alerts
        self.alerts = [
            a for a in self.alerts
            if a.timestamp > cutoff_date
        ]


def main():
    """Demo and testing of model monitor."""
    monitor = ModelMonitor()
    
    # Register baseline for a sample model
    baseline_metrics = {
        'accuracy': 0.85,
        'mae': 0.05,
        'mse': 0.003,
        'confidence': 0.78,
        'inference_time_ms': 150.0
    }
    
    monitor.register_model_baseline('TimesFM', baseline_metrics)
    
    # Simulate model performance recording
    for i in range(5):
        # Simulate gradual degradation
        accuracy_drop = i * 0.02
        error_increase = i * 0.01
        
        predictions = np.random.normal(0.5, 0.1, 100)
        actuals = np.random.normal(0.5, 0.1, 100)
        
        metadata = {
            'confidence': 0.78 - (i * 0.02),
            'inference_time_ms': 150.0 + (i * 20),
            'data_coverage': 0.95
        }
        
        metrics = monitor.record_model_performance(
            'TimesFM', predictions, actuals, metadata
        )
        
        print(f"Iteration {i+1}: Accuracy={metrics.accuracy:.3f}, MAE={metrics.mean_absolute_error:.3f}")
    
    # Generate monitoring report
    report = monitor.generate_monitoring_report()
    
    print("\n" + "="*50)
    print("MODEL MONITORING REPORT")
    print("="*50)
    print(f"Healthy Models: {report['system_summary']['healthy_models']}")
    print(f"Degraded Models: {report['system_summary']['degraded_models']}")
    print(f"Failed Models: {report['system_summary']['failed_models']}")
    print(f"Alerts (24h): {report['system_summary']['total_alerts_24h']}")
    
    print("\nModel Status:")
    for model_name, status in report['models'].items():
        print(f"  {model_name}: {status['health_status'].value} - {status['message']}")
    
    print("\nRecommendations:")
    for i, rec in enumerate(report['recommendations'][:5], 1):
        print(f"  {i}. {rec}")


if __name__ == "__main__":
    main()
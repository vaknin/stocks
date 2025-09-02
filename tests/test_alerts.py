"""
Alert System Testing Suite - System Failure and Performance Monitoring

Tests for the alert and notification system that monitors recommendation system
health, performance degradation, data quality issues, and critical failures
to ensure reliable investment advice delivery.

Priority: TIER 3 - Risk management for system reliability and user confidence
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys
import json
from enum import Enum

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.monitoring.alert_system import AlertSystem, Alert, AlertType, AlertSeverity
from src.config.settings import config


class AlertType(Enum):
    """Types of alerts that can be generated."""
    SYSTEM_FAILURE = "system_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_QUALITY_ISSUE = "data_quality_issue"
    MODEL_DRIFT = "model_drift"
    API_ERROR = "api_error"
    MEMORY_WARNING = "memory_warning"
    LATENCY_WARNING = "latency_warning"
    CONFIDENCE_DROP = "confidence_drop"


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    EMERGENCY = 5


class Alert:
    """Represents a system alert."""
    
    def __init__(self, 
                 alert_type: AlertType,
                 severity: AlertSeverity,
                 title: str,
                 description: str,
                 component: str = None,
                 metric_value: float = None,
                 threshold: float = None,
                 timestamp: datetime = None,
                 metadata: Dict[str, Any] = None):
        self.alert_type = alert_type
        self.severity = severity
        self.title = title
        self.description = description
        self.component = component
        self.metric_value = metric_value
        self.threshold = threshold
        self.timestamp = timestamp or datetime.now()
        self.metadata = metadata or {}
        self.is_resolved = False
        self.resolution_time = None
        
    def resolve(self):
        """Mark alert as resolved."""
        self.is_resolved = True
        self.resolution_time = datetime.now()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'description': self.description,
            'component': self.component,
            'metric_value': self.metric_value,
            'threshold': self.threshold,
            'timestamp': self.timestamp.isoformat(),
            'is_resolved': self.is_resolved,
            'resolution_time': self.resolution_time.isoformat() if self.resolution_time else None,
            'metadata': self.metadata
        }


class AlertSystem:
    """Manages system alerts and notifications."""
    
    def __init__(self, 
                 max_alerts: int = 1000,
                 alert_retention_days: int = 30):
        self.max_alerts = max_alerts
        self.alert_retention_days = alert_retention_days
        self.active_alerts: List[Alert] = []
        self.alert_history: List[Alert] = []
        self.alert_thresholds = self._initialize_thresholds()
        self.notification_handlers = []
        self.alert_suppression_rules = {}
        
    def _initialize_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Initialize default alert thresholds."""
        return {
            'performance': {
                'accuracy_threshold': 0.50,      # Alert if accuracy < 50%
                'confidence_threshold': 0.60,    # Alert if avg confidence < 60%
                'latency_threshold': 2.0,        # Alert if latency > 2s
                'error_rate_threshold': 0.10     # Alert if error rate > 10%
            },
            'system': {
                'memory_threshold_mb': 2000,     # Alert if memory > 2GB
                'cpu_threshold': 0.80,           # Alert if CPU > 80%
                'disk_threshold': 0.90           # Alert if disk > 90%
            },
            'data': {
                'missing_data_threshold': 0.20,  # Alert if >20% data missing
                'stale_data_hours': 6,           # Alert if data >6 hours old
                'correlation_drop_threshold': 0.30  # Alert if correlation drops >30%
            }
        }
    
    def add_alert(self, alert: Alert) -> None:
        """Add new alert to the system."""
        # Check for duplicate alerts (suppress if similar alert exists)
        if not self._should_suppress_alert(alert):
            self.active_alerts.append(alert)
            self.alert_history.append(alert)
            
            # Notify handlers
            self._notify_handlers(alert)
            
            # Clean up old alerts
            self._cleanup_old_alerts()
    
    def _should_suppress_alert(self, new_alert: Alert) -> bool:
        """Check if alert should be suppressed due to similar recent alerts."""
        suppression_window = timedelta(minutes=15)  # 15-minute suppression window
        
        for active_alert in self.active_alerts:
            if (active_alert.alert_type == new_alert.alert_type and
                active_alert.component == new_alert.component and
                not active_alert.is_resolved and
                (new_alert.timestamp - active_alert.timestamp) < suppression_window):
                return True
        
        return False
    
    def _notify_handlers(self, alert: Alert) -> None:
        """Notify all registered alert handlers."""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                # Log notification failure but don't fail the alert
                print(f"Alert notification failed: {e}")
    
    def _cleanup_old_alerts(self) -> None:
        """Clean up old alerts to prevent unbounded growth."""
        cutoff_time = datetime.now() - timedelta(days=self.alert_retention_days)
        
        # Remove old alerts from history
        self.alert_history = [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
        
        # Limit total number of alerts
        if len(self.alert_history) > self.max_alerts:
            self.alert_history = self.alert_history[-self.max_alerts:]
        
        # Remove resolved alerts from active list
        self.active_alerts = [
            alert for alert in self.active_alerts
            if not alert.is_resolved and alert.timestamp > cutoff_time
        ]
    
    def check_performance_thresholds(self, 
                                   performance_metrics: Dict[str, float]) -> List[Alert]:
        """Check performance metrics against thresholds and generate alerts."""
        alerts = []
        thresholds = self.alert_thresholds['performance']
        
        # Check accuracy threshold
        accuracy = performance_metrics.get('directional_accuracy', 1.0)
        if accuracy < thresholds['accuracy_threshold']:
            alert = Alert(
                alert_type=AlertType.PERFORMANCE_DEGRADATION,
                severity=AlertSeverity.HIGH if accuracy < 0.4 else AlertSeverity.MEDIUM,
                title="Low Prediction Accuracy",
                description=f"Model accuracy dropped to {accuracy:.1%} (threshold: {thresholds['accuracy_threshold']:.1%})",
                component="prediction_model",
                metric_value=accuracy,
                threshold=thresholds['accuracy_threshold']
            )
            alerts.append(alert)
        
        # Check confidence threshold
        avg_confidence = performance_metrics.get('avg_confidence', 1.0)
        if avg_confidence < thresholds['confidence_threshold']:
            alert = Alert(
                alert_type=AlertType.CONFIDENCE_DROP,
                severity=AlertSeverity.MEDIUM,
                title="Low Confidence Scores",
                description=f"Average confidence dropped to {avg_confidence:.1%} (threshold: {thresholds['confidence_threshold']:.1%})",
                component="confidence_calibration",
                metric_value=avg_confidence,
                threshold=thresholds['confidence_threshold']
            )
            alerts.append(alert)
        
        # Check latency threshold
        avg_latency = performance_metrics.get('avg_latency', 0.0)
        if avg_latency > thresholds['latency_threshold']:
            alert = Alert(
                alert_type=AlertType.LATENCY_WARNING,
                severity=AlertSeverity.HIGH if avg_latency > 5.0 else AlertSeverity.MEDIUM,
                title="High Prediction Latency",
                description=f"Average latency increased to {avg_latency:.2f}s (threshold: {thresholds['latency_threshold']:.2f}s)",
                component="prediction_pipeline",
                metric_value=avg_latency,
                threshold=thresholds['latency_threshold']
            )
            alerts.append(alert)
        
        # Check error rate threshold
        error_rate = performance_metrics.get('error_rate', 0.0)
        if error_rate > thresholds['error_rate_threshold']:
            alert = Alert(
                alert_type=AlertType.SYSTEM_FAILURE,
                severity=AlertSeverity.CRITICAL if error_rate > 0.25 else AlertSeverity.HIGH,
                title="High Error Rate",
                description=f"System error rate increased to {error_rate:.1%} (threshold: {thresholds['error_rate_threshold']:.1%})",
                component="system",
                metric_value=error_rate,
                threshold=thresholds['error_rate_threshold']
            )
            alerts.append(alert)
        
        return alerts
    
    def check_data_quality_thresholds(self, 
                                    data_quality_metrics: Dict[str, float]) -> List[Alert]:
        """Check data quality metrics against thresholds."""
        alerts = []
        thresholds = self.alert_thresholds['data']
        
        # Check missing data threshold
        missing_data_pct = data_quality_metrics.get('missing_data_percentage', 0.0)
        if missing_data_pct > thresholds['missing_data_threshold']:
            alert = Alert(
                alert_type=AlertType.DATA_QUALITY_ISSUE,
                severity=AlertSeverity.HIGH if missing_data_pct > 0.4 else AlertSeverity.MEDIUM,
                title="High Missing Data Percentage",
                description=f"Missing data percentage: {missing_data_pct:.1%} (threshold: {thresholds['missing_data_threshold']:.1%})",
                component="data_pipeline",
                metric_value=missing_data_pct,
                threshold=thresholds['missing_data_threshold']
            )
            alerts.append(alert)
        
        # Check data staleness
        data_age_hours = data_quality_metrics.get('data_age_hours', 0.0)
        if data_age_hours > thresholds['stale_data_hours']:
            alert = Alert(
                alert_type=AlertType.DATA_QUALITY_ISSUE,
                severity=AlertSeverity.CRITICAL if data_age_hours > 24 else AlertSeverity.HIGH,
                title="Stale Data Detected",
                description=f"Data age: {data_age_hours:.1f} hours (threshold: {thresholds['stale_data_hours']} hours)",
                component="data_ingestion",
                metric_value=data_age_hours,
                threshold=thresholds['stale_data_hours']
            )
            alerts.append(alert)
        
        return alerts
    
    def check_system_health(self, system_metrics: Dict[str, float]) -> List[Alert]:
        """Check system health metrics against thresholds."""
        alerts = []
        thresholds = self.alert_thresholds['system']
        
        # Check memory usage
        memory_mb = system_metrics.get('memory_usage_mb', 0.0)
        if memory_mb > thresholds['memory_threshold_mb']:
            alert = Alert(
                alert_type=AlertType.MEMORY_WARNING,
                severity=AlertSeverity.HIGH if memory_mb > 4000 else AlertSeverity.MEDIUM,
                title="High Memory Usage",
                description=f"Memory usage: {memory_mb:.0f}MB (threshold: {thresholds['memory_threshold_mb']}MB)",
                component="system_resources",
                metric_value=memory_mb,
                threshold=thresholds['memory_threshold_mb']
            )
            alerts.append(alert)
        
        return alerts
    
    def get_active_alerts(self, 
                         severity_filter: AlertSeverity = None,
                         component_filter: str = None) -> List[Alert]:
        """Get currently active alerts with optional filtering."""
        filtered_alerts = self.active_alerts
        
        if severity_filter:
            filtered_alerts = [a for a in filtered_alerts if a.severity == severity_filter]
        
        if component_filter:
            filtered_alerts = [a for a in filtered_alerts if a.component == component_filter]
        
        return [a for a in filtered_alerts if not a.is_resolved]
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of alerts within specified time window."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.timestamp > cutoff_time
        ]
        
        # Count by type
        type_counts = {}
        for alert in recent_alerts:
            alert_type = alert.alert_type.value
            type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
        
        # Count by severity
        severity_counts = {}
        for alert in recent_alerts:
            severity = alert.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'total_alerts': len(recent_alerts),
            'active_alerts': len(self.get_active_alerts()),
            'alerts_by_type': type_counts,
            'alerts_by_severity': severity_counts,
            'time_window_hours': hours
        }
    
    def resolve_alert(self, alert: Alert) -> None:
        """Resolve an active alert."""
        alert.resolve()
        
    def add_notification_handler(self, handler) -> None:
        """Add a notification handler function."""
        self.notification_handlers.append(handler)


class TestAlertSystem:
    """Test suite for alert system functionality."""
    
    @pytest.fixture
    def alert_system(self):
        """Create alert system for testing."""
        return AlertSystem(max_alerts=100, alert_retention_days=7)
    
    @pytest.fixture
    def mock_notification_handler(self):
        """Create mock notification handler."""
        return Mock()
    
    def test_alert_creation_and_properties(self):
        """Test basic alert creation and properties."""
        alert = Alert(
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.HIGH,
            title="Test Alert",
            description="This is a test alert",
            component="test_component",
            metric_value=0.45,
            threshold=0.50
        )
        
        # Validate basic properties
        assert alert.alert_type == AlertType.PERFORMANCE_DEGRADATION
        assert alert.severity == AlertSeverity.HIGH
        assert alert.title == "Test Alert"
        assert alert.description == "This is a test alert"
        assert alert.component == "test_component"
        assert alert.metric_value == 0.45
        assert alert.threshold == 0.50
        assert not alert.is_resolved
        assert alert.resolution_time is None
        assert isinstance(alert.timestamp, datetime)
        
        # Test alert resolution
        alert.resolve()
        assert alert.is_resolved
        assert alert.resolution_time is not None
        assert isinstance(alert.resolution_time, datetime)
        
        # Test serialization
        alert_dict = alert.to_dict()
        assert isinstance(alert_dict, dict)
        assert alert_dict['alert_type'] == 'performance_degradation'
        assert alert_dict['severity'] == 3
        assert alert_dict['is_resolved'] is True
    
    def test_alert_system_initialization(self, alert_system):
        """Test alert system initialization and configuration."""
        assert isinstance(alert_system.alert_thresholds, dict)
        assert 'performance' in alert_system.alert_thresholds
        assert 'system' in alert_system.alert_thresholds
        assert 'data' in alert_system.alert_thresholds
        
        # Validate threshold structure
        perf_thresholds = alert_system.alert_thresholds['performance']
        assert 'accuracy_threshold' in perf_thresholds
        assert 'confidence_threshold' in perf_thresholds
        assert 'latency_threshold' in perf_thresholds
        
        # Validate initial state
        assert len(alert_system.active_alerts) == 0
        assert len(alert_system.alert_history) == 0
        assert len(alert_system.notification_handlers) == 0
    
    def test_performance_threshold_monitoring(self, alert_system):
        """Test performance threshold monitoring and alert generation."""
        # Test case 1: Normal performance (no alerts)
        normal_metrics = {
            'directional_accuracy': 0.70,  # Above threshold
            'avg_confidence': 0.75,        # Above threshold
            'avg_latency': 0.5,            # Below threshold
            'error_rate': 0.02             # Below threshold
        }
        
        alerts = alert_system.check_performance_thresholds(normal_metrics)
        assert len(alerts) == 0, "Should not generate alerts for normal performance"
        
        # Test case 2: Low accuracy (should alert)
        low_accuracy_metrics = {
            'directional_accuracy': 0.40,  # Below threshold
            'avg_confidence': 0.75,
            'avg_latency': 0.5,
            'error_rate': 0.02
        }
        
        alerts = alert_system.check_performance_thresholds(low_accuracy_metrics)
        assert len(alerts) == 1, "Should generate alert for low accuracy"
        
        accuracy_alert = alerts[0]
        assert accuracy_alert.alert_type == AlertType.PERFORMANCE_DEGRADATION
        assert accuracy_alert.severity == AlertSeverity.HIGH  # <0.4 should be HIGH
        assert accuracy_alert.component == "prediction_model"
        assert "accuracy" in accuracy_alert.title.lower()
        
        # Test case 3: Multiple threshold violations
        multiple_issues_metrics = {
            'directional_accuracy': 0.45,  # Below threshold
            'avg_confidence': 0.55,        # Below threshold
            'avg_latency': 3.0,            # Above threshold
            'error_rate': 0.15             # Above threshold
        }
        
        alerts = alert_system.check_performance_thresholds(multiple_issues_metrics)
        assert len(alerts) == 4, "Should generate multiple alerts for multiple issues"
        
        # Validate different alert types are generated
        alert_types = {alert.alert_type for alert in alerts}
        expected_types = {
            AlertType.PERFORMANCE_DEGRADATION,
            AlertType.CONFIDENCE_DROP,
            AlertType.LATENCY_WARNING,
            AlertType.SYSTEM_FAILURE
        }
        assert alert_types == expected_types, f"Expected alert types not generated: {alert_types}"
    
    def test_data_quality_threshold_monitoring(self, alert_system):
        """Test data quality threshold monitoring."""
        # Test case 1: Normal data quality
        normal_data_metrics = {
            'missing_data_percentage': 0.05,  # Below threshold
            'data_age_hours': 2.0             # Below threshold
        }
        
        alerts = alert_system.check_data_quality_thresholds(normal_data_metrics)
        assert len(alerts) == 0, "Should not alert for normal data quality"
        
        # Test case 2: High missing data
        high_missing_data_metrics = {
            'missing_data_percentage': 0.30,  # Above threshold (20%)
            'data_age_hours': 2.0
        }
        
        alerts = alert_system.check_data_quality_thresholds(high_missing_data_metrics)
        assert len(alerts) == 1, "Should alert for high missing data"
        
        missing_data_alert = alerts[0]
        assert missing_data_alert.alert_type == AlertType.DATA_QUALITY_ISSUE
        assert missing_data_alert.component == "data_pipeline"
        assert "missing data" in missing_data_alert.title.lower()
        
        # Test case 3: Stale data
        stale_data_metrics = {
            'missing_data_percentage': 0.05,
            'data_age_hours': 12.0  # Above threshold (6 hours)
        }
        
        alerts = alert_system.check_data_quality_thresholds(stale_data_metrics)
        assert len(alerts) == 1, "Should alert for stale data"
        
        stale_alert = alerts[0]
        assert stale_alert.alert_type == AlertType.DATA_QUALITY_ISSUE
        assert stale_alert.component == "data_ingestion"
        assert "stale" in stale_alert.title.lower()
        assert stale_alert.severity == AlertSeverity.HIGH  # 12h should be HIGH
        
        # Test case 4: Critical stale data (>24 hours)
        critical_stale_metrics = {
            'missing_data_percentage': 0.05,
            'data_age_hours': 30.0  # Above 24h threshold
        }
        
        alerts = alert_system.check_data_quality_thresholds(critical_stale_metrics)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL, "Very stale data should be CRITICAL"
    
    def test_system_health_monitoring(self, alert_system):
        """Test system health threshold monitoring."""
        # Normal system health
        normal_system_metrics = {
            'memory_usage_mb': 1500,  # Below threshold
            'cpu_usage': 0.60,        # Below threshold
            'disk_usage': 0.70        # Below threshold
        }
        
        alerts = alert_system.check_system_health(normal_system_metrics)
        assert len(alerts) == 0, "Should not alert for normal system health"
        
        # High memory usage
        high_memory_metrics = {
            'memory_usage_mb': 3000,  # Above threshold (2000MB)
            'cpu_usage': 0.60,
            'disk_usage': 0.70
        }
        
        alerts = alert_system.check_system_health(high_memory_metrics)
        assert len(alerts) == 1, "Should alert for high memory usage"
        
        memory_alert = alerts[0]
        assert memory_alert.alert_type == AlertType.MEMORY_WARNING
        assert memory_alert.component == "system_resources"
        assert "memory" in memory_alert.title.lower()
        
        # Critical memory usage
        critical_memory_metrics = {
            'memory_usage_mb': 5000,  # Above critical threshold
        }
        
        alerts = alert_system.check_system_health(critical_memory_metrics)
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.HIGH, "Very high memory should be HIGH severity"
    
    def test_alert_suppression_mechanism(self, alert_system):
        """Test alert suppression to prevent spam."""
        # Create identical alerts within suppression window
        alert1 = Alert(
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.MEDIUM,
            title="Test Alert",
            description="Test description",
            component="test_component"
        )
        
        alert2 = Alert(
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.MEDIUM,
            title="Test Alert 2",
            description="Test description 2",
            component="test_component",
            timestamp=alert1.timestamp + timedelta(minutes=5)  # Within suppression window
        )
        
        # Add first alert
        alert_system.add_alert(alert1)
        assert len(alert_system.active_alerts) == 1, "First alert should be added"
        
        # Add second alert (should be suppressed)
        alert_system.add_alert(alert2)
        assert len(alert_system.active_alerts) == 1, "Second alert should be suppressed"
        
        # Add alert outside suppression window
        alert3 = Alert(
            alert_type=AlertType.PERFORMANCE_DEGRADATION,
            severity=AlertSeverity.MEDIUM,
            title="Test Alert 3",
            description="Test description 3",
            component="test_component",
            timestamp=alert1.timestamp + timedelta(minutes=20)  # Outside suppression window
        )
        
        alert_system.add_alert(alert3)
        assert len(alert_system.active_alerts) == 2, "Alert outside window should be added"
    
    def test_notification_handler_system(self, alert_system, mock_notification_handler):
        """Test notification handler registration and execution."""
        # Register notification handler
        alert_system.add_notification_handler(mock_notification_handler)
        assert len(alert_system.notification_handlers) == 1
        
        # Create and add alert
        test_alert = Alert(
            alert_type=AlertType.SYSTEM_FAILURE,
            severity=AlertSeverity.CRITICAL,
            title="Test System Failure",
            description="System failure for testing"
        )
        
        alert_system.add_alert(test_alert)
        
        # Verify notification handler was called
        mock_notification_handler.assert_called_once_with(test_alert)
        
        # Test notification handler failure doesn't break system
        failing_handler = Mock(side_effect=Exception("Notification failed"))
        alert_system.add_notification_handler(failing_handler)
        
        test_alert2 = Alert(
            alert_type=AlertType.API_ERROR,
            severity=AlertSeverity.MEDIUM,
            title="API Error Test",
            description="API error for testing"
        )
        
        # Should not raise exception despite failing handler
        try:
            alert_system.add_alert(test_alert2)
            assert True, "System should handle notification failures gracefully"
        except Exception as e:
            pytest.fail(f"Alert system should handle notification failures: {e}")
    
    def test_alert_filtering_and_retrieval(self, alert_system):
        """Test alert filtering and retrieval functionality."""
        # Create alerts with different severities and components
        alerts_to_add = [
            Alert(AlertType.PERFORMANCE_DEGRADATION, AlertSeverity.LOW, "Low Perf", "desc", "model"),
            Alert(AlertType.SYSTEM_FAILURE, AlertSeverity.CRITICAL, "Critical System", "desc", "system"),
            Alert(AlertType.DATA_QUALITY_ISSUE, AlertSeverity.MEDIUM, "Data Issue", "desc", "data_pipeline"),
            Alert(AlertType.MEMORY_WARNING, AlertSeverity.HIGH, "Memory High", "desc", "system")
        ]
        
        for alert in alerts_to_add:
            alert_system.add_alert(alert)
        
        # Test getting all active alerts
        all_active = alert_system.get_active_alerts()
        assert len(all_active) == 4, "Should retrieve all active alerts"
        
        # Test severity filtering
        critical_alerts = alert_system.get_active_alerts(severity_filter=AlertSeverity.CRITICAL)
        assert len(critical_alerts) == 1, "Should find 1 critical alert"
        assert critical_alerts[0].alert_type == AlertType.SYSTEM_FAILURE
        
        high_alerts = alert_system.get_active_alerts(severity_filter=AlertSeverity.HIGH)
        assert len(high_alerts) == 1, "Should find 1 high severity alert"
        
        # Test component filtering
        system_alerts = alert_system.get_active_alerts(component_filter="system")
        assert len(system_alerts) == 2, "Should find 2 system component alerts"
        
        model_alerts = alert_system.get_active_alerts(component_filter="model")
        assert len(model_alerts) == 1, "Should find 1 model component alert"
        
        # Test combined filtering
        high_system_alerts = alert_system.get_active_alerts(
            severity_filter=AlertSeverity.HIGH,
            component_filter="system"
        )
        assert len(high_system_alerts) == 1, "Should find 1 high-severity system alert"
    
    def test_alert_summary_generation(self, alert_system):
        """Test alert summary statistics generation."""
        # Add alerts with different timestamps
        now = datetime.now()
        
        recent_alerts = [
            Alert(AlertType.PERFORMANCE_DEGRADATION, AlertSeverity.MEDIUM, "Recent 1", "desc",
                  timestamp=now - timedelta(hours=2)),
            Alert(AlertType.SYSTEM_FAILURE, AlertSeverity.CRITICAL, "Recent 2", "desc",
                  timestamp=now - timedelta(hours=5)),
        ]
        
        old_alerts = [
            Alert(AlertType.DATA_QUALITY_ISSUE, AlertSeverity.LOW, "Old 1", "desc",
                  timestamp=now - timedelta(hours=30)),
            Alert(AlertType.API_ERROR, AlertSeverity.HIGH, "Old 2", "desc",
                  timestamp=now - timedelta(hours=48))
        ]
        
        all_alerts = recent_alerts + old_alerts
        for alert in all_alerts:
            alert_system.add_alert(alert)
        
        # Test 24-hour summary
        summary_24h = alert_system.get_alert_summary(hours=24)
        
        assert summary_24h['total_alerts'] == 2, "Should find 2 alerts in last 24 hours"
        assert summary_24h['time_window_hours'] == 24
        
        # Test type counts
        type_counts = summary_24h['alerts_by_type']
        assert type_counts.get('performance_degradation', 0) == 1
        assert type_counts.get('system_failure', 0) == 1
        
        # Test severity counts
        severity_counts = summary_24h['alerts_by_severity']
        assert severity_counts.get(AlertSeverity.MEDIUM.value, 0) == 1
        assert severity_counts.get(AlertSeverity.CRITICAL.value, 0) == 1
        
        # Test longer time window
        summary_72h = alert_system.get_alert_summary(hours=72)
        assert summary_72h['total_alerts'] == 4, "Should find all alerts in 72-hour window"
    
    def test_alert_cleanup_and_retention(self, alert_system):
        """Test alert cleanup and retention policies."""
        # Test with short retention for quick testing
        short_retention_system = AlertSystem(max_alerts=10, alert_retention_days=1)
        
        # Add old alerts
        old_time = datetime.now() - timedelta(days=2)  # Older than retention period
        
        for i in range(15):  # More than max_alerts
            alert = Alert(
                AlertType.PERFORMANCE_DEGRADATION,
                AlertSeverity.LOW,
                f"Old Alert {i}",
                "description",
                timestamp=old_time
            )
            short_retention_system.add_alert(alert)
        
        # Old alerts should be cleaned up
        assert len(short_retention_system.alert_history) <= 10, "Should respect max_alerts limit"
        assert len(short_retention_system.active_alerts) == 0, "Old alerts should not be active"
        
        # Add recent alerts
        for i in range(5):
            alert = Alert(
                AlertType.SYSTEM_FAILURE,
                AlertSeverity.MEDIUM,
                f"Recent Alert {i}",
                "description"
            )
            short_retention_system.add_alert(alert)
        
        assert len(short_retention_system.active_alerts) == 5, "Recent alerts should be active"
        
        # Resolve some alerts
        for alert in short_retention_system.active_alerts[:2]:
            short_retention_system.resolve_alert(alert)
        
        # Trigger cleanup
        short_retention_system._cleanup_old_alerts()
        
        active_count = len(short_retention_system.get_active_alerts())
        assert active_count == 3, f"Should have 3 unresolved active alerts, got {active_count}"
    
    def test_alert_escalation_scenarios(self, alert_system):
        """Test alert escalation based on severity and persistence."""
        # Simulate escalating performance issues
        performance_scenarios = [
            ({'directional_accuracy': 0.55}, AlertSeverity.MEDIUM),  # Initial degradation
            ({'directional_accuracy': 0.45}, AlertSeverity.HIGH),   # Further degradation
            ({'directional_accuracy': 0.35}, AlertSeverity.HIGH),   # Severe degradation
        ]
        
        generated_alerts = []
        
        for metrics, expected_min_severity in performance_scenarios:
            alerts = alert_system.check_performance_thresholds(metrics)
            
            if alerts:
                perf_alerts = [a for a in alerts if a.alert_type == AlertType.PERFORMANCE_DEGRADATION]
                if perf_alerts:
                    generated_alerts.append(perf_alerts[0])
                    
                    # Verify escalation
                    assert perf_alerts[0].severity.value >= expected_min_severity.value, (
                        f"Expected severity >= {expected_min_severity.value}, "
                        f"got {perf_alerts[0].severity.value} for accuracy {metrics['directional_accuracy']}"
                    )
        
        # Should generate alerts for degrading performance
        assert len(generated_alerts) >= 2, "Should generate alerts for degrading performance"
    
    def test_comprehensive_monitoring_scenario(self, alert_system, mock_notification_handler):
        """Test comprehensive monitoring scenario with multiple simultaneous issues."""
        # Register notification handler
        alert_system.add_notification_handler(mock_notification_handler)
        
        # Simulate system under stress with multiple issues
        stress_metrics = {
            'performance': {
                'directional_accuracy': 0.42,  # Low accuracy
                'avg_confidence': 0.58,        # Low confidence
                'avg_latency': 4.0,            # High latency
                'error_rate': 0.18             # High error rate
            },
            'data_quality': {
                'missing_data_percentage': 0.35,  # High missing data
                'data_age_hours': 8.0              # Stale data
            },
            'system_health': {
                'memory_usage_mb': 3500,  # High memory
            }
        }
        
        # Check all thresholds
        all_alerts = []
        all_alerts.extend(alert_system.check_performance_thresholds(stress_metrics['performance']))
        all_alerts.extend(alert_system.check_data_quality_thresholds(stress_metrics['data_quality']))
        all_alerts.extend(alert_system.check_system_health(stress_metrics['system_health']))
        
        # Should generate multiple alerts
        assert len(all_alerts) >= 6, f"Should generate multiple alerts for system stress, got {len(all_alerts)}"
        
        # Add all alerts to system
        for alert in all_alerts:
            alert_system.add_alert(alert)
        
        # Verify notifications were sent
        assert mock_notification_handler.call_count == len(all_alerts), \
            "Should send notification for each alert"
        
        # Verify different alert types are present
        alert_types = {alert.alert_type for alert in all_alerts}
        expected_types = {
            AlertType.PERFORMANCE_DEGRADATION,
            AlertType.DATA_QUALITY_ISSUE,
            AlertType.MEMORY_WARNING
        }
        
        assert len(alert_types.intersection(expected_types)) >= 3, \
            f"Should have multiple alert types, got {alert_types}"
        
        # Verify severity distribution
        critical_alerts = [a for a in all_alerts if a.severity == AlertSeverity.CRITICAL]
        high_alerts = [a for a in all_alerts if a.severity == AlertSeverity.HIGH]
        
        assert len(critical_alerts) + len(high_alerts) >= 3, \
            "Should have multiple high-severity alerts during system stress"
        
        # Test alert summary during crisis
        summary = alert_system.get_alert_summary()
        assert summary['total_alerts'] >= 6, "Summary should reflect all alerts"
        assert summary['active_alerts'] >= 6, "All alerts should be active"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
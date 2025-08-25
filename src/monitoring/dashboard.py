"""Performance monitoring dashboard for the trading system."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
from pathlib import Path
from loguru import logger

from ..config.settings import config


class PerformanceDashboard:
    """Real-time performance monitoring and metrics dashboard."""
    
    def __init__(self, metrics_file: Optional[str] = None):
        """Initialize performance dashboard.
        
        Args:
            metrics_file: Path to store metrics data. Defaults to logs/performance_metrics.json
        """
        self.metrics_file = metrics_file or (Path(__file__).parent.parent.parent / "logs" / "performance_metrics.json")
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Performance metrics storage
        self.metrics_history = []
        self.current_metrics = {}
        
        # Load existing metrics
        self._load_metrics_history()
        
        logger.info(f"Performance dashboard initialized with metrics file: {self.metrics_file}")
    
    def record_system_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a system performance metric.
        
        Args:
            metric_name: Name of the metric (e.g., 'data_ingestion_latency')
            value: Metric value
            timestamp: When the metric was recorded (defaults to now)
            metadata: Additional context information
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        metric_record = {
            'metric_name': metric_name,
            'value': value,
            'timestamp': timestamp.isoformat(),
            'metadata': metadata or {}
        }
        
        self.metrics_history.append(metric_record)
        self.current_metrics[metric_name] = {
            'value': value,
            'timestamp': timestamp.isoformat(),
            'metadata': metadata or {}
        }
        
        # Save to file
        self._save_metrics()
        
        logger.debug(f"Recorded metric: {metric_name} = {value}")
    
    def record_trading_performance(
        self,
        portfolio_value: float,
        daily_return: float,
        drawdown: float,
        sharpe_ratio: Optional[float] = None,
        win_rate: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record trading performance metrics.
        
        Args:
            portfolio_value: Current portfolio value
            daily_return: Daily return percentage
            drawdown: Current drawdown from peak
            sharpe_ratio: Sharpe ratio (optional)
            win_rate: Win rate percentage (optional)
            timestamp: When recorded (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        trading_metrics = {
            'portfolio_value': portfolio_value,
            'daily_return': daily_return,
            'drawdown': drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate
        }
        
        for metric_name, value in trading_metrics.items():
            if value is not None:
                self.record_system_metric(
                    metric_name=f"trading_{metric_name}",
                    value=value,
                    timestamp=timestamp,
                    metadata={'category': 'trading_performance'}
                )
    
    def record_model_performance(
        self,
        model_name: str,
        prediction_accuracy: float,
        prediction_latency_ms: float,
        confidence_score: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Record ML model performance metrics.
        
        Args:
            model_name: Name of the model (e.g., 'TimesFM', 'TSMamba')
            prediction_accuracy: Accuracy score (0-1)
            prediction_latency_ms: Prediction latency in milliseconds
            confidence_score: Average confidence score
            timestamp: When recorded (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        model_metrics = {
            'accuracy': prediction_accuracy,
            'latency_ms': prediction_latency_ms,
            'confidence': confidence_score
        }
        
        for metric_type, value in model_metrics.items():
            self.record_system_metric(
                metric_name=f"model_{model_name.lower()}_{metric_type}",
                value=value,
                timestamp=timestamp,
                metadata={'category': 'model_performance', 'model': model_name}
            )
    
    def get_system_health_summary(self, lookback_hours: int = 24) -> Dict[str, Any]:
        """Get system health summary for the last N hours.
        
        Args:
            lookback_hours: Hours to look back for metrics
            
        Returns:
            Dictionary with system health indicators
        """
        cutoff_time = datetime.now() - timedelta(hours=lookback_hours)
        
        # Filter recent metrics
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m['timestamp']) >= cutoff_time
        ]
        
        if not recent_metrics:
            return {'status': 'NO_DATA', 'message': 'No recent metrics available'}
        
        # Calculate health indicators
        health_summary = {
            'status': 'HEALTHY',
            'last_update': max(m['timestamp'] for m in recent_metrics),
            'metrics_count': len(recent_metrics),
            'categories': {}
        }
        
        # Group by category
        categories = {}
        for metric in recent_metrics:
            category = metric['metadata'].get('category', 'system')
            if category not in categories:
                categories[category] = []
            categories[category].append(metric)
        
        # Analyze each category
        for category, metrics in categories.items():
            category_health = self._analyze_category_health(metrics)
            health_summary['categories'][category] = category_health
        
        # Overall health determination
        category_statuses = [cat['status'] for cat in health_summary['categories'].values()]
        if 'CRITICAL' in category_statuses:
            health_summary['status'] = 'CRITICAL'
        elif 'WARNING' in category_statuses:
            health_summary['status'] = 'WARNING'
        else:
            health_summary['status'] = 'HEALTHY'
        
        return health_summary
    
    def get_performance_metrics_summary(self, days_back: int = 30) -> Dict[str, Any]:
        """Get performance metrics summary.
        
        Args:
            days_back: Number of days to look back
            
        Returns:
            Dictionary with performance statistics
        """
        cutoff_time = datetime.now() - timedelta(days=days_back)
        
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m['timestamp']) >= cutoff_time
        ]
        
        if not recent_metrics:
            return {'status': 'NO_DATA'}
        
        # Create DataFrame for analysis
        df = pd.DataFrame(recent_metrics)
        
        # Performance summary
        summary = {
            'period_days': days_back,
            'total_metrics': len(recent_metrics),
            'metric_types': df['metric_name'].nunique(),
            'date_range': {
                'start': df['timestamp'].min(),
                'end': df['timestamp'].max()
            }
        }
        
        # Key performance indicators
        kpis = {}
        
        # Trading performance
        trading_metrics = df[df['metric_name'].str.startswith('trading_')]
        if not trading_metrics.empty:
            portfolio_values = trading_metrics[trading_metrics['metric_name'] == 'trading_portfolio_value']['value']
            if not portfolio_values.empty:
                kpis['portfolio_growth'] = {
                    'start_value': portfolio_values.iloc[0],
                    'end_value': portfolio_values.iloc[-1],
                    'total_return': (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
                }
            
            daily_returns = trading_metrics[trading_metrics['metric_name'] == 'trading_daily_return']['value']
            if not daily_returns.empty:
                kpis['returns_stats'] = {
                    'mean_daily_return': daily_returns.mean(),
                    'std_daily_return': daily_returns.std(),
                    'best_day': daily_returns.max(),
                    'worst_day': daily_returns.min(),
                    'positive_days': (daily_returns > 0).sum(),
                    'negative_days': (daily_returns < 0).sum()
                }
                
                # Calculate Sharpe ratio approximation
                if daily_returns.std() > 0:
                    kpis['sharpe_ratio'] = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
        
        # Model performance
        model_metrics = df[df['metric_name'].str.contains('model_')]
        if not model_metrics.empty:
            accuracy_metrics = model_metrics[model_metrics['metric_name'].str.contains('accuracy')]
            latency_metrics = model_metrics[model_metrics['metric_name'].str.contains('latency')]
            
            if not accuracy_metrics.empty:
                kpis['model_accuracy'] = {
                    'mean': accuracy_metrics['value'].mean(),
                    'std': accuracy_metrics['value'].std(),
                    'latest': accuracy_metrics['value'].iloc[-1]
                }
            
            if not latency_metrics.empty:
                kpis['model_latency'] = {
                    'mean_ms': latency_metrics['value'].mean(),
                    'p95_ms': latency_metrics['value'].quantile(0.95),
                    'latest_ms': latency_metrics['value'].iloc[-1]
                }
        
        summary['kpis'] = kpis
        return summary
    
    def get_real_time_status(self) -> Dict[str, Any]:
        """Get real-time system status.
        
        Returns:
            Current system status and recent metrics
        """
        current_time = datetime.now()
        
        # Get most recent metrics for each type
        latest_metrics = {}
        for metric_name, metric_data in self.current_metrics.items():
            metric_age = current_time - datetime.fromisoformat(metric_data['timestamp'])
            latest_metrics[metric_name] = {
                'value': metric_data['value'],
                'age_seconds': metric_age.total_seconds(),
                'metadata': metric_data['metadata']
            }
        
        # System status
        status = {
            'timestamp': current_time.isoformat(),
            'uptime_indicators': {
                'data_fresh': any(
                    age['age_seconds'] < 3600  # Data within last hour
                    for age in latest_metrics.values()
                ),
                'models_active': any(
                    'model_' in name for name in latest_metrics.keys()
                ),
                'trading_active': any(
                    'trading_' in name for name in latest_metrics.keys()
                )
            },
            'latest_metrics': latest_metrics,
            'alerts': self._generate_alerts(latest_metrics)
        }
        
        return status
    
    def export_metrics(self, start_date: str, end_date: str, output_file: str) -> None:
        """Export metrics to CSV file.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format  
            output_file: Output CSV file path
        """
        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        
        # Filter metrics by date range
        filtered_metrics = [
            m for m in self.metrics_history
            if start_dt <= datetime.fromisoformat(m['timestamp']) <= end_dt
        ]
        
        if not filtered_metrics:
            logger.warning(f"No metrics found for date range {start_date} to {end_date}")
            return
        
        # Convert to DataFrame and export
        df = pd.DataFrame(filtered_metrics)
        df.to_csv(output_file, index=False)
        
        logger.info(f"Exported {len(filtered_metrics)} metrics to {output_file}")
    
    def _load_metrics_history(self) -> None:
        """Load existing metrics from file."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.metrics_history = data.get('metrics', [])
                    
                # Rebuild current_metrics from history
                for metric in self.metrics_history:
                    metric_name = metric['metric_name']
                    if metric_name not in self.current_metrics or \
                       metric['timestamp'] > self.current_metrics[metric_name]['timestamp']:
                        self.current_metrics[metric_name] = {
                            'value': metric['value'],
                            'timestamp': metric['timestamp'],
                            'metadata': metric.get('metadata', {})
                        }
                        
                logger.info(f"Loaded {len(self.metrics_history)} historical metrics")
            except Exception as e:
                logger.warning(f"Could not load metrics history: {e}")
                self.metrics_history = []
    
    def _save_metrics(self) -> None:
        """Save metrics to file."""
        try:
            data = {
                'last_updated': datetime.now().isoformat(),
                'metrics': self.metrics_history
            }
            
            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Could not save metrics: {e}")
    
    def _analyze_category_health(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze health for a category of metrics."""
        if not metrics:
            return {'status': 'NO_DATA', 'message': 'No metrics in category'}
        
        # Calculate basic statistics
        values = [m['value'] for m in metrics]
        timestamps = [datetime.fromisoformat(m['timestamp']) for m in metrics]
        
        # Recency check
        latest_timestamp = max(timestamps)
        age_hours = (datetime.now() - latest_timestamp).total_seconds() / 3600
        
        health_status = 'HEALTHY'
        issues = []
        
        # Check for stale data
        if age_hours > 24:
            health_status = 'WARNING'
            issues.append(f'Data is {age_hours:.1f} hours old')
        elif age_hours > 72:
            health_status = 'CRITICAL'
        
        # Check for anomalous values (basic outlier detection)
        if len(values) >= 3:
            q75, q25 = np.percentile(values, [75, 25])
            iqr = q75 - q25
            outlier_threshold = q75 + 1.5 * iqr
            
            recent_values = values[-5:]  # Last 5 values
            if any(v > outlier_threshold for v in recent_values):
                health_status = 'WARNING'
                issues.append('Anomalous values detected')
        
        return {
            'status': health_status,
            'issues': issues,
            'stats': {
                'count': len(metrics),
                'latest_value': values[-1],
                'mean': np.mean(values),
                'std': np.std(values),
                'age_hours': age_hours
            }
        }
    
    def _generate_alerts(self, latest_metrics: Dict) -> List[Dict[str, Any]]:
        """Generate alerts based on current metrics."""
        alerts = []
        current_time = datetime.now()
        
        # Check for stale data
        for metric_name, metric_info in latest_metrics.items():
            age_seconds = metric_info['age_seconds']
            
            if age_seconds > 3600:  # 1 hour
                alerts.append({
                    'severity': 'WARNING',
                    'type': 'STALE_DATA',
                    'message': f'{metric_name} is {age_seconds/3600:.1f} hours old',
                    'metric': metric_name
                })
        
        # Check for performance issues
        for metric_name, metric_info in latest_metrics.items():
            value = metric_info['value']
            
            # Model latency alerts
            if 'latency_ms' in metric_name and value > 1000:
                alerts.append({
                    'severity': 'WARNING',
                    'type': 'HIGH_LATENCY',
                    'message': f'{metric_name} is high: {value}ms',
                    'metric': metric_name
                })
            
            # Trading performance alerts
            if metric_name == 'trading_drawdown' and value > 0.1:
                alerts.append({
                    'severity': 'CRITICAL',
                    'type': 'HIGH_DRAWDOWN',
                    'message': f'Portfolio drawdown is {value*100:.1f}%',
                    'metric': metric_name
                })
        
        return alerts


def create_sample_dashboard_data():
    """Create sample data for dashboard testing."""
    dashboard = PerformanceDashboard()
    
    # Simulate some metrics over the past week
    base_time = datetime.now() - timedelta(days=7)
    
    for i in range(7 * 24):  # Hourly metrics for a week
        timestamp = base_time + timedelta(hours=i)
        
        # Trading performance
        portfolio_value = 50000 + np.random.normal(1000, 500) * i / (7 * 24) * 10
        daily_return = np.random.normal(0.001, 0.02)
        drawdown = max(0, np.random.normal(0.02, 0.01))
        
        dashboard.record_trading_performance(
            portfolio_value=portfolio_value,
            daily_return=daily_return,
            drawdown=drawdown,
            timestamp=timestamp
        )
        
        # Model performance
        if i % 4 == 0:  # Every 4 hours
            dashboard.record_model_performance(
                model_name='TimesFM',
                prediction_accuracy=np.random.normal(0.58, 0.05),
                prediction_latency_ms=np.random.normal(50, 10),
                confidence_score=np.random.normal(0.75, 0.1),
                timestamp=timestamp
            )
        
        # System metrics
        if i % 2 == 0:  # Every 2 hours
            dashboard.record_system_metric(
                'data_ingestion_latency',
                np.random.normal(2.5, 0.5),
                timestamp=timestamp,
                metadata={'category': 'system'}
            )
    
    logger.info("Sample dashboard data created")
    return dashboard


if __name__ == '__main__':
    # Demo usage
    dashboard = create_sample_dashboard_data()
    
    # Print system health
    health = dashboard.get_system_health_summary()
    print("System Health:", json.dumps(health, indent=2))
    
    # Print performance summary
    performance = dashboard.get_performance_metrics_summary()
    print("Performance Summary:", json.dumps(performance, indent=2))
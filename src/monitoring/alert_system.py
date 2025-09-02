#!/usr/bin/env python3
"""
Automated Alert System for AI Trading System
Real-time monitoring and notification system for portfolio events.
"""

import sys
import json
import smtplib
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from threading import Thread, Event
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from monitoring.portfolio_dashboard import PortfolioDashboard, Alert, AlertLevel
from config.settings import config


class NotificationChannel(Enum):
    """Available notification channels."""
    EMAIL = "email"
    CONSOLE = "console"
    FILE = "file"
    WEBHOOK = "webhook"


@dataclass
class NotificationConfig:
    """Notification configuration."""
    enabled: bool = True
    channels: List[NotificationChannel] = None
    email_settings: Dict[str, str] = None
    webhook_url: Optional[str] = None
    file_path: Optional[str] = None
    throttle_minutes: int = 15  # Minimum time between similar alerts
    
    def __post_init__(self):
        if self.channels is None:
            self.channels = [NotificationChannel.CONSOLE, NotificationChannel.FILE]
        if self.email_settings is None:
            self.email_settings = {}


class AlertThrottler:
    """Throttle similar alerts to prevent spam."""
    
    def __init__(self):
        self.last_alerts: Dict[str, datetime] = {}
    
    def should_send_alert(self, alert_key: str, throttle_minutes: int) -> bool:
        """Check if alert should be sent based on throttling rules."""
        now = datetime.now()
        
        if alert_key in self.last_alerts:
            time_since_last = now - self.last_alerts[alert_key]
            if time_since_last.total_seconds() < (throttle_minutes * 60):
                return False
        
        self.last_alerts[alert_key] = now
        return True
    
    def clear_old_alerts(self, hours: int = 24):
        """Clear old alert timestamps."""
        cutoff = datetime.now() - timedelta(hours=hours)
        self.last_alerts = {
            key: timestamp for key, timestamp in self.last_alerts.items()
            if timestamp > cutoff
        }


class AlertNotifier:
    """Handle alert notifications across different channels."""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.throttler = AlertThrottler()
        
        # Setup logging
        self.alert_log = Path("logs/alerts.log")
        self.alert_log.parent.mkdir(exist_ok=True)
        
        # Email setup if configured
        self.email_configured = self._validate_email_config()
        
    def send_alert(self, alert: Alert, context: Dict[str, Any] = None) -> bool:
        """Send alert through configured channels."""
        if not self.config.enabled:
            return False
        
        # Create alert key for throttling
        alert_key = f"{alert.metric_name}_{alert.level.value}"
        
        # Check throttling
        if not self.throttler.should_send_alert(alert_key, self.config.throttle_minutes):
            logger.debug(f"Alert throttled: {alert.title}")
            return False
        
        success = True
        
        # Send through each configured channel
        for channel in self.config.channels:
            try:
                if channel == NotificationChannel.CONSOLE:
                    self._send_console_alert(alert, context)
                elif channel == NotificationChannel.FILE:
                    self._send_file_alert(alert, context)
                elif channel == NotificationChannel.EMAIL and self.email_configured:
                    self._send_email_alert(alert, context)
                elif channel == NotificationChannel.WEBHOOK and self.config.webhook_url:
                    self._send_webhook_alert(alert, context)
                    
            except Exception as e:
                logger.error(f"Failed to send alert via {channel.value}: {e}")
                success = False
        
        # Clean up old throttled alerts periodically
        if len(self.throttler.last_alerts) > 100:
            self.throttler.clear_old_alerts()
        
        return success
    
    def _validate_email_config(self) -> bool:
        """Validate email configuration."""
        required_fields = ['smtp_server', 'smtp_port', 'from_email', 'to_email']
        return all(field in self.config.email_settings for field in required_fields)
    
    def _send_console_alert(self, alert: Alert, context: Dict[str, Any] = None):
        """Send alert to console."""
        emoji_map = {
            AlertLevel.CRITICAL: "🔴",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.INFO: "ℹ️"
        }
        
        emoji = emoji_map.get(alert.level, "⚠️")
        timestamp = alert.timestamp.strftime("%H:%M:%S")
        
        print(f"\n{emoji} ALERT [{timestamp}] {alert.title}")
        print(f"   {alert.message}")
        
        if alert.action_required:
            print("   🎯 ACTION REQUIRED")
        
        if context:
            portfolio_value = context.get('portfolio_value', 0)
            if portfolio_value:
                print(f"   💰 Portfolio Value: ${portfolio_value:,.2f}")
        
        print()
    
    def _send_file_alert(self, alert: Alert, context: Dict[str, Any] = None):
        """Send alert to log file."""
        timestamp = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = {
            'timestamp': timestamp,
            'level': alert.level.value,
            'title': alert.title,
            'message': alert.message,
            'metric': alert.metric_name,
            'current_value': alert.current_value,
            'threshold_value': alert.threshold_value,
            'action_required': alert.action_required
        }
        
        if context:
            log_entry['context'] = context
        
        try:
            with open(self.alert_log, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write alert to file: {e}")
    
    def _send_email_alert(self, alert: Alert, context: Dict[str, Any] = None):
        """Send alert via email."""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.email_settings['from_email']
            msg['To'] = self.config.email_settings['to_email']
            msg['Subject'] = f"Trading Alert: {alert.title}"
            
            # Create email body
            body = self._create_email_body(alert, context)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(
                self.config.email_settings['smtp_server'],
                self.config.email_settings['smtp_port']
            )
            
            if self.config.email_settings.get('use_tls', True):
                server.starttls()
            
            if 'username' in self.config.email_settings:
                server.login(
                    self.config.email_settings['username'],
                    self.config.email_settings['password']
                )
            
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            raise
    
    def _create_email_body(self, alert: Alert, context: Dict[str, Any] = None) -> str:
        """Create HTML email body."""
        level_colors = {
            AlertLevel.CRITICAL: "#FF4444",
            AlertLevel.WARNING: "#FFA500", 
            AlertLevel.INFO: "#4CAF50"
        }
        
        level_color = level_colors.get(alert.level, "#FFA500")
        timestamp = alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .alert-header {{ background-color: {level_color}; color: white; padding: 15px; border-radius: 5px; }}
                .alert-body {{ padding: 20px; border: 1px solid #ddd; border-radius: 5px; margin-top: 10px; }}
                .metric-table {{ border-collapse: collapse; width: 100%; margin-top: 15px; }}
                .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .metric-table th {{ background-color: #f2f2f2; }}
                .action-required {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; margin-top: 10px; }}
            </style>
        </head>
        <body>
            <div class="alert-header">
                <h2>🚨 Trading System Alert</h2>
                <p><strong>{alert.title}</strong></p>
                <p>Level: {alert.level.value.upper()} | Time: {timestamp}</p>
            </div>
            
            <div class="alert-body">
                <p><strong>Message:</strong> {alert.message}</p>
                
                <table class="metric-table">
                    <tr>
                        <th>Metric</th>
                        <td>{alert.metric_name}</td>
                    </tr>
                    <tr>
                        <th>Current Value</th>
                        <td>{alert.current_value:.2f}</td>
                    </tr>
                    <tr>
                        <th>Threshold</th>
                        <td>{alert.threshold_value:.2f}</td>
                    </tr>
                </table>
        """
        
        if context:
            html += """
                <h3>Portfolio Context</h3>
                <table class="metric-table">
            """
            
            if 'portfolio_value' in context:
                html += f"""
                    <tr>
                        <th>Portfolio Value</th>
                        <td>${context['portfolio_value']:,.2f}</td>
                    </tr>
                """
            
            if 'daily_return_pct' in context:
                html += f"""
                    <tr>
                        <th>Daily Return</th>
                        <td>{context['daily_return_pct']:+.2f}%</td>
                    </tr>
                """
            
            if 'total_return_pct' in context:
                html += f"""
                    <tr>
                        <th>Total Return</th>
                        <td>{context['total_return_pct']:+.2f}%</td>
                    </tr>
                """
            
            html += "</table>"
        
        if alert.action_required:
            html += """
                <div class="action-required">
                    <strong>⚠️ ACTION REQUIRED:</strong> This alert requires immediate attention and action.
                </div>
            """
        
        html += """
            <p><em>This alert was generated by the AI Trading System automated monitoring.</em></p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _send_webhook_alert(self, alert: Alert, context: Dict[str, Any] = None):
        """Send alert via webhook."""
        try:
            import requests
            
            payload = {
                'timestamp': alert.timestamp.isoformat(),
                'level': alert.level.value,
                'title': alert.title,
                'message': alert.message,
                'metric_name': alert.metric_name,
                'current_value': alert.current_value,
                'threshold_value': alert.threshold_value,
                'action_required': alert.action_required
            }
            
            if context:
                payload['context'] = context
            
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                timeout=10
            )
            
            response.raise_for_status()
            logger.info(f"Webhook alert sent: {alert.title}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            raise


class AlertMonitor:
    """Main alert monitoring system."""
    
    def __init__(self, portfolio_file: str = "portfolio.toml", config_file: str = "alert_config.json"):
        self.portfolio_file = portfolio_file
        self.dashboard = PortfolioDashboard(portfolio_file)
        
        # Load configuration
        self.config = self._load_config(config_file)
        self.notifier = AlertNotifier(self.config)
        
        # Monitoring state
        self.monitoring = Event()
        self.monitor_thread = None
        
        # Alert history for trend analysis
        self.alert_history: List[Alert] = []
    
    def _load_config(self, config_file: str) -> NotificationConfig:
        """Load alert configuration from file."""
        config_path = Path(config_file)
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                return NotificationConfig(
                    enabled=config_data.get('enabled', True),
                    channels=[NotificationChannel(ch) for ch in config_data.get('channels', ['console', 'file'])],
                    email_settings=config_data.get('email_settings', {}),
                    webhook_url=config_data.get('webhook_url'),
                    file_path=config_data.get('file_path'),
                    throttle_minutes=config_data.get('throttle_minutes', 15)
                )
                
            except Exception as e:
                logger.warning(f"Could not load alert config: {e}")
        
        # Return default configuration
        return NotificationConfig()
    
    def create_default_config(self, config_file: str = "alert_config.json"):
        """Create default alert configuration file."""
        default_config = {
            "enabled": True,
            "channels": ["console", "file"],
            "throttle_minutes": 15,
            "email_settings": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "use_tls": True,
                "from_email": "your_email@gmail.com",
                "to_email": "your_email@gmail.com",
                "username": "your_email@gmail.com",
                "password": "your_app_password"
            },
            "webhook_url": null
        }
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"Created default alert configuration: {config_file}")
        print("Please edit the file to configure your notification settings.")
    
    def check_alerts_once(self) -> List[Alert]:
        """Run single alert check with enhanced real-time trade signal detection."""
        try:
            # Generate dashboard report
            report = self.dashboard.generate_dashboard_report(update_prices=True)
            
            if 'error' in report:
                logger.error(f"Dashboard error: {report['error']}")
                return []
            
            # Get alerts from report
            alerts = [Alert(
                timestamp=datetime.fromisoformat(alert_data['timestamp']),
                level=AlertLevel(alert_data['level']),
                title=alert_data['title'],
                message=alert_data['message'],
                metric_name=alert_data['metric_name'],
                current_value=alert_data['current_value'],
                threshold_value=alert_data['threshold_value'],
                action_required=alert_data['action_required']
            ) for alert_data in report.get('alerts', [])]
            
            # Add enhanced real-time trade signal alerts
            enhanced_alerts = self._check_real_time_signals(report)
            alerts.extend(enhanced_alerts)
            
            # Send notifications
            if alerts:
                context = {
                    'portfolio_value': report.get('metrics', {}).get('total_value', 0),
                    'daily_return_pct': report.get('metrics', {}).get('daily_return_pct', 0),
                    'total_return_pct': report.get('metrics', {}).get('total_return_pct', 0),
                    'timestamp': datetime.now().isoformat(),
                    'volatility': report.get('metrics', {}).get('volatility', 0),
                    'active_positions': report.get('metrics', {}).get('active_positions', 0)
                }
                
                for alert in alerts:
                    self.notifier.send_alert(alert, context)
                
                # Store alerts in history
                self.alert_history.extend(alerts)
                
                # Keep only recent history (last 1000 alerts)
                if len(self.alert_history) > 1000:
                    self.alert_history = self.alert_history[-1000:]
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
            return []
    
    def _check_real_time_signals(self, report: Dict[str, Any]) -> List[Alert]:
        """Enhanced real-time trade signal detection for revenue optimization."""
        signals = []
        
        try:
            metrics = report.get('metrics', {})
            positions = report.get('positions', [])
            
            # 1. High-Momentum Entry Signals (Revenue Critical)
            for position in positions:
                if position.get('momentum_score', 0) > 0.75:
                    signals.append(Alert(
                        timestamp=datetime.now(),
                        level=AlertLevel.WARNING,
                        title=f"🚀 High Momentum Entry Signal: {position['symbol']}",
                        message=f"Strong momentum detected (Score: {position['momentum_score']:.2f}). Consider position increase for revenue optimization.",
                        metric_name="momentum_score",
                        current_value=position['momentum_score'],
                        threshold_value=0.75,
                        action_required=True
                    ))
            
            # 2. Profit Target Achievement Alerts
            portfolio_return = metrics.get('total_return_pct', 0)
            if portfolio_return >= 2.5:  # Monthly revenue target approach
                signals.append(Alert(
                    timestamp=datetime.now(),
                    level=AlertLevel.INFO,
                    title="🎯 Monthly Revenue Target Approached",
                    message=f"Portfolio return: {portfolio_return:.2f}%. Consider profit taking strategies.",
                    metric_name="total_return_pct",
                    current_value=portfolio_return,
                    threshold_value=2.5,
                    action_required=True
                ))
            
            # 3. Volatility-Based Position Sizing Alerts
            volatility = metrics.get('volatility', 0)
            if volatility > 25:  # High volatility requires position adjustments
                signals.append(Alert(
                    timestamp=datetime.now(),
                    level=AlertLevel.WARNING,
                    title="⚠️ High Market Volatility Detected",
                    message=f"Market volatility: {volatility:.1f}%. Consider reducing position sizes for risk management.",
                    metric_name="volatility",
                    current_value=volatility,
                    threshold_value=25,
                    action_required=True
                ))
            
            # 4. Sector Rotation Opportunities
            sector_performance = metrics.get('sector_performance', {})
            if sector_performance:
                best_sector = max(sector_performance.items(), key=lambda x: x[1])
                worst_sector = min(sector_performance.items(), key=lambda x: x[1])
                
                performance_gap = best_sector[1] - worst_sector[1]
                if performance_gap > 5:  # Significant sector divergence
                    signals.append(Alert(
                        timestamp=datetime.now(),
                        level=AlertLevel.INFO,
                        title="🔄 Sector Rotation Opportunity",
                        message=f"Strong sector divergence detected: {best_sector[0]} (+{best_sector[1]:.1f}%) vs {worst_sector[0]} ({worst_sector[1]:+.1f}%). Consider rebalancing.",
                        metric_name="sector_divergence",
                        current_value=performance_gap,
                        threshold_value=5.0,
                        action_required=True
                    ))
            
            # 5. Kelly Criterion Position Sizing Alerts
            for position in positions:
                current_weight = position.get('weight', 0)
                optimal_kelly_weight = position.get('kelly_optimal', 0)
                
                if abs(current_weight - optimal_kelly_weight) > 0.05:  # 5% deviation
                    signals.append(Alert(
                        timestamp=datetime.now(),
                        level=AlertLevel.INFO,
                        title=f"📊 Kelly Rebalancing Signal: {position['symbol']}",
                        message=f"Current: {current_weight:.1%}, Kelly Optimal: {optimal_kelly_weight:.1%}. Consider rebalancing for optimal growth.",
                        metric_name="kelly_deviation",
                        current_value=abs(current_weight - optimal_kelly_weight),
                        threshold_value=0.05,
                        action_required=True
                    ))
            
            # 6. Market Regime Change Detection
            daily_return = metrics.get('daily_return_pct', 0)
            if abs(daily_return) > 3:  # Significant daily move
                regime_signal = "Bull Market Acceleration" if daily_return > 0 else "Bear Market Warning"
                signals.append(Alert(
                    timestamp=datetime.now(),
                    level=AlertLevel.CRITICAL if daily_return < -3 else AlertLevel.WARNING,
                    title=f"📈 Market Regime Signal: {regime_signal}",
                    message=f"Significant daily move: {daily_return:+.2f}%. Review portfolio positioning and risk exposure.",
                    metric_name="daily_return_pct",
                    current_value=abs(daily_return),
                    threshold_value=3.0,
                    action_required=True
                ))
            
        except Exception as e:
            logger.error(f"Error generating enhanced trade signals: {e}")
        
        return signals
    
    def start_monitoring(self, check_interval_minutes: int = 5):
        """Start continuous alert monitoring."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.warning("Monitoring already started")
            return
        
        self.monitoring.set()
        self.monitor_thread = Thread(
            target=self._monitor_loop,
            args=(check_interval_minutes,),
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info(f"Alert monitoring started (checking every {check_interval_minutes} minutes)")
    
    def stop_monitoring(self):
        """Stop continuous alert monitoring."""
        if self.monitoring.is_set():
            self.monitoring.clear()
            logger.info("Alert monitoring stopped")
    
    def _monitor_loop(self, check_interval_minutes: int):
        """Main monitoring loop."""
        while self.monitoring.is_set():
            try:
                alerts = self.check_alerts_once()
                
                if alerts:
                    logger.info(f"Found {len(alerts)} alerts")
                else:
                    logger.debug("No alerts found")
                
                # Wait for next check
                for _ in range(check_interval_minutes * 60):
                    if not self.monitoring.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent alerts."""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_alerts = [alert for alert in self.alert_history if alert.timestamp > cutoff]
        
        if not recent_alerts:
            return {
                'total_alerts': 0,
                'by_level': {},
                'by_metric': {},
                'most_frequent': None
            }
        
        # Count by level
        by_level = {}
        for alert in recent_alerts:
            level = alert.level.value
            by_level[level] = by_level.get(level, 0) + 1
        
        # Count by metric
        by_metric = {}
        for alert in recent_alerts:
            metric = alert.metric_name
            by_metric[metric] = by_metric.get(metric, 0) + 1
        
        # Find most frequent alert type
        most_frequent = max(by_metric.items(), key=lambda x: x[1]) if by_metric else None
        
        return {
            'total_alerts': len(recent_alerts),
            'by_level': by_level,
            'by_metric': by_metric,
            'most_frequent': most_frequent,
            'time_range_hours': hours
        }


def main():
    """Main entry point for alert system CLI."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Alert Monitoring System")
    parser.add_argument("--portfolio", "-p", default="portfolio.toml", help="Portfolio file")
    parser.add_argument("--config", "-c", default="alert_config.json", help="Alert configuration file")
    parser.add_argument("--check-once", action="store_true", help="Run single alert check")
    parser.add_argument("--monitor", "-m", type=int, metavar="MINUTES", help="Start continuous monitoring (check interval in minutes)")
    parser.add_argument("--create-config", action="store_true", help="Create default configuration file")
    parser.add_argument("--summary", "-s", type=int, metavar="HOURS", help="Show alert summary for last N hours")
    
    args = parser.parse_args()
    
    # Create default config if requested
    if args.create_config:
        AlertMonitor(args.portfolio, args.config).create_default_config(args.config)
        return
    
    monitor = AlertMonitor(args.portfolio, args.config)
    
    try:
        if args.check_once:
            print("🔍 Running single alert check...")
            alerts = monitor.check_alerts_once()
            print(f"✅ Check complete. Found {len(alerts)} alerts.")
            
        elif args.monitor:
            print(f"🔄 Starting continuous monitoring (every {args.monitor} minutes)")
            print("Press Ctrl+C to stop...")
            
            monitor.start_monitoring(args.monitor)
            
            try:
                while monitor.monitoring.is_set():
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n⚠️ Stopping monitoring...")
                monitor.stop_monitoring()
                
        elif args.summary:
            summary = monitor.get_alert_summary(args.summary)
            print(f"\n📊 Alert Summary (Last {args.summary} hours)")
            print("-" * 40)
            print(f"Total Alerts: {summary['total_alerts']}")
            
            if summary['by_level']:
                print("\nBy Level:")
                for level, count in summary['by_level'].items():
                    print(f"  {level.upper()}: {count}")
            
            if summary['by_metric']:
                print("\nBy Metric:")
                for metric, count in summary['by_metric'].items():
                    print(f"  {metric}: {count}")
            
            if summary['most_frequent']:
                metric, count = summary['most_frequent']
                print(f"\nMost Frequent: {metric} ({count} times)")
            
            print()
            
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
    except Exception as e:
        logger.error(f"Alert system error: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
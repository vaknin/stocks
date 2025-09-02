#!/usr/bin/env python3
"""
System Health Validator for AI Trading System
Comprehensive validation of system setup, models, and trading components.
"""

import sys
import os
import json
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SystemValidator:
    """Comprehensive system health and setup validation."""
    
    def __init__(self):
        self.results = {
            'system_info': {},
            'environment': {},
            'dependencies': {},
            'models': {},
            'trading_components': {},
            'data_sources': {},
            'overall_status': 'unknown',
            'critical_issues': [],
            'warnings': [],
            'recommendations': []
        }
        
    def validate_all(self) -> Dict[str, Any]:
        """Run comprehensive system validation."""
        print("🔍 AI Trading System - Health Check")
        print("=" * 50)
        
        try:
            # System environment validation
            self._validate_system_environment()
            
            # Python and dependency validation  
            self._validate_python_environment()
            
            # AI model validation
            self._validate_ai_models()
            
            # Trading system validation
            self._validate_trading_system()
            
            # Data source validation
            self._validate_data_sources()
            
            # Performance validation
            self._validate_system_performance()
            
            # Generate overall assessment
            self._generate_overall_assessment()
            
        except Exception as e:
            self.results['critical_issues'].append(f"Validation failed: {e}")
            self.results['overall_status'] = 'failed'
        
        return self.results
    
    def _validate_system_environment(self):
        """Validate system environment and resources."""
        print("📋 Validating System Environment...")
        
        # Check CUDA availability
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_version = torch.version.cuda if cuda_available else None
            device_count = torch.cuda.device_count() if cuda_available else 0
            
            self.results['system_info'].update({
                'cuda_available': cuda_available,
                'cuda_version': cuda_version,
                'gpu_count': device_count,
                'torch_version': torch.__version__
            })
            
            if cuda_available:
                print(f"  ✅ CUDA {cuda_version} available with {device_count} GPU(s)")
            else:
                print("  ⚠️ CUDA not available - using CPU only")
                self.results['warnings'].append("GPU acceleration unavailable")
                
        except ImportError:
            self.results['critical_issues'].append("PyTorch not installed")
            print("  ❌ PyTorch not found")
        
        # Check memory
        try:
            import psutil
            memory = psutil.virtual_memory()
            self.results['system_info']['memory_gb'] = memory.total / (1024**3)
            self.results['system_info']['available_memory_gb'] = memory.available / (1024**3)
            
            if memory.available < 2 * (1024**3):  # Less than 2GB
                self.results['warnings'].append("Low available memory (<2GB)")
                print(f"  ⚠️ Available memory: {memory.available/(1024**3):.1f}GB")
            else:
                print(f"  ✅ Available memory: {memory.available/(1024**3):.1f}GB")
                
        except ImportError:
            self.results['warnings'].append("Cannot check system memory")
    
    def _validate_python_environment(self):
        """Validate Python version and critical dependencies."""
        print("🐍 Validating Python Environment...")
        
        # Python version
        python_version = sys.version.split()[0]
        self.results['environment']['python_version'] = python_version
        
        if not python_version.startswith('3.10') and not python_version.startswith('3.11'):
            self.results['warnings'].append(f"Python {python_version} - recommend 3.10 or 3.11")
            print(f"  ⚠️ Python {python_version} (recommend 3.10 or 3.11)")
        else:
            print(f"  ✅ Python {python_version}")
        
        # Critical dependencies
        critical_deps = {
            'torch': '2.7.1',
            'transformers': '4.44.0',
            'timesfm': '1.3.0',
            'yfinance': '0.2.65',
            'pandas': '2.3.2',
            'numpy': '2.2.6',
            'sklearn': '1.5.2',
            'pydantic': '2.11.7'
        }
        
        for package, expected_version in critical_deps.items():
            try:
                if package == 'sklearn':
                    import sklearn
                    version = sklearn.__version__
                    module = sklearn
                else:
                    module = importlib.import_module(package)
                    version = getattr(module, '__version__', 'unknown')
                
                self.results['dependencies'][package] = version
                
                if version.startswith(expected_version.split('.')[0]):  # Major version match
                    print(f"  ✅ {package}: {version}")
                else:
                    print(f"  ⚠️ {package}: {version} (expected {expected_version})")
                    self.results['warnings'].append(f"{package} version mismatch")
                    
            except ImportError:
                self.results['critical_issues'].append(f"Missing {package}")
                print(f"  ❌ {package}: Not installed")
    
    def _validate_ai_models(self):
        """Validate AI model loading and basic functionality."""
        print("🤖 Validating AI Models...")
        
        # TimesFM Model
        try:
            sys.path.append(str(Path(__file__).parent.parent / "src"))
            from src.models.timesfm_predictor import TimesFMPredictor
            
            print("  📥 Loading TimesFM model...")
            predictor = TimesFMPredictor()
            
            # Test with dummy data
            import pandas as pd
            import numpy as np
            
            test_data = pd.DataFrame({
                'close': np.random.randn(100).cumsum() + 100
            }, index=pd.date_range('2023-01-01', periods=100))
            
            result = predictor.predict(test_data)
            
            if result and 'prediction' in result:
                print("  ✅ TimesFM: Loaded and functional")
                self.results['models']['timesfm'] = 'operational'
            else:
                print("  ⚠️ TimesFM: Loaded but prediction failed")
                self.results['models']['timesfm'] = 'degraded'
                self.results['warnings'].append("TimesFM prediction issues")
                
        except Exception as e:
            print(f"  ❌ TimesFM: {str(e)[:50]}...")
            self.results['models']['timesfm'] = 'failed'
            self.results['critical_issues'].append("TimesFM model failure")
        
        # TSMamba Model
        try:
            from src.models.tsmamba_predictor import TSMambaPredictor
            predictor = TSMambaPredictor()
            
            # Test basic functionality
            test_data = pd.DataFrame({
                'close': np.random.randn(50).cumsum() + 100,
                'volume': np.random.randint(1000, 10000, 50)
            }, index=pd.date_range('2023-01-01', periods=50))
            
            result = predictor.predict(test_data)
            
            if result and 'prediction' in result:
                print("  ✅ TSMamba: Loaded and functional")  
                self.results['models']['tsmamba'] = 'operational'
            else:
                print("  ⚠️ TSMamba: Loaded but prediction failed")
                self.results['models']['tsmamba'] = 'degraded'
                
        except Exception as e:
            print(f"  ❌ TSMamba: {str(e)[:50]}...")
            self.results['models']['tsmamba'] = 'failed'
            self.results['warnings'].append("TSMamba model issues")
        
        # SAMBA Model
        try:
            from src.models.samba_predictor import SAMBAPredictor
            predictor = SAMBAPredictor()
            
            # Test with multiple stocks
            test_stocks = ['AAPL', 'MSFT', 'GOOGL']
            test_data = {}
            for stock in test_stocks:
                test_data[stock] = pd.DataFrame({
                    'close': np.random.randn(30).cumsum() + 100
                }, index=pd.date_range('2023-01-01', periods=30))
            
            result = predictor.predict(test_data, target_stock='AAPL')
            
            if result and 'prediction' in result:
                print("  ✅ SAMBA: Loaded and functional")
                self.results['models']['samba'] = 'operational'  
            else:
                print("  ⚠️ SAMBA: Loaded but prediction failed")
                self.results['models']['samba'] = 'degraded'
                
        except Exception as e:
            print(f"  ❌ SAMBA: {str(e)[:50]}...")
            self.results['models']['samba'] = 'failed'
            self.results['warnings'].append("SAMBA model issues")
    
    def _validate_trading_system(self):
        """Validate trading system components."""
        print("📈 Validating Trading System...")
        
        # Risk Manager
        try:
            from src.trading.risk_manager import RiskManager
            risk_manager = RiskManager()
            
            # Test basic risk check
            test_result = risk_manager.check_position_limits('TEST', 1000, 0.1)
            print("  ✅ Risk Manager: Operational")
            self.results['trading_components']['risk_manager'] = 'operational'
            
        except Exception as e:
            print(f"  ❌ Risk Manager: {str(e)[:50]}...")
            self.results['trading_components']['risk_manager'] = 'failed'
            self.results['critical_issues'].append("Risk management failure")
        
        # Position Sizer
        try:
            from src.trading.position_sizer import KellyPositionSizer
            sizer = KellyPositionSizer()
            
            # Test position sizing
            size = sizer.calculate_position_size(
                confidence=0.8,
                expected_return=0.05,
                volatility=0.2,
                portfolio_value=10000
            )
            
            if size > 0:
                print("  ✅ Position Sizer: Operational")
                self.results['trading_components']['position_sizer'] = 'operational'
            else:
                print("  ⚠️ Position Sizer: Issues detected")
                self.results['trading_components']['position_sizer'] = 'degraded'
                
        except Exception as e:
            print(f"  ❌ Position Sizer: {str(e)[:50]}...")
            self.results['trading_components']['position_sizer'] = 'failed'
        
        # Portfolio Tracker
        try:
            from src.trading.portfolio_tracker import PortfolioTracker
            tracker = PortfolioTracker("test_portfolio.toml")
            
            print("  ✅ Portfolio Tracker: Operational")
            self.results['trading_components']['portfolio_tracker'] = 'operational'
            
        except Exception as e:
            print(f"  ❌ Portfolio Tracker: {str(e)[:50]}...")
            self.results['trading_components']['portfolio_tracker'] = 'failed'
        
        # Recommendation Engine
        try:
            from src.trading.recommendation_engine import TradingRecommendationEngine
            print("  ✅ Recommendation Engine: Loaded")
            self.results['trading_components']['recommendation_engine'] = 'operational'
            
        except Exception as e:
            print(f"  ❌ Recommendation Engine: {str(e)[:50]}...")
            self.results['trading_components']['recommendation_engine'] = 'failed'
    
    def _validate_data_sources(self):
        """Validate data source connectivity and quality."""
        print("📊 Validating Data Sources...")
        
        # Yahoo Finance
        try:
            import yfinance as yf
            
            # Test data fetch
            test_ticker = yf.Ticker("AAPL")
            data = test_ticker.history(period="5d")
            
            if not data.empty and len(data) >= 3:
                print("  ✅ Yahoo Finance: Operational")
                self.results['data_sources']['yahoo_finance'] = 'operational'
                
                # Check data freshness
                latest_date = data.index[-1].date()
                days_old = (datetime.now().date() - latest_date).days
                
                if days_old > 3:
                    self.results['warnings'].append(f"Market data is {days_old} days old")
                    print(f"  ⚠️ Data freshness: {days_old} days old")
                else:
                    print(f"  ✅ Data freshness: {days_old} days old")
                    
            else:
                print("  ❌ Yahoo Finance: No data returned")
                self.results['data_sources']['yahoo_finance'] = 'failed'
                self.results['critical_issues'].append("Market data unavailable")
                
        except Exception as e:
            print(f"  ❌ Yahoo Finance: {str(e)[:50]}...")
            self.results['data_sources']['yahoo_finance'] = 'failed'
            self.results['critical_issues'].append("Market data connection failed")
        
        # Database connectivity
        try:
            from src.data.database import DatabaseManager
            db = DatabaseManager()
            
            print("  ✅ Database: Accessible")
            self.results['data_sources']['database'] = 'operational'
            
        except Exception as e:
            print(f"  ⚠️ Database: {str(e)[:50]}...")
            self.results['data_sources']['database'] = 'degraded'
            self.results['warnings'].append("Database connection issues")
    
    def _validate_system_performance(self):
        """Validate system performance and response times."""
        print("⚡ Validating System Performance...")
        
        try:
            import time
            
            # Test prediction speed
            from src.models.timesfm_predictor import TimesFMPredictor
            predictor = TimesFMPredictor()
            
            # Generate test data
            test_data = pd.DataFrame({
                'close': np.random.randn(100).cumsum() + 100
            }, index=pd.date_range('2023-01-01', periods=100))
            
            start_time = time.time()
            result = predictor.predict(test_data)
            prediction_time = time.time() - start_time
            
            self.results['system_info']['prediction_latency'] = prediction_time
            
            if prediction_time < 5.0:  # Under 5 seconds
                print(f"  ✅ Prediction Speed: {prediction_time:.2f}s")
            else:
                print(f"  ⚠️ Prediction Speed: {prediction_time:.2f}s (slow)")
                self.results['warnings'].append("Slow prediction performance")
                
        except Exception as e:
            print(f"  ❌ Performance Test Failed: {str(e)[:50]}...")
            self.results['warnings'].append("Cannot measure performance")
    
    def _generate_overall_assessment(self):
        """Generate overall system health assessment."""
        print("\n📋 System Health Summary")
        print("=" * 30)
        
        critical_count = len(self.results['critical_issues'])
        warning_count = len(self.results['warnings'])
        
        # Determine overall status
        if critical_count == 0:
            if warning_count == 0:
                self.results['overall_status'] = 'excellent'
                status_emoji = "🟢"
                status_text = "EXCELLENT"
            elif warning_count <= 3:
                self.results['overall_status'] = 'good'  
                status_emoji = "🟡"
                status_text = "GOOD"
            else:
                self.results['overall_status'] = 'fair'
                status_emoji = "🟠"
                status_text = "FAIR"
        else:
            self.results['overall_status'] = 'poor'
            status_emoji = "🔴"
            status_text = "POOR"
        
        print(f"{status_emoji} Overall Status: {status_text}")
        print(f"Critical Issues: {critical_count}")  
        print(f"Warnings: {warning_count}")
        
        # Report critical issues
        if self.results['critical_issues']:
            print("\n❌ Critical Issues:")
            for issue in self.results['critical_issues']:
                print(f"  • {issue}")
        
        # Report warnings
        if self.results['warnings']:
            print("\n⚠️ Warnings:")
            for warning in self.results['warnings']:
                print(f"  • {warning}")
        
        # Generate recommendations
        self._generate_recommendations()
        
        if self.results['recommendations']:
            print("\n💡 Recommendations:")
            for rec in self.results['recommendations']:
                print(f"  • {rec}")
    
    def _generate_recommendations(self):
        """Generate actionable recommendations based on validation results."""
        recs = []
        
        if self.results['critical_issues']:
            recs.append("Address critical issues before trading")
        
        if 'timesfm' in self.results['models'] and self.results['models']['timesfm'] != 'operational':
            recs.append("Reinstall TimesFM model dependencies")
        
        if self.results['system_info'].get('cuda_available') == False:
            recs.append("Enable GPU runtime in Colab for faster predictions")
        
        if len(self.results['warnings']) > 3:
            recs.append("Review and resolve warnings for optimal performance")
        
        if 'yahoo_finance' in self.results['data_sources'] and self.results['data_sources']['yahoo_finance'] != 'operational':
            recs.append("Check internet connectivity for market data")
        
        latency = self.results['system_info'].get('prediction_latency', 0)
        if latency > 10:
            recs.append("Consider restarting runtime to improve performance")
        
        if self.results['system_info'].get('available_memory_gb', 0) < 2:
            recs.append("Free up memory or upgrade to Colab Pro")
        
        self.results['recommendations'] = recs
    
    def save_results(self, filename: str = None):
        """Save validation results to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"system_validation_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filename}")
        return filename


def main():
    """Main validation entry point."""
    validator = SystemValidator()
    
    try:
        # Run comprehensive validation
        results = validator.validate_all()
        
        # Save results
        validator.save_results()
        
        # Exit code based on results
        if results['overall_status'] in ['excellent', 'good']:
            print("\n🎉 System is ready for trading!")
            sys.exit(0)
        elif results['overall_status'] == 'fair':
            print("\n⚠️ System has issues but may be usable")
            sys.exit(1)
        else:
            print("\n❌ System has critical issues - do not trade")
            sys.exit(2)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Validation cancelled by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
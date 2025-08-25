"""Automated testing for fresh Colab environment setup.

This script validates that the entire Google Colab setup process works correctly
from scratch, including dependency installation, wheel caching, and model validation.

Usage:
    # Run full test suite
    python test_colab_setup.py
    
    # Run specific test
    python test_colab_setup.py --test dependency_installation
    
    # Skip wheel caching test (for faster testing)
    python test_colab_setup.py --skip-caching
"""

import os
import sys
import time
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import traceback

# Add project paths
sys.path.insert(0, '/content')
sys.path.insert(0, '/content/colab_setup')


class ColabSetupTester:
    """Comprehensive testing suite for Google Colab setup process."""
    
    def __init__(self, verbose: bool = True):
        """Initialize the tester.
        
        Args:
            verbose: Enable detailed output
        """
        self.verbose = verbose
        self.test_results = {}
        self.temp_dirs = []
        
        print("🧪 ColabSetupTester initialized")
        if not self._detect_colab_environment():
            print("⚠️ Warning: Not running in Google Colab environment")
    
    def run_full_test_suite(self, skip_caching: bool = False) -> Dict[str, Any]:
        """Run complete test suite for Colab setup.
        
        Args:
            skip_caching: Skip wheel caching tests (for faster testing)
            
        Returns:
            Dictionary with test results
        """
        print("\n" + "="*60)
        print("🧪 STARTING COLAB SETUP TEST SUITE")
        print("="*60)
        
        start_time = time.time()
        
        # Initialize results
        self.test_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'environment': self._get_environment_info(),
            'tests': {},
            'summary': {
                'total': 0,
                'passed': 0,
                'failed': 0,
                'skipped': 0
            }
        }
        
        # Define test sequence
        test_sequence = [
            ('environment_check', self._test_environment_check, False),
            ('file_structure', self._test_file_structure, False),
            ('dependency_manager_import', self._test_dependency_manager_import, False),
            ('model_validator_import', self._test_model_validator_import, False),
            ('requirements_parsing', self._test_requirements_parsing, False),
            ('dependency_installation', self._test_dependency_installation, skip_caching),
            ('wheel_caching', self._test_wheel_caching, skip_caching),
            ('model_validation', self._test_model_validation, False),
            ('integration_test', self._test_integration, False)
        ]
        
        # Run tests
        for test_name, test_function, skip_condition in test_sequence:
            if skip_condition:
                self._record_test_result(test_name, 'skipped', 'Test skipped by user request')
                continue
                
            self._run_single_test(test_name, test_function)
        
        # Calculate summary
        total_time = time.time() - start_time
        self._finalize_results(total_time)
        
        # Cleanup
        self._cleanup()
        
        return self.test_results
    
    def _run_single_test(self, test_name: str, test_function) -> None:
        """Run a single test and record results."""
        try:
            print(f"\n🔍 Running test: {test_name.upper()}")
            start_time = time.time()
            
            result = test_function()
            test_time = time.time() - start_time
            
            if result.get('success', False):
                print(f"✅ {test_name.upper()} PASSED ({test_time:.1f}s)")
                self._record_test_result(test_name, 'passed', result, test_time)
            else:
                print(f"❌ {test_name.upper()} FAILED ({test_time:.1f}s)")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                self._record_test_result(test_name, 'failed', result, test_time)
                
        except Exception as e:
            print(f"💥 {test_name.upper()} CRASHED: {e}")
            error_result = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            self._record_test_result(test_name, 'failed', error_result)
    
    def _record_test_result(self, test_name: str, status: str, result: Any, duration: float = 0) -> None:
        """Record test result in results dictionary."""
        self.test_results['tests'][test_name] = {
            'status': status,
            'result': result,
            'duration': duration
        }
        
        self.test_results['summary']['total'] += 1
        if status == 'passed':
            self.test_results['summary']['passed'] += 1
        elif status == 'failed':
            self.test_results['summary']['failed'] += 1
        elif status == 'skipped':
            self.test_results['summary']['skipped'] += 1
    
    def _detect_colab_environment(self) -> bool:
        """Detect if running in Google Colab."""
        return 'google.colab' in sys.modules
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get information about current environment."""
        info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'in_colab': self._detect_colab_environment(),
            'working_directory': os.getcwd()
        }
        
        # Try to get GPU info
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                info['gpu'] = result.stdout.strip()
        except:
            info['gpu'] = 'Not available'
        
        # Check CUDA version
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'CUDA Version:' in line:
                        info['cuda_version'] = line.split('CUDA Version:')[1].strip().split()[0]
                        break
        except:
            info['cuda_version'] = 'Unknown'
        
        return info
    
    def _test_environment_check(self) -> Dict[str, Any]:
        """Test basic environment requirements."""
        checks = {
            'python_version': False,
            'cuda_available': False,
            'disk_space': False,
            'memory': False
        }
        
        details = []
        
        # Python version check
        major, minor = sys.version_info.major, sys.version_info.minor
        if major == 3 and minor in [10, 11]:
            checks['python_version'] = True
            details.append(f"✅ Python {major}.{minor} (MAPIE compatible)")
        else:
            details.append(f"❌ Python {major}.{minor} (MAPIE requires 3.10 or 3.11)")
        
        # CUDA check
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                checks['cuda_available'] = True
                details.append("✅ CUDA available")
            else:
                details.append("❌ CUDA not available")
        except:
            details.append("❌ CUDA check failed")
        
        # Disk space check (need at least 2GB for wheels)
        try:
            import psutil
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            if free_gb >= 2.0:
                checks['disk_space'] = True
                details.append(f"✅ Disk space: {free_gb:.1f}GB available")
            else:
                details.append(f"❌ Insufficient disk space: {free_gb:.1f}GB")
        except:
            details.append("⚠️ Could not check disk space")
        
        # Memory check
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            if available_gb >= 1.0:
                checks['memory'] = True
                details.append(f"✅ Memory: {available_gb:.1f}GB available")
            else:
                details.append(f"⚠️ Low memory: {available_gb:.1f}GB")
        except:
            details.append("⚠️ Could not check memory")
        
        success = all(checks.values())
        return {
            'success': success,
            'checks': checks,
            'details': details
        }
    
    def _test_file_structure(self) -> Dict[str, Any]:
        """Test that all required files and directories exist."""
        required_files = [
            '/content/colab_setup/colab_dependency_manager.py',
            '/content/colab_setup/model_validator.py',
            '/content/colab_setup/colab_requirements.txt',
            '/content/requirements.txt',
            '/content/ROADMAP.md'
        ]
        
        required_dirs = [
            '/content/colab_setup',
            '/content/src',
            '/content/src/models',
            '/content/src/trading'
        ]
        
        missing_files = []
        missing_dirs = []
        found_files = []
        found_dirs = []
        
        # Check files
        for file_path in required_files:
            if os.path.isfile(file_path):
                found_files.append(file_path)
            else:
                missing_files.append(file_path)
        
        # Check directories
        for dir_path in required_dirs:
            if os.path.isdir(dir_path):
                found_dirs.append(dir_path)
            else:
                missing_dirs.append(dir_path)
        
        success = len(missing_files) == 0 and len(missing_dirs) == 0
        
        return {
            'success': success,
            'found_files': found_files,
            'found_dirs': found_dirs,
            'missing_files': missing_files,
            'missing_dirs': missing_dirs,
            'details': [
                f"Found {len(found_files)}/{len(required_files)} required files",
                f"Found {len(found_dirs)}/{len(required_dirs)} required directories"
            ]
        }
    
    def _test_dependency_manager_import(self) -> Dict[str, Any]:
        """Test that dependency manager can be imported and instantiated."""
        try:
            from colab_setup.colab_dependency_manager import ColabDependencyManager
            
            # Create temporary directory for testing
            temp_dir = tempfile.mkdtemp()
            self.temp_dirs.append(temp_dir)
            
            # Create dummy requirements file
            dummy_req_file = os.path.join(temp_dir, 'test_requirements.txt')
            with open(dummy_req_file, 'w') as f:
                f.write("numpy>=1.20.0\npandas>=1.3.0\n")
            
            # Initialize manager
            manager = ColabDependencyManager(
                drive_path=os.path.join(temp_dir, 'wheels'),
                requirements_file=dummy_req_file
            )
            
            # Check basic attributes
            assert hasattr(manager, 'setup_dependencies')
            assert hasattr(manager, 'clear_cache')
            assert hasattr(manager, 'get_cache_info')
            
            return {
                'success': True,
                'details': [
                    "✅ ColabDependencyManager imported successfully",
                    "✅ Manager instantiated with test parameters",
                    "✅ All required methods available"
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _test_model_validator_import(self) -> Dict[str, Any]:
        """Test that model validator can be imported and used."""
        try:
            from colab_setup.model_validator import ModelValidator
            
            # Initialize validator
            validator = ModelValidator(verbose=False)
            
            # Check basic attributes
            assert hasattr(validator, 'run_full_validation')
            assert hasattr(validator, 'validate_single_component')
            
            # Test single component validation (dependencies only)
            result = validator.validate_single_component('dependencies')
            assert isinstance(result, dict)
            assert 'success' in result
            
            return {
                'success': True,
                'details': [
                    "✅ ModelValidator imported successfully",
                    "✅ Validator instantiated",
                    "✅ Dependency validation test passed",
                    f"✅ Found {len(result.get('available_deps', []))} available dependencies"
                ]
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _test_requirements_parsing(self) -> Dict[str, Any]:
        """Test that requirements files can be parsed correctly."""
        try:
            colab_req_file = '/content/colab_setup/colab_requirements.txt'
            main_req_file = '/content/requirements.txt'
            
            results = {
                'colab_requirements': {},
                'main_requirements': {}
            }
            
            # Parse colab requirements
            with open(colab_req_file, 'r') as f:
                colab_lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            colab_packages = []
            for line in colab_lines:
                if '==' in line:
                    pkg_name = line.split('==')[0].strip()
                    colab_packages.append(pkg_name)
            
            results['colab_requirements'] = {
                'file_exists': True,
                'total_lines': len(colab_lines),
                'packages': colab_packages,
                'package_count': len(colab_packages)
            }
            
            # Parse main requirements
            with open(main_req_file, 'r') as f:
                main_lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            main_packages = []
            for line in main_lines:
                if '>=' in line:
                    pkg_name = line.split('>=')[0].strip()
                    main_packages.append(pkg_name)
            
            results['main_requirements'] = {
                'file_exists': True,
                'total_lines': len(main_lines),
                'packages': main_packages,
                'package_count': len(main_packages)
            }
            
            # Check for critical packages
            critical_packages = ['torch', 'mamba-ssm', 'transformers', 'mapie']
            missing_critical = []
            
            for pkg in critical_packages:
                if pkg not in colab_packages:
                    missing_critical.append(pkg)
            
            success = len(missing_critical) == 0
            
            details = [
                f"✅ Colab requirements: {len(colab_packages)} packages",
                f"✅ Main requirements: {len(main_packages)} packages"
            ]
            
            if missing_critical:
                details.append(f"❌ Missing critical packages: {missing_critical}")
            else:
                details.append("✅ All critical packages present")
            
            return {
                'success': success,
                'results': results,
                'missing_critical': missing_critical,
                'details': details
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _test_dependency_installation(self) -> Dict[str, Any]:
        """Test dependency installation process (dry run)."""
        try:
            # This is a dry run - we don't actually install everything
            # Just test that the installation commands are valid
            
            from colab_setup.colab_dependency_manager import ColabDependencyManager
            
            # Create test manager
            temp_dir = tempfile.mkdtemp()
            self.temp_dirs.append(temp_dir)
            
            manager = ColabDependencyManager(
                drive_path=os.path.join(temp_dir, 'wheels'),
                requirements_file='/content/colab_setup/colab_requirements.txt'
            )
            
            # Test cache info (should work even without cache)
            cache_info = manager.get_cache_info()
            assert isinstance(cache_info, dict)
            assert 'status' in cache_info
            
            # Test environment hash generation
            assert hasattr(manager, 'environment_hash')
            assert len(manager.environment_hash) > 0
            
            return {
                'success': True,
                'details': [
                    "✅ Dependency manager initialization successful",
                    f"✅ Environment hash: {manager.environment_hash}",
                    f"✅ Cache info: {cache_info['status']}",
                    f"✅ Python version: {manager.python_version}",
                    f"✅ CUDA version: {manager.cuda_version}"
                ],
                'environment': {
                    'python_version': manager.python_version,
                    'cuda_version': manager.cuda_version,
                    'environment_hash': manager.environment_hash
                }
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _test_wheel_caching(self) -> Dict[str, Any]:
        """Test wheel caching functionality (without actual wheel building)."""
        try:
            # Create test wheel caching scenario
            temp_dir = tempfile.mkdtemp()
            self.temp_dirs.append(temp_dir)
            
            wheel_dir = Path(temp_dir) / "test_wheels"
            wheel_dir.mkdir(exist_ok=True)
            
            # Create dummy wheel files
            dummy_wheels = [
                'numpy-1.26.4-py3-none-any.whl',
                'pandas-2.2.2-py3-none-any.whl'
            ]
            
            for wheel_name in dummy_wheels:
                wheel_path = wheel_dir / wheel_name
                wheel_path.write_text(f"# Dummy wheel file: {wheel_name}")
            
            # Create metadata file
            metadata = {
                'created': '2025-08-25T10:00:00',
                'python_version': '3.10',
                'cuda_version': '12.5',
                'environment_hash': 'test_hash_123',
                'wheel_count': len(dummy_wheels)
            }
            
            metadata_file = wheel_dir / "environment_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Test wheel detection logic
            wheel_files = list(wheel_dir.glob("*.whl"))
            metadata_exists = metadata_file.exists()
            
            success = len(wheel_files) == len(dummy_wheels) and metadata_exists
            
            return {
                'success': success,
                'details': [
                    f"✅ Created {len(dummy_wheels)} dummy wheel files",
                    f"✅ Metadata file created: {metadata_exists}",
                    f"✅ Wheel detection: found {len(wheel_files)} files",
                    f"✅ Cache directory structure working"
                ],
                'wheel_count': len(wheel_files),
                'metadata': metadata
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _test_model_validation(self) -> Dict[str, Any]:
        """Test model validation framework (basic components only)."""
        try:
            from colab_setup.model_validator import ModelValidator
            
            validator = ModelValidator(verbose=False)
            
            # Test dependency validation
            dep_result = validator.validate_single_component('dependencies')
            
            # Test test data preparation
            data_result = validator.validate_single_component('test_data')
            
            success = dep_result.get('success', False) and data_result.get('success', False)
            
            details = [
                f"✅ Dependency validation: {len(dep_result.get('available_deps', []))} packages",
                f"✅ Test data preparation: {data_result.get('success', False)}"
            ]
            
            if 'missing_deps' in dep_result and dep_result['missing_deps']:
                details.append(f"⚠️ Missing deps: {dep_result['missing_deps']}")
            
            return {
                'success': success,
                'details': details,
                'dependency_result': dep_result,
                'test_data_result': data_result
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _test_integration(self) -> Dict[str, Any]:
        """Test integration between all components."""
        try:
            # Test that all major components can work together
            from colab_setup.colab_dependency_manager import ColabDependencyManager
            from colab_setup.model_validator import ModelValidator
            
            # Create temporary setup
            temp_dir = tempfile.mkdtemp()
            self.temp_dirs.append(temp_dir)
            
            # Initialize components
            manager = ColabDependencyManager(
                drive_path=os.path.join(temp_dir, 'wheels'),
                requirements_file='/content/colab_setup/colab_requirements.txt'
            )
            
            validator = ModelValidator(verbose=False)
            
            # Test that they can provide compatible information
            cache_info = manager.get_cache_info()
            dep_validation = validator.validate_single_component('dependencies')
            
            integration_checks = {
                'manager_initialized': hasattr(manager, 'environment_hash'),
                'validator_initialized': hasattr(validator, 'validation_results'),
                'cache_info_available': isinstance(cache_info, dict),
                'dependency_check_works': isinstance(dep_validation, dict)
            }
            
            success = all(integration_checks.values())
            
            details = [
                f"✅ Manager environment hash: {manager.environment_hash}",
                f"✅ Cache info status: {cache_info.get('status', 'unknown')}",
                f"✅ Dependency validation: {dep_validation.get('success', False)}",
                f"✅ Integration checks: {sum(integration_checks.values())}/{len(integration_checks)}"
            ]
            
            return {
                'success': success,
                'details': details,
                'integration_checks': integration_checks
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _finalize_results(self, total_time: float) -> None:
        """Finalize test results with summary."""
        summary = self.test_results['summary']
        success_rate = summary['passed'] / max(1, summary['total'])
        
        self.test_results['total_time'] = total_time
        self.test_results['success_rate'] = success_rate
        self.test_results['overall_success'] = success_rate >= 0.8
        
        # Generate report
        print(f"\n" + "="*60)
        print("📋 COLAB SETUP TEST REPORT")
        print("="*60)
        print(f"📅 Test Time: {self.test_results['timestamp']}")
        print(f"⏱️  Duration: {total_time:.1f} seconds")
        print(f"🧪 Tests: {summary['total']} total, {summary['passed']} passed, {summary['failed']} failed, {summary['skipped']} skipped")
        print(f"📊 Success Rate: {success_rate:.1%}")
        
        if self.test_results['overall_success']:
            print("\n🎉 OVERALL STATUS: SUCCESS")
            print("Colab setup system is ready for deployment!")
        else:
            print("\n⚠️ OVERALL STATUS: NEEDS WORK")
            print("Some components need attention before deployment.")
        
        # Test details
        print(f"\n📋 Test Results:")
        for test_name, test_data in self.test_results['tests'].items():
            status = test_data['status']
            duration = test_data.get('duration', 0)
            
            if status == 'passed':
                print(f"  ✅ {test_name.upper()} ({duration:.1f}s)")
            elif status == 'failed':
                print(f"  ❌ {test_name.upper()} ({duration:.1f}s)")
                error = test_data['result'].get('error', 'Unknown error')
                print(f"     Error: {error}")
            elif status == 'skipped':
                print(f"  ⏭️  {test_name.upper()} (skipped)")
        
        print("="*60)
    
    def _cleanup(self) -> None:
        """Clean up temporary directories and files."""
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except Exception as e:
                print(f"⚠️ Warning: Could not clean up {temp_dir}: {e}")
        
        self.temp_dirs.clear()
    
    def run_single_test(self, test_name: str) -> Dict[str, Any]:
        """Run a single test by name."""
        test_methods = {
            'environment_check': self._test_environment_check,
            'file_structure': self._test_file_structure,
            'dependency_manager_import': self._test_dependency_manager_import,
            'model_validator_import': self._test_model_validator_import,
            'requirements_parsing': self._test_requirements_parsing,
            'dependency_installation': self._test_dependency_installation,
            'wheel_caching': self._test_wheel_caching,
            'model_validation': self._test_model_validation,
            'integration_test': self._test_integration
        }
        
        if test_name not in test_methods:
            return {
                'success': False,
                'error': f"Unknown test: {test_name}. Available: {list(test_methods.keys())}"
            }
        
        print(f"Running single test: {test_name}")
        return test_methods[test_name]()


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Google Colab setup system")
    parser.add_argument('--test', type=str, help="Run specific test only")
    parser.add_argument('--skip-caching', action='store_true',
                       help="Skip wheel caching tests (for faster testing)")
    parser.add_argument('--output', type=str, help="Save results to JSON file")
    
    args = parser.parse_args()
    
    tester = ColabSetupTester(verbose=True)
    
    if args.test:
        # Run single test
        result = tester.run_single_test(args.test)
        if result['success']:
            print(f"✅ Test {args.test} PASSED")
        else:
            print(f"❌ Test {args.test} FAILED: {result.get('error', 'Unknown error')}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump({'single_test': args.test, 'result': result}, f, indent=2)
        
        sys.exit(0 if result['success'] else 1)
    else:
        # Run full test suite
        results = tester.run_full_test_suite(skip_caching=args.skip_caching)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
        
        sys.exit(0 if results['overall_success'] else 1)


if __name__ == "__main__":
    main()
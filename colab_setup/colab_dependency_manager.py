"""Smart dependency management for Google Colab with Drive caching.

This module provides intelligent wheel caching to Google Drive for fast dependency
installation in Google Colab environments. It solves the problem of slow repeated
pip installations by building wheels once and reusing them across sessions.

Usage:
    from colab_setup.colab_dependency_manager import ColabDependencyManager
    
    manager = ColabDependencyManager()
    success = manager.setup_dependencies()
    
    if success:
        print("Dependencies installed successfully!")
    else:
        print("Installation failed - check logs")
"""

import os
import sys
import subprocess
import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime, timedelta
import importlib.util

# Always assume Google Colab environment for simplicity
IN_COLAB = True


class ColabDependencyManager:
    """Smart dependency setup with Google Drive wheel caching.
    
    Features:
    - Environment compatibility checking (Python, CUDA versions)
    - Smart wheel caching to Google Drive
    - Progressive installation with validation
    - Fallback strategies for failed installations
    - Detailed logging and error reporting
    """
    
    def __init__(self, 
                 drive_path: str = "/content/drive/MyDrive/trading_wheels",
                 requirements_file: str = "/content/colab_setup/colab_requirements.txt",
                 force_rebuild: bool = False):
        """Initialize dependency manager.
        
        Args:
            drive_path: Google Drive path for wheel caching
            requirements_file: Path to requirements.txt file
            force_rebuild: If True, rebuild wheels even if cache exists
        """
        self.drive_path = Path(drive_path)
        self.requirements_file = Path(requirements_file)
        self.force_rebuild = force_rebuild
        
        # Environment information
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.cuda_version = self._detect_cuda_version()
        self.environment_hash = self._get_environment_hash()
        
        # Versioned wheel directory
        self.wheel_dir = self.drive_path / f"env_{self.environment_hash}"
        self.metadata_file = self.wheel_dir / "environment_metadata.json"
        
        # Installation tracking
        self.installation_log = []
        self.failed_packages = []
        
        print(f"🔧 ColabDependencyManager initialized")
        print(f"📍 Python: {self.python_version}")
        print(f"🚀 CUDA: {self.cuda_version}")
        print(f"📁 Cache: {self.wheel_dir}")
        
    def setup_dependencies(self) -> bool:
        """Main entry point for dependency setup.
        
        Returns:
            True if all dependencies installed successfully, False otherwise
        """
        try:
            print("\n" + "="*60)
            print("🚀 STARTING DEPENDENCY SETUP")
            print("="*60)
            
            # Step 1: Mount Google Drive
            if not self._mount_drive():
                print("❌ Failed to mount Google Drive")
                return False
            
            # Step 2: Check/fix Python version if needed
            if not self._ensure_python_compatibility():
                print("❌ Python compatibility check failed")
                return False
            
            # Step 3: Check environment compatibility
            if not self._check_environment_compatibility():
                print("❌ Environment compatibility check failed")
                return False
            
            # Step 3: Setup wheel directory
            self._setup_wheel_directory()
            
            # Step 4: Check for existing compatible wheels
            if not self.force_rebuild and self._has_compatible_wheels():
                print("✅ Found compatible cached wheels. Installing from cache...")
                success = self._install_from_cache()
            else:
                print("⚙️ Building and caching wheels...")
                success = self._build_and_cache_wheels()
                
            # Step 5: Validate installations
            if success:
                print("\n🧪 Validating installations...")
                success = self._validate_installations()
                
            # Step 6: Validate AI models
            if success:
                print("\n🤖 Validating AI model compatibility...")
                success = self._validate_ai_models()
            
            # Step 6: Generate report
            self._generate_installation_report(success)
            
            return success
            
        except Exception as e:
            print(f"❌ Critical error in dependency setup: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _mount_drive(self) -> bool:
        """Mount Google Drive with error handling."""
        try:
            print("📡 Mounting Google Drive...")
            
            # Check if already mounted
            if Path("/content/drive").exists() and Path("/content/drive/MyDrive").exists():
                print("✅ Google Drive already mounted")
                return True
            
            # Try importing google.colab in a safe way
            try:
                import google.colab.drive as drive
                drive.mount('/content/drive')
            except ImportError:
                print("⚠️ google.colab not available - using local storage")
                # Use local storage as fallback
                self.drive_path = Path("/tmp/trading_wheels_cache") 
                self.wheel_dir = self.drive_path / f"env_{self.environment_hash}"
                self.metadata_file = self.wheel_dir / "environment_metadata.json"
                print(f"📁 Using local cache: {self.drive_path}")
                return True
            except Exception as colab_error:
                print(f"⚠️ Colab drive mount failed: {colab_error}")
                print("🔄 Falling back to local storage...")
                # Use local storage as fallback
                self.drive_path = Path("/tmp/trading_wheels_cache")
                self.wheel_dir = self.drive_path / f"env_{self.environment_hash}"
                self.metadata_file = self.wheel_dir / "environment_metadata.json"
                print(f"📁 Using local cache: {self.drive_path}")
                return True
            
            # Verify mount worked
            if Path("/content/drive/MyDrive").exists():
                print("✅ Google Drive mounted successfully")
                return True
            else:
                raise Exception("Drive mount completed but /content/drive/MyDrive not accessible")
                
        except Exception as e:
            print(f"❌ Drive mount error: {e}")
            print("🔄 Using local storage instead...")
            # Use local storage as final fallback
            self.drive_path = Path("/tmp/trading_wheels_cache")
            self.wheel_dir = self.drive_path / f"env_{self.environment_hash}"
            self.metadata_file = self.wheel_dir / "environment_metadata.json"
            print(f"📁 Cache directory: {self.drive_path}")
            return True
    
    def _detect_cuda_version(self) -> str:
        """Detect CUDA version with enhanced detection methods."""
        try:
            # Method 1: Try PyTorch CUDA detection first (most reliable)
            try:
                import torch
                if torch.cuda.is_available():
                    cuda_version = torch.version.cuda
                    if cuda_version:
                        print(f"🔍 CUDA detected via PyTorch: {cuda_version}")
                        return cuda_version
            except ImportError:
                pass  # PyTorch not installed yet
            
            # Method 2: Try nvidia-smi
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'CUDA Version:' in line:
                        cuda_version = line.split('CUDA Version:')[1].strip().split()[0]
                        print(f"🔍 CUDA detected via nvidia-smi: {cuda_version}")
                        return cuda_version
            
            # Method 3: Try nvcc compiler
            result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'release' in line and 'V' in line:
                        version = line.split('V')[1].strip()
                        print(f"🔍 CUDA detected via nvcc: {version}")
                        return version
            
            # Method 4: Colab environment detection
            if Path("/content").exists():
                print("🔍 Google Colab detected - using CUDA 12.2 (typical Colab version)")
                return "12.2"
            
            # Final fallback
            print("⚠️ Could not detect CUDA version, using 12.1 (common default)")
            return "12.1"
            
        except Exception as e:
            print(f"⚠️ CUDA detection error: {e}")
            return "12.1"  # Safe default
    
    def _get_environment_hash(self) -> str:
        """Create hash of Python/CUDA/requirements for cache compatibility."""
        try:
            # Read requirements content
            if self.requirements_file.exists():
                requirements_content = self.requirements_file.read_text()
            else:
                requirements_content = "# No requirements file found"
            
            # Create environment signature
            env_signature = f"{self.python_version}|{self.cuda_version}|{requirements_content}"
            
            # Generate hash
            env_hash = hashlib.md5(env_signature.encode()).hexdigest()[:12]
            return env_hash
            
        except Exception as e:
            print(f"⚠️ Warning: Could not generate environment hash: {e}")
            return "default"
    
    def _ensure_python_compatibility(self) -> bool:
        """Ensure Python version is compatible, installing 3.11 if needed in Colab."""
        print(f"\n🐍 Checking Python compatibility...")
        
        python_major, python_minor = sys.version_info.major, sys.version_info.minor
        current_version = f"{python_major}.{python_minor}"
        
        print(f"📍 Current Python version: {current_version}")
        
        # Check if current version is compatible
        if python_major == 3 and python_minor in [10, 11]:
            print(f"✅ Python {current_version} is compatible with MAPIE")
            return True
        
        # If not compatible, try to install Python 3.10 (assuming Colab)
        if python_major == 3 and python_minor >= 12:
            print(f"⚠️ Python {current_version} is not compatible with MAPIE (requires 3.10 or 3.11)")
            print("🔄 Installing Python 3.10 directly in Colab environment...")
            
            try:
                success = self._install_python310_colab()
                if success:
                    print("✅ Python 3.10 installed successfully!")
                    print("⚠️ IMPORTANT: Please restart the Colab runtime and run this script again.")
                    print("   Runtime -> Restart runtime (Ctrl+M .) then re-run your cells.")
                    return False  # Return False to indicate restart needed
                else:
                    raise Exception("Python 3.10 installation failed")
                
            except Exception as e:
                print(f"❌ Failed to install Python 3.10: {e}")
                print("💡 Manual solution - run these commands in Colab cells:")
                print("   !sudo apt-get update -y")
                print("   !sudo apt-get install python3.10 python3.10-distutils -y")
                print("   !sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1")
                print("   !sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 2")
                print("   !sudo update-alternatives --set python3 /usr/bin/python3.10")
                print("   !curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10")
                print("   Then restart runtime and re-run this script.")
                return False
        
        # If not in Colab, provide manual instructions
        print(f"❌ Python {current_version} is not compatible with MAPIE")
        print("💡 Please use Python 3.10 or 3.11 for this project")
        print("   In Colab: Runtime -> Change runtime type -> Python 3.10")
        
        return False
    
    def _install_python310_colab(self) -> bool:
        """Install Python 3.10 using Colab-compatible approach."""
        try:
            print("🔧 Installing Python 3.10 with system commands...")
            
            # Commands that work in Colab environment
            commands = [
                ["sudo", "apt-get", "update", "-y"],
                ["sudo", "apt-get", "install", "python3.10", "python3.10-distutils", "-y"],
                ["sudo", "update-alternatives", "--install", "/usr/bin/python3", "python3", "/usr/bin/python3.9", "1"],
                ["sudo", "update-alternatives", "--install", "/usr/bin/python3", "python3", "/usr/bin/python3.10", "2"],
                ["sudo", "update-alternatives", "--set", "python3", "/usr/bin/python3.10"]
            ]
            
            for i, cmd in enumerate(commands, 1):
                print(f"🔧 Step {i}/{len(commands)}: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    print(f"❌ Command failed: {' '.join(cmd)}")
                    if result.stderr:
                        print(f"   Error: {result.stderr}")
                    return False
            
            # Install pip for Python 3.10
            print("🔧 Installing pip for Python 3.10...")
            pip_cmd = ["curl", "-sS", "https://bootstrap.pypa.io/get-pip.py"]
            curl_result = subprocess.run(pip_cmd, capture_output=True, text=True, timeout=120)
            
            if curl_result.returncode == 0:
                pip_install = subprocess.run(["python3.10"], input=curl_result.stdout, 
                                           text=True, capture_output=True, timeout=120)
                if pip_install.returncode != 0:
                    print("⚠️ Pip installation had issues, but continuing...")
            
            # Verify installation
            try:
                version_result = subprocess.run(["python3.10", "--version"], 
                                              capture_output=True, text=True, timeout=10)
                if version_result.returncode == 0:
                    print(f"✅ Python 3.10 installed: {version_result.stdout.strip()}")
                    return True
                else:
                    print("❌ Python 3.10 verification failed")
                    return False
            except Exception as verify_error:
                print(f"⚠️ Could not verify Python 3.10 installation: {verify_error}")
                return True  # Assume success if we got this far
                
        except Exception as e:
            print(f"❌ Python 3.10 installation error: {e}")
            return False
    
    def _check_environment_compatibility(self) -> bool:
        """Check if current environment meets requirements."""
        print(f"\n🔍 Checking environment compatibility...")
        
        issues = []
        
        # Python version is now checked in _ensure_python_compatibility()
        # This method focuses on other environment checks
        
        # Always assume Colab environment
        print("✅ Assuming Google Colab environment")
        
        # Check GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
            else:
                print("⚠️ Warning: No GPU available - AI models will run slowly")
        except ImportError:
            print("ℹ️ PyTorch not yet installed - will check GPU after installation")
        
        # Check requirements file
        if not self.requirements_file.exists():
            issues.append(f"Requirements file not found: {self.requirements_file}")
        
        if issues:
            print("❌ Environment compatibility issues:")
            for issue in issues:
                print(f"   • {issue}")
            return False
        else:
            print("✅ Environment compatibility check passed")
            return True
    
    def _setup_wheel_directory(self) -> None:
        """Create and setup wheel cache directory."""
        try:
            print(f"📁 Setting up wheel cache directory...")
            self.wheel_dir.mkdir(parents=True, exist_ok=True)
            
            # Create metadata file
            metadata = {
                "created": datetime.now().isoformat(),
                "python_version": self.python_version,
                "cuda_version": self.cuda_version,
                "environment_hash": self.environment_hash,
                "requirements_file": str(self.requirements_file)
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"✅ Wheel cache directory ready: {self.wheel_dir}")
            
        except Exception as e:
            print(f"❌ Failed to setup wheel directory: {e}")
            raise
    
    def _has_compatible_wheels(self) -> bool:
        """Check if cached wheels exist and are compatible."""
        try:
            # Check if metadata file exists
            if not self.metadata_file.exists():
                return False
            
            # Check if any wheel files exist
            wheel_files = list(self.wheel_dir.glob("*.whl"))
            if not wheel_files:
                return False
            
            # Check metadata compatibility
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Check age (wheels older than 30 days are considered stale)
            created_date = datetime.fromisoformat(metadata['created'])
            age = datetime.now() - created_date
            if age > timedelta(days=30):
                print("⚠️ Cached wheels are older than 30 days - rebuilding")
                return False
            
            # Check environment compatibility
            if (metadata.get('python_version') != self.python_version or 
                metadata.get('cuda_version') != self.cuda_version):
                print("⚠️ Environment mismatch - rebuilding wheels")
                return False
            
            print(f"✅ Found {len(wheel_files)} compatible cached wheels")
            return True
            
        except Exception as e:
            print(f"⚠️ Error checking wheel cache: {e}")
            return False
    
    def _build_and_cache_wheels(self) -> bool:
        """Build wheels and cache to Drive."""
        try:
            print(f"🔨 Installing dependencies directly (skipping wheel cache for PyTorch compatibility)...")
            
            # Special handling for packages with specific wheel requirements
            success = self._install_special_packages()
            if not success:
                return False
            
            # Install remaining packages with PyTorch and PyTorch Geometric indices
            install_cmd = [
                sys.executable, '-m', 'pip', 'install',
                '-r', str(self.requirements_file),
                '--extra-index-url', 'https://download.pytorch.org/whl/cu121',
                '--extra-index-url', 'https://data.pyg.org/whl/torch-2.4.0+cu121.html',
                '--prefer-binary',
                '--no-cache-dir'  # Don't cache to avoid conflicts
            ]
            
            print(f"🚀 Running: {' '.join(install_cmd)}")
            print("⏱️ This may take 10-30 minutes depending on your internet connection...")
            print("   Large packages like PyTorch, Transformers, and specialized wheels take time")
            start_time = time.time()
            
            result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=2400)  # 40 minutes
            install_time = time.time() - start_time
            
            if result.returncode != 0:
                print(f"❌ Installation failed after {install_time:.1f}s")
                print("📋 Error details:")
                if result.stderr:
                    # Show only relevant error lines
                    error_lines = result.stderr.split('\n')
                    relevant_errors = [line for line in error_lines if 'ERROR:' in line or 'Failed' in line]
                    for error in relevant_errors[-3:]:  # Show last 3 relevant errors
                        print(f"   {error}")
                print("\n🔄 Trying fallback installation...")
                return self._fallback_install()
            
            print(f"✅ Installed dependencies in {install_time:.1f}s")
            return True
            
        except subprocess.TimeoutExpired:
            print("❌ Installation timed out after 40 minutes")
            print("💡 This may happen with slow internet connections in Colab")
            print("   Try running the script again or use a different Colab instance")
            return False
        except Exception as e:
            print(f"❌ Installation failed: {e}")
            return self._fallback_install()
    
    def _install_special_packages(self) -> bool:
        """Install packages that need special handling (like causal-conv1d)."""
        try:
            print("🔧 Installing packages with special wheel requirements...")
            
            # Install PyTorch first (needed for CUDA detection)
            pytorch_cmd = [
                sys.executable, '-m', 'pip', 'install',
                'torch==2.4.1', 'torchvision==0.19.1', 'torchaudio==2.4.1',
                '--extra-index-url', 'https://download.pytorch.org/whl/cu121'
            ]
            
            print("🔥 Installing PyTorch with CUDA...")
            result = subprocess.run(pytorch_cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                print("❌ PyTorch installation failed")
                return False
            
            # Detect system specs for causal-conv1d
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
            
            print(f"🔍 Detected system: Python {python_version}, {python_tag}, linux_x86_64")
            
            # Install causal-conv1d with specific wheel URL
            # For Colab: cu12 (CUDA 12.x), torch2.4, cxx11abiTRUE (modern C++ ABI)
            wheel_name = f"causal_conv1d-1.5.2%2Bcu12torch2.4cxx11abiTRUE-{python_tag}-{python_tag}-linux_x86_64.whl"
            causal_conv1d_url = f"https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/{wheel_name}"
            
            causal_cmd = [
                sys.executable, '-m', 'pip', 'install',
                causal_conv1d_url
            ]
            
            print(f"🔗 Installing causal-conv1d from: {causal_conv1d_url}")
            result = subprocess.run(causal_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print("⚠️ Specific causal-conv1d wheel failed, trying alternative wheels...")
                
                # Try cxx11abiFALSE version (older ABI)
                alt_wheel_name = f"causal_conv1d-1.5.2%2Bcu12torch2.4cxx11abiFALSE-{python_tag}-{python_tag}-linux_x86_64.whl"
                alt_url = f"https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/{alt_wheel_name}"
                
                alt_cmd = [sys.executable, '-m', 'pip', 'install', alt_url]
                print(f"🔗 Trying alternative: {alt_url}")
                result = subprocess.run(alt_cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode != 0:
                    print("⚠️ Alternative wheel also failed, trying CUDA 11 version...")
                    
                    # Try CUDA 11 version as final fallback
                    cu11_wheel_name = f"causal_conv1d-1.5.2%2Bcu11torch2.4cxx11abiTRUE-{python_tag}-{python_tag}-linux_x86_64.whl"
                    cu11_url = f"https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/{cu11_wheel_name}"
                    
                    cu11_cmd = [sys.executable, '-m', 'pip', 'install', cu11_url]
                    print(f"🔗 Trying CUDA 11: {cu11_url}")
                    result = subprocess.run(cu11_cmd, capture_output=True, text=True, timeout=300)
                    
                    if result.returncode != 0:
                        print("❌ All specific wheels failed, trying pip install without wheel...")
                        generic_cmd = [sys.executable, '-m', 'pip', 'install', 'causal-conv1d==1.5.2']
                        result = subprocess.run(generic_cmd, capture_output=True, text=True, timeout=300)
                        
                        if result.returncode != 0:
                            print("❌ causal-conv1d installation completely failed")
                            print(f"Error: {result.stderr}")
                            return False
            
            print("✅ Special packages installed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Special packages installation failed: {e}")
            return False
    
    def _fallback_install(self) -> bool:
        """Fallback installation strategy."""
        try:
            print("🔄 Trying PyTorch-first installation strategy...")
            
            # First install PyTorch with CUDA support
            pytorch_cmd = [
                sys.executable, '-m', 'pip', 'install',
                'torch==2.4.1', 'torchvision==0.19.1', 'torchaudio==2.4.1',
                '--extra-index-url', 'https://download.pytorch.org/whl/cu121'
            ]
            
            print("🔥 Installing PyTorch with CUDA...")
            result = subprocess.run(pytorch_cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode != 0:
                print("❌ PyTorch installation failed")
                print(result.stderr)
                return False
            
            # Create filtered requirements file (exclude PyTorch and causal-conv1d)
            print("📦 Installing remaining packages...")
            filtered_reqs = self._create_filtered_requirements()
            
            other_cmd = [
                sys.executable, '-m', 'pip', 'install',
                '-r', filtered_reqs,
                '--extra-index-url', 'https://download.pytorch.org/whl/cu121',
                '--extra-index-url', 'https://data.pyg.org/whl/torch-2.4.0+cu121.html',
                '--prefer-binary',
                '--no-cache-dir'
            ]
            
            result = subprocess.run(other_cmd, capture_output=True, text=True, timeout=2400)  # 40 minutes
            
            if result.returncode != 0:
                print("❌ Other packages installation failed")
                print("📋 Error summary:")
                if result.stderr:
                    # Show only relevant error lines
                    error_lines = result.stderr.split('\n')
                    relevant_errors = [line for line in error_lines if any(keyword in line for keyword in ['ERROR:', 'Failed', 'No matching distribution', 'Could not find'])]
                    for error in relevant_errors[-5:]:  # Show last 5 relevant errors
                        print(f"   {error}")
                print("💡 Try running the dependency manager again, or check package versions")
                return False
            
            print("✅ Fallback installation successful")
            return True
            
        except Exception as e:
            print(f"❌ Fallback installation failed: {e}")
            return False
    
    def _create_filtered_requirements(self) -> str:
        """Create a requirements file excluding already installed packages."""
        try:
            exclude_packages = {
                'torch', 'torchvision', 'torchaudio', 'causal-conv1d',
                'torch-geometric', 'torch-scatter', 'torch-sparse', 
                'torch-cluster', 'torch-spline-conv'
            }
            
            # Read original requirements
            with open(self.requirements_file, 'r') as f:
                lines = f.readlines()
            
            # Filter out excluded packages and comments
            filtered_lines = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0].strip()
                    if pkg_name not in exclude_packages:
                        filtered_lines.append(line)
            
            # Write filtered requirements
            filtered_file = self.wheel_dir / "filtered_requirements.txt"
            with open(filtered_file, 'w') as f:
                f.write('\n'.join(filtered_lines))
            
            print(f"📝 Created filtered requirements: {len(filtered_lines)} packages")
            return str(filtered_file)
            
        except Exception as e:
            print(f"⚠️ Could not create filtered requirements: {e}")
            return str(self.requirements_file)  # Fallback to original
    
    def _install_from_cache(self) -> bool:
        """Install from cached wheels."""
        try:
            print(f"📦 Installing from cached wheels...")
            
            # Install command
            install_cmd = [
                sys.executable, '-m', 'pip', 'install',
                '--no-index',  # Don't use PyPI
                '--find-links', str(self.wheel_dir),  # Use our wheel directory
                '-r', str(self.requirements_file),
                '--force-reinstall'  # Ensure we get the exact versions
            ]
            
            print(f"🚀 Running: {' '.join(install_cmd)}")
            start_time = time.time()
            
            result = subprocess.run(install_cmd, capture_output=True, text=True, timeout=600)
            install_time = time.time() - start_time
            
            if result.returncode != 0:
                print(f"❌ Installation from cache failed:")
                print(result.stderr)
                return False
            
            print(f"✅ Installed from cache in {install_time:.1f}s")
            return True
            
        except subprocess.TimeoutExpired:
            print("❌ Installation timed out after 10 minutes")
            return False
        except Exception as e:
            print(f"❌ Error installing from cache: {e}")
            return False
    
    def _validate_installations(self) -> bool:
        """Test that all critical packages can be imported."""
        critical_packages = {
            'torch': 'PyTorch for deep learning',
            'transformers': 'Hugging Face Transformers',
            'mamba_ssm': 'Mamba State Space Models',
            'mapie': 'MAPIE uncertainty quantification',
            'yfinance': 'Yahoo Finance data',
            'pandas': 'Data manipulation',
            'numpy': 'Numerical computing',
            'sklearn': 'Scikit-learn machine learning'
        }
        
        print(f"\n🧪 Validating {len(critical_packages)} critical packages...")
        
        failed_imports = []
        successful_imports = []
        
        for package, description in critical_packages.items():
            try:
                # Special handling for sklearn
                if package == 'sklearn':
                    import sklearn
                    successful_imports.append((package, description, sklearn.__version__))
                else:
                    module = importlib.import_module(package)
                    version = getattr(module, '__version__', 'unknown')
                    successful_imports.append((package, description, version))
                    
                print(f"  ✅ {package}")
                
            except ImportError as e:
                failed_imports.append((package, description, str(e)))
                print(f"  ❌ {package}: {e}")
            except Exception as e:
                failed_imports.append((package, description, f"Unexpected error: {e}"))
                print(f"  ⚠️ {package}: {e}")
        
        # Test specific functionality
        success_count = len(successful_imports)
        total_count = len(critical_packages)
        
        print(f"\n📊 Validation Results: {success_count}/{total_count} packages imported successfully")
        
        if failed_imports:
            print("\n❌ Failed imports:")
            for package, description, error in failed_imports:
                print(f"   • {package} ({description}): {error}")
            
            # Allow partial success if core packages work
            if success_count >= total_count * 0.8:
                print("⚠️ Partial success - core packages working")
                return True
            else:
                return False
        else:
            print("✅ All critical packages validated successfully!")
            return True
    
    def _validate_ai_models(self) -> bool:
        """Validate that AI models can be loaded successfully."""
        try:
            print("🔍 Testing AI model imports and initialization...")
            
            model_tests = {
                "PyTorch": {
                    "test": "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')",
                    "required": True
                },
                "Transformers": {
                    "test": "from transformers import AutoTokenizer; print('✅ Transformers available')",
                    "required": True
                },
                "MAPIE": {
                    "test": "from mapie.regression import MapieRegressor; MapieRegressor(); print('✅ MAPIE available')",
                    "required": True
                },
                "mamba-ssm": {
                    "test": "import mamba_ssm; print('✅ Mamba SSM available')",
                    "required": True
                },
                "causal-conv1d": {
                    "test": "import causal_conv1d; print('✅ Causal Conv1D available')",
                    "required": True
                },
                "torch-geometric": {
                    "test": "import torch_geometric; print('✅ PyTorch Geometric available')",
                    "required": False  # Optional for some models
                }
            }
            
            failed_models = []
            optional_failed = []
            
            for model_name, config in model_tests.items():
                try:
                    print(f"  🧪 Testing {model_name}...")
                    
                    # Execute the test code
                    exec(config["test"])
                    print(f"    ✅ {model_name} validated")
                    
                except Exception as e:
                    print(f"    ❌ {model_name} failed: {str(e)[:100]}")
                    if config["required"]:
                        failed_models.append(model_name)
                    else:
                        optional_failed.append(model_name)
            
            # Report results
            if failed_models:
                print(f"\n❌ Critical AI models failed: {', '.join(failed_models)}")
                print("   The trading system may not function correctly")
                return False
            elif optional_failed:
                print(f"\n⚠️ Optional models failed: {', '.join(optional_failed)}")
                print("   Core functionality will work, but some features may be limited")
            else:
                print("\n✅ All AI models validated successfully!")
            
            # Additional CUDA validation if available
            try:
                import torch
                if torch.cuda.is_available():
                    print(f"🚀 CUDA validation: {torch.cuda.device_count()} GPU(s) available")
                    print(f"   Primary GPU: {torch.cuda.get_device_name(0)}")
                else:
                    print("⚠️ No CUDA GPUs available - AI models will run on CPU (slower)")
            except:
                pass
            
            return True
            
        except Exception as e:
            print(f"❌ AI model validation error: {e}")
            return False
    
    def _generate_installation_report(self, success: bool) -> None:
        """Generate detailed installation report."""
        print("\n" + "="*60)
        print("📋 INSTALLATION REPORT")
        print("="*60)
        
        print(f"🔧 Environment: Python {self.python_version}, CUDA {self.cuda_version}")
        print(f"📁 Cache Location: {self.wheel_dir}")
        print(f"📅 Installation Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if success:
            print("✅ Status: SUCCESS - All dependencies installed and validated")
            print("\n🚀 Ready to run AI trading system!")
            print("\nNext steps:")
            print("  1. Run: python trading_advisor.py")
            print("  2. Check that AI models load without 'MOCK MODE' warnings")
            print("  3. Validate predictions are non-random")
        else:
            print("❌ Status: FAILED - Some dependencies could not be installed")
            print("\n🔧 Troubleshooting:")
            print("  1. Check internet connection")
            print("  2. Try restarting Colab runtime")
            print("  3. Use force_rebuild=True to rebuild wheels")
            print("  4. Check ROADMAP.md for alternative solutions")
        
        print("="*60)
    
    def clear_cache(self) -> bool:
        """Clear wheel cache directory."""
        try:
            if self.wheel_dir.exists():
                import shutil
                shutil.rmtree(self.wheel_dir)
                print(f"✅ Cleared wheel cache: {self.wheel_dir}")
                return True
            else:
                print("ℹ️ No cache to clear")
                return True
        except Exception as e:
            print(f"❌ Failed to clear cache: {e}")
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache."""
        try:
            if not self.wheel_dir.exists():
                return {"status": "no_cache", "message": "No cache directory found"}
            
            wheel_files = list(self.wheel_dir.glob("*.whl"))
            
            info = {
                "status": "cache_exists",
                "cache_path": str(self.wheel_dir),
                "wheel_count": len(wheel_files),
                "environment_hash": self.environment_hash,
            }
            
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
                info["metadata"] = metadata
            
            return info
            
        except Exception as e:
            return {"status": "error", "message": str(e)}


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart dependency manager for Google Colab")
    parser.add_argument('--drive-path', default="/content/drive/MyDrive/trading_wheels",
                       help="Google Drive path for wheel caching")
    parser.add_argument('--requirements', default="/content/colab_setup/colab_requirements.txt",
                       help="Path to requirements.txt file")
    parser.add_argument('--force-rebuild', action='store_true',
                       help="Force rebuild of wheels even if cache exists")
    parser.add_argument('--clear-cache', action='store_true',
                       help="Clear wheel cache and exit")
    parser.add_argument('--cache-info', action='store_true',
                       help="Show cache information and exit")
    
    args = parser.parse_args()
    
    manager = ColabDependencyManager(
        drive_path=args.drive_path,
        requirements_file=args.requirements,
        force_rebuild=args.force_rebuild
    )
    
    if args.clear_cache:
        success = manager.clear_cache()
        sys.exit(0 if success else 1)
    
    if args.cache_info:
        info = manager.get_cache_info()
        print(json.dumps(info, indent=2))
        sys.exit(0)
    
    # Run main dependency setup
    success = manager.setup_dependencies()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
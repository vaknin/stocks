#!/usr/bin/env python3
"""
Quick verification that all Google Colab setup scripts work correctly.
Run this to test the entire workflow without actually being in Colab.
"""
import sys
import os
from pathlib import Path
import traceback

def test_all_components():
    """Test all colab setup components"""
    results = {
        "dependency_manager": False,
        "model_validator": False, 
        "test_framework": False,
        "requirements_file": False
    }
    
    print("🧪 Testing Google Colab Setup Components...")
    print("="*50)
    
    # Test 1: Dependency Manager
    try:
        import colab_dependency_manager
        manager = colab_dependency_manager.ColabDependencyManager(
            drive_path='/tmp/test_wheels',
            requirements_file='colab_requirements.txt'
        )
        env_hash = manager._get_environment_hash()
        results["dependency_manager"] = True
        print("✅ ColabDependencyManager: PASS")
    except Exception as e:
        print(f"❌ ColabDependencyManager: FAIL - {e}")
    
    # Test 2: Model Validator
    try:
        import model_validator
        validator = model_validator.ModelValidator()
        results["model_validator"] = True
        print("✅ ModelValidator: PASS")
    except Exception as e:
        print(f"❌ ModelValidator: FAIL - {e}")
    
    # Test 3: Test Framework
    try:
        import test_colab_setup
        tester = test_colab_setup.ColabSetupTester()
        results["test_framework"] = True
        print("✅ ColabSetupTester: PASS")
    except Exception as e:
        print(f"❌ ColabSetupTester: FAIL - {e}")
    
    # Test 4: Requirements File
    try:
        req_file = Path("colab_requirements.txt")
        if req_file.exists():
            content = req_file.read_text()
            if "torch==" in content and "mapie==" in content:
                results["requirements_file"] = True
                print("✅ colab_requirements.txt: PASS")
            else:
                print("❌ colab_requirements.txt: FAIL - Missing key packages")
        else:
            print("❌ colab_requirements.txt: FAIL - File not found")
    except Exception as e:
        print(f"❌ colab_requirements.txt: FAIL - {e}")
    
    # Summary
    print("\n" + "="*50)
    print("📊 SUMMARY")
    print("="*50)
    
    passed = sum(results.values())
    total = len(results)
    
    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component}")
    
    print(f"\n🎯 Overall: {passed}/{total} components working")
    
    if passed == total:
        print("🟢 ALL TESTS PASSED - Google Colab setup is ready!")
        print("\n📋 Next Steps:")
        print("1. Upload this folder to Google Colab")
        print("2. Run: python colab_setup/colab_dependency_manager.py")
        print("3. Run: python colab_setup/model_validator.py")
        print("4. Start using the trading system!")
        return True
    else:
        print("🟡 SOME ISSUES FOUND - Please fix the failing components")
        return False

if __name__ == "__main__":
    # Change to the colab_setup directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    success = test_all_components()
    sys.exit(0 if success else 1)
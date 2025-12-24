#!/usr/bin/env python3
"""
Setup Checker - Verifies all dependencies are installed correctly
"""

import sys
import subprocess
from pathlib import Path

def check_module(module_name, package_name=None):
    """Check if a Python module is installed"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} - Run: pip install {package_name}")
        return False

def check_ollama():
    """Check if Ollama is installed and running"""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        
        if result.returncode == 0:
            print("✅ Ollama is installed")
            
            # Check if qwen3:1.7b is pulled
            if 'qwen3:1.7b' in result.stdout.lower():
                print("✅ qwen3:1.7b model is downloaded")
                return True
            else:
                print("⚠️  qwen3:1.7b model not found")
                print("   Run: ollama pull qwen3:1.7b")
                return False
        else:
            print("❌ Ollama not responding")
            print("   Run: ollama serve")
            return False
    except FileNotFoundError:
        print("❌ Ollama not installed")
        print("   Download from: https://ollama.ai")
        return False
    except subprocess.TimeoutExpired:
        print("❌ Ollama not responding (timeout)")
        print("   Run: ollama serve")
        return False
    except Exception as e:
        print(f"⚠️  Could not check Ollama: {e}")
        return False

def check_files():
    """Check if all required files are present"""
    required_files = [
        'meeting_notes_generator.py',
        'realtime_meeting_recorder.py',
        'meeting_recorder_gui.py',
        'requirements.txt'
    ]
    
    print("\n📁 Checking files...")
    all_present = True
    
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - Missing!")
            all_present = False
    
    return all_present

def main():
    print("="*60)
    print("🔍 MEETING NOTES GENERATOR - SETUP CHECKER")
    print("="*60)
    
    print("\n📦 Checking Python packages...")
    
    modules = [
        ('faster_whisper', 'faster-whisper'),
        ('langchain_ollama', 'langchain-ollama'),
        ('reportlab', 'reportlab'),
        ('loguru', 'loguru'),
        ('pyaudio', 'pyaudio'),
        ('watchdog', 'watchdog'),
        ('flask', 'flask'),
    ]
    
    all_installed = True
    for module, package in modules:
        if not check_module(module, package):
            all_installed = False
    
    # Check tkinter separately (usually comes with Python)
    try:
        import tkinter
        print("✅ tkinter (GUI support)")
    except ImportError:
        print("❌ tkinter - Install: sudo apt-get install python3-tk (Linux)")
        all_installed = False
    
    # Check Ollama
    print("\n🤖 Checking Ollama...")
    ollama_ok = check_ollama()
    
    # Check files
    files_ok = check_files()
    
    # Summary
    print("\n" + "="*60)
    if all_installed and ollama_ok and files_ok:
        print("✅ ALL CHECKS PASSED!")
        print("\nYou're ready to run:")
        print("  python meeting_recorder_gui.py")
        print("\nOr CLI version:")
        print("  python realtime_meeting_recorder.py")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("\nFix the issues above, then run this checker again:")
        print("  python setup_checker.py")
        
        if not all_installed:
            print("\n💡 Install missing packages:")
            print("  pip install -r requirements.txt")
        
        if not ollama_ok:
            print("\n💡 Setup Ollama:")
            print("  1. Download from https://ollama.ai")
            print("  2. Run: ollama serve")
            print("  3. Run: ollama pull qwen3:1.7b")
        
        if not files_ok:
            print("\n💡 Download missing files from the chat!")
    
    print("="*60)
    
    return 0 if (all_installed and ollama_ok and files_ok) else 1

if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
سكريبت فحص البيئة الافتراضية
يفحص جميع المكتبات المهمة لمشروع Riva Mini
"""

import sys
import subprocess

print("=" * 60)
print("🔍 فحص البيئة الافتراضية")
print("=" * 60)

# فحص إصدار Python
print(f"\n✅ Python Version: {sys.version}")
print(f"📍 Python Path: {sys.executable}")

# المكتبات المهمة للمشروع
required_packages = [
    'fastapi',
    'uvicorn',
    'nemo_toolkit',
    'torch',
    'torchaudio',
    'numpy',
    'librosa',
    'python-multipart',
    'pydantic'
]

print("\n" + "=" * 60)
print("📦 المكتبات المثبتة:")
print("=" * 60)

installed = {}
missing = []

for package in required_packages:
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', package],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            # استخراج الإصدار
            for line in result.stdout.split('\n'):
                if line.startswith('Version:'):
                    version = line.split('Version:')[1].strip()
                    installed[package] = version
                    print(f"✅ {package:20s} -> {version}")
                    break
        else:
            missing.append(package)
            print(f"❌ {package:20s} -> غير مثبت")
    except Exception as e:
        missing.append(package)
        print(f"❌ {package:20s} -> خطأ في الفحص: {e}")

# عرض الملخص
print("\n" + "=" * 60)
print("📊 الملخص:")
print("=" * 60)
print(f"✅ مثبت: {len(installed)}/{len(required_packages)}")
print(f"❌ ناقص: {len(missing)}/{len(required_packages)}")

if missing:
    print("\n⚠️  المكتبات الناقصة:")
    for pkg in missing:
        print(f"   - {pkg}")
    
    print("\n💡 لتثبيت المكتبات الناقصة، شغّل:")
    print(f"   pip install {' '.join(missing)}")

# فحص CUDA (إذا كان متوفر)
print("\n" + "=" * 60)
print("🎮 فحص GPU/CUDA:")
print("=" * 60)
try:
    import torch
    print(f"✅ PyTorch Version: {torch.__version__}")
    print(f"🎮 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 CUDA Version: {torch.version.cuda}")
        print(f"🎮 GPU Count: {torch.cuda.device_count()}")
        print(f"🎮 GPU Name: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("❌ PyTorch غير مثبت")

print("\n" + "=" * 60)
print("✨ انتهى الفحص!")
print("=" * 60)

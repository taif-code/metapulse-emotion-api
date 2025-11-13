#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
سكريبت لفحص موديل NeMo وقراءة الـ config
"""

import yaml
import json
from pathlib import Path


def print_section(title):
    """طباعة عنوان قسم"""
    print("\n" + "=" * 60)
    print(f"📋 {title}")
    print("=" * 60)

def inspect_yaml_file(yaml_path):
    """فحص ملف YAML"""
    print_section(f"فحص ملف: {yaml_path}")
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("✅ تم قراءة الملف بنجاح!")
        print(f"📊 عدد الأقسام الرئيسية: {len(config)}")
        
        # طباعة الأقسام الرئيسية
        print("\n🔑 الأقسام الرئيسية:")
        for key in config.keys():
            print(f"   - {key}")
        
        # البحث عن Labels
        print_section("البحث عن Emotion Labels")
        labels_found = False
        
        def search_labels(obj, path=""):
            """بحث عن labels في جميع مستويات الـ config"""
            nonlocal labels_found
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    if 'label' in key.lower():
                        print(f"🎯 وُجد في: {current_path}")
                        print(f"   القيمة: {value}")
                        labels_found = True
                    search_labels(value, current_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    search_labels(item, f"{path}[{i}]")
        
        search_labels(config)
        
        if not labels_found:
            print("⚠️  لم يتم العثور على 'labels' في الـ config")
        
        # معلومات أخرى مهمة
        print_section("معلومات الموديل")
        
        # Sample Rate
        if 'sample_rate' in config:
            print(f"🎵 Sample Rate: {config['sample_rate']}")
        elif 'preprocessor' in config and 'sample_rate' in config['preprocessor']:
            print(f"🎵 Sample Rate: {config['preprocessor']['sample_rate']}")
        
        # Model Type
        if 'model' in config:
            if '_target_' in config['model']:
                print(f"🤖 Model Type: {config['model']['_target_']}")
        
        # Decoder/Head info
        if 'decoder' in config:
            print(f"🧠 Decoder: {config['decoder'].get('_target_', 'N/A')}")
            if 'num_classes' in config['decoder']:
                print(f"📊 Number of Classes: {config['decoder']['num_classes']}")
        
        # طباعة الـ config كامل (اختياري)
        print_section("Config الكامل (JSON)")
        print(json.dumps(config, indent=2, ensure_ascii=False))
        
        return config
        
    except FileNotFoundError:
        print(f"❌ الملف غير موجود: {yaml_path}")
        return None
    except Exception as e:
        print(f"❌ خطأ في قراءة الملف: {str(e)}")
        return None

def inspect_nemo_model(model_path):
    """فحص موديل NeMo"""
    print_section(f"فحص موديل NeMo: {model_path}")
    
    try:
        from nemo.collections.asr.models import EncDecClassificationModel
        
        print("🔄 جاري تحميل الموديل...")
        model = EncDecClassificationModel.restore_from(model_path)
        
        print("✅ تم تحميل الموديل بنجاح!")
        
        # معلومات الموديل
        if hasattr(model, 'cfg'):
            cfg = model.cfg
            print("\n📊 معلومات من Config:")
            
            # Labels
            if hasattr(cfg, 'labels'):
                print(f"🎯 Labels: {list(cfg.labels)}")
            
            # Sample Rate
            if hasattr(cfg, 'sample_rate'):
                print(f"🎵 Sample Rate: {cfg.sample_rate}")
            
            # Preprocessor
            if hasattr(cfg, 'preprocessor'):
                print(f"🔧 Preprocessor: {cfg.preprocessor._target_}")
            
            # Decoder
            if hasattr(cfg, 'decoder'):
                print(f"🧠 Decoder: {cfg.decoder._target_}")
                if hasattr(cfg.decoder, 'num_classes'):
                    print(f"📊 Num Classes: {cfg.decoder.num_classes}")
        
        return model
        
    except Exception as e:
        print(f"❌ خطأ في تحميل الموديل: {str(e)}")
        return None

def main():
    """الدالة الرئيسية"""
    print("=" * 60)
    print("🔍 فحص موديل NeMo والـ Config")
    print("=" * 60)
    
    # البحث عن ملفات YAML
    yaml_files = list(Path('.').glob('*.yaml')) + list(Path('.').glob('*.yml'))
    
    if yaml_files:
        print(f"\n✅ تم العثور على {len(yaml_files)} ملف YAML:")
        for f in yaml_files:
            print(f"   - {f.name}")
        
        # فحص كل ملف
        for yaml_file in yaml_files:
            inspect_yaml_file(yaml_file)
    else:
        print("\n⚠️  لم يتم العثور على ملفات YAML في المجلد الحالي")
    
    # البحث عن ملف .nemo
    nemo_files = list(Path('.').glob('*.nemo'))
    
    if nemo_files:
        print(f"\n✅ تم العثور على {len(nemo_files)} ملف .nemo:")
        for f in nemo_files:
            print(f"   - {f.name}")
        
        # فحص أول ملف
        if nemo_files:
            print(f"\n🔍 سيتم فحص: {nemo_files[0].name}")
            inspect_nemo_model(str(nemo_files[0]))
    else:
        print("\n⚠️  لم يتم العثور على ملفات .nemo في المجلد الحالي")
    
    print("\n" + "=" * 60)
    print("✨ انتهى الفحص!")
    print("=" * 60)

if __name__ == "__main__":
    main()
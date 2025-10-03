import sys
import subprocess
import importlib.util

# List of required packages/modules for gpu_image_blazer.py
REQUIRED = {
    'os': 'built-in',
    'argparse': 'built-in',
    'PIL': 'pillow (pip install pillow)',
    'torch': 'torch (pip install torch)',
    'torch.nn.functional': 'torch (already in torch)',
    'torch.utils.data': 'torch (already in torch)',
    'torchvision.transforms': 'torchvision (pip install torchvision)',
    'time': 'built-in',
    'warnings': 'built-in',
}

def check_package(package_name, install_cmd=None):
    """Check if module can be imported; return True if ok."""
    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        try:
            importlib.import_module(package_name)
            return True
        except Exception:
            return False
    return False

def main():
    print("🔥 GPU Image Blazer Dependency Checker (Park Bench Edition) 🔥")
    print("=" * 50)
    
    missing = []
    for mod, info in REQUIRED.items():
        if mod == 'os' or mod == 'argparse' or mod == 'time' or mod == 'warnings':
            print(f"✅ {mod}: Built-in (good to go)")
        elif 'torch' in mod:
            if check_package('torch'):
                print(f"✅ {mod}: Torch loaded (your 3050's ready)")
            else:
                missing.append(f"torch - {info}")
                print(f"❌ {mod}: Missing - {info}")
        elif mod == 'PIL':
            if check_package('PIL'):
                print(f"✅ {mod}: Pillow loaded")
            else:
                missing.append(f"PIL (pillow) - {info}")
                print(f"❌ {mod}: Missing - {info}")
        elif mod == 'torchvision.transforms':
            if check_package('torchvision') and check_package('torchvision.transforms'):
                print(f"✅ {mod}: Torchvision loaded")
            else:
                missing.append(f"torchvision - pip install torchvision")
                print(f"❌ {mod}: Missing - pip install torchvision")
        else:
            print(f"⚠️  {mod}: Check torch install")
    
    if missing:
        print("\n❌ Missing bits—fire these when hardline drops:")
        for m in missing:
            print(f"   pip install {m.split(' - ')[1]}")
        print("\nTip: For torch/torchvision, use: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
    else:
        print("\n✅ All deps pukka—blaze away with gpu_image_blazer.py!")
    
    # Bonus: Quick torch GPU check
    try:
        import torch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎯 Torch device: {device} (3050 vibes: {'Yes!' if torch.cuda.is_available() else 'CPU fallback'})")
    except ImportError:
        print("⚠️  Torch not loaded—install first.")

if __name__ == "__main__":
    main()

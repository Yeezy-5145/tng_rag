"""
Setup verification script
Checks if all dependencies are installed correctly
"""

import sys


def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        print(f"✅ {package_name or module_name}")
        return True
    except ImportError:
        print(f"❌ {package_name or module_name} - Not installed")
        return False


def main():
    """Check all required dependencies"""
    print("Checking dependencies...")
    print("=" * 50)

    required = [
        ("chromadb", "ChromaDB"),
        ("sentence_transformers", "sentence-transformers"),
        ("transformers", "transformers"),
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("bs4", "beautifulsoup4"),
        ("requests", "requests"),
    ]

    optional = [
        ("ollama", "ollama (optional)"),
    ]

    all_ok = True
    for module, name in required:
        if not check_import(module, name):
            all_ok = False

    print("\nOptional dependencies:")
    for module, name in optional:
        check_import(module, name)

    print("\n" + "=" * 50)
    if all_ok:
        print("✅ All required dependencies are installed!")
        print("\nNext steps:")
        print("1. Run: python scraper.py (to collect FAQ data)")
        print("2. Run: python chat.py (to test the system)")
    else:
        print("❌ Some dependencies are missing!")
        print("Please run: pip install -r requirements.txt")
        sys.exit(1)


if __name__ == "__main__":
    main()
